"""实验9：lr5e-4/ep30 配置的 5-seed MLP 集成 + 融合重调 -> v3 提交分数。

阶段A：顺序协议（前80%训练）5 seed 训练，评估单模型与集成的 model-only 分数，
       并网格重搜三信号融合权重。
阶段B：全量数据重训 5 seed 集成，生成 test 三路分数与融合结果。
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import load_data, load_name_vectors, pad_symptoms  # noqa: E402
from metrics import f1_at_5  # noqa: E402
from final_pipeline import build_statistics, build_prior_features, shrink_scores, cond_scores, normalize_rows, HerbNet  # noqa: E402

SEEDS = (2026, 7, 42, 123, 777)


def train_eval(seed, train_ids, va_ids, ctx, epochs=30, lr=5e-4, eval_each=True, final_epochs=15):
    """在给定训练集上训练。eval_each=True 时逐轮早停选最佳；False 时训练 final_epochs 轮取最终模型。"""
    features, labels, herbs, symptoms, sym_bge = ctx['features'], ctx['labels'], ctx['herbs'], ctx['symptoms'], ctx['sym_bge']
    n_herbs, n_symptoms = len(herbs), len(symptoms)
    pop, sym2herb, sym_count, combo2herb = ctx['stats']

    prior_train = build_prior_features(train_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs)
    prior_val = build_prior_features(va_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs)
    x_train = pad_symptoms(features, train_ids, 'cpu')
    x_val = pad_symptoms(features, va_ids, 'cpu')
    p_train = torch.from_numpy(prior_train)
    p_val = torch.from_numpy(prior_val)
    y_train = torch.zeros((len(train_ids), n_herbs))
    for row, u in enumerate(train_ids):
        y_train[row, sorted(labels[u])] = 1.0

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HerbNet(n_symptoms, n_herbs, sym_bge, in_prior_dim=prior_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)

    n_ep = epochs if eval_each else final_epochs
    best_f1, best_state, best_ep = 0.0, None, 0
    for ep in range(1, n_ep + 1):
        model.train()
        perm = rng.permutation(len(train_ids))
        for st in range(0, len(perm), 256):
            idx = torch.as_tensor(perm[st:st + 256], dtype=torch.long)
            loss = loss_fn(model(x_train[idx], p_train[idx]), y_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if eval_each:
            model.eval()
            with torch.inference_mode():
                logits = model(x_val, p_val).numpy()
            pred = {u: np.argsort(-logits[i], kind='stable')[:5].tolist() for i, u in enumerate(va_ids)}
            f1 = f1_at_5(pred, ctx['va'])
            if f1 > best_f1:
                best_f1, best_ep = f1, ep
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            best_ep = ep
    if eval_each:
        model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        logits = model(x_val, p_val).numpy()
    return logits, best_f1, best_ep


def main():
    torch.set_num_threads(8)
    features, labels, test, herbs, symptoms = load_data(ROOT / 'data')
    n_herbs, n_symptoms = len(herbs), len(symptoms)
    all_train = sorted(labels)
    tr_ids, va_ids = all_train[:21610], all_train[21610:]
    va = {u: sorted(labels[u]) for u in va_ids}

    sym_bge = load_name_vectors(ROOT / 'features' / 'symptom_bge.npz',
                                symptoms, 'symptom_ids', 'symptom_names')
    sym_bge = torch.nn.functional.pad(sym_bge, (0, 0, 1, 0))

    t0 = time.time()
    # ---- 阶段A：顺序协议 ----
    stats = build_statistics(tr_ids, features, labels)
    ctx = dict(features=features, labels=labels, herbs=herbs, symptoms=symptoms,
               sym_bge=sym_bge, stats=stats, va=va)

    logits_list, f1s = [], []
    for seed in SEEDS:
        lg, f1, ep = train_eval(seed, tr_ids, va_ids, ctx)
        logits_list.append(lg)
        f1s.append(f1)
        print(f'seed={seed}: best {f1:.4f} @ep{ep}  ({time.time()-t0:.0f}s)', flush=True)
    ens = np.mean(logits_list, axis=0)
    pred = {u: np.argsort(-ens[i], kind='stable')[:5].tolist() for i, u in enumerate(va_ids)}
    f1_ens = f1_at_5(pred, va)
    print(f'5-seed ensemble model-only: {f1_ens:.4f}')

    stat_va = shrink_scores(va_ids, features, *stats, n_herbs=n_herbs, tau=30.0) \
        if False else shrink_scores(va_ids, features, stats[0], stats[1], stats[2], stats[3], n_herbs, 30.0)
    cond_va = cond_scores(va_ids, features, stats[1], stats[2], n_herbs)

    stat_n, cond_n, ens_n = normalize_rows(stat_va), normalize_rows(cond_va), normalize_rows(ens)
    best = (None, 0.0)
    for w_s in (0.3, 0.4, 0.5, 0.6):
        for w_m in (0.2, 0.3, 0.4, 0.5):
            for w_c in (0.0, 0.1, 0.2, 0.3):
                if abs(w_s + w_m + w_c - 1.0) > 1e-9:
                    continue
                sc = w_s * stat_n + w_m * ens_n + w_c * cond_n
                pred = {u: np.argsort(-sc[i], kind='stable')[:5].tolist() for i, u in enumerate(va_ids)}
                f1 = f1_at_5(pred, va)
                if f1 > best[1]:
                    best = ((w_s, w_m, w_c), f1)
    (w_s, w_m, w_c), f1_best = best
    print(f'best fusion: stat={w_s} model={w_m} cond={w_c}  F1@5 = {f1_best:.4f}')

    # ---- 阶段B：全量重训 5 seed（无早停，固定轮数 = 阶段A最佳轮数中位数）----
    stats_full = build_statistics(all_train, features, labels)
    ctx_full = dict(features=features, labels=labels, herbs=herbs, symptoms=symptoms,
                    sym_bge=sym_bge, stats=stats_full, va=va)
    test_ids = sorted(test)
    logits_full = []
    fixed_ep = 15  # 阶段A best ep: 19/11/11/15/15 的中位数
    for seed in SEEDS:
        lg, f1, ep = train_eval(seed, all_train, test_ids, ctx_full, eval_each=False, final_epochs=fixed_ep)
        logits_full.append(lg)
        print(f'full seed={seed} done ({time.time()-t0:.0f}s)', flush=True)
    ens_te = np.mean(logits_full, axis=0)

    stat_te = shrink_scores(test_ids, features, stats_full[0], stats_full[1], stats_full[2], stats_full[3], n_herbs, 30.0)
    cond_te = cond_scores(test_ids, features, stats_full[1], stats_full[2], n_herbs)
    fused_te = w_s * normalize_rows(stat_te) + w_m * normalize_rows(ens_te) + w_c * normalize_rows(cond_te)

    out = ROOT / 'experiments' / 'artifacts_v3'
    out.mkdir(exist_ok=True)
    np.save(out / 'fused_test.npy', fused_te)
    np.save(out / 'test_ids.npy', np.array(test_ids))
    print('v3 saved.', f'total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
