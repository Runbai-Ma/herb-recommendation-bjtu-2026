"""实验10：语义匹配特征 + 10 模型多样性集成 + 融合重搜 -> v4。

新增特征：sym_bge 均值与 herb_bge 的点积（805 维跨塔语义匹配）。
集成配方：5x(lr5e-4,dp0.1) + 3x(lr5e-4,dp0.2) + 2x(lr1e-3,dp0.1)，30 轮早停。
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

SEEDS = [(2026, 5e-4, 0.1), (7, 5e-4, 0.1), (42, 5e-4, 0.1), (123, 5e-4, 0.1), (777, 5e-4, 0.1),
         (314, 5e-4, 0.2), (159, 5e-4, 0.2), (265, 5e-4, 0.2),
         (358, 1e-3, 0.1), (979, 1e-3, 0.1)]


def build_prior_v2(user_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs, herb_bge, sym_bge_np):
    """v2 先验 = v1 先验(3x805) + 症状均值与药名向量的语义匹配(805)。"""
    base = build_prior_features(user_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs)
    n = len(user_ids)
    sem = np.zeros((n, n_herbs), dtype=np.float32)
    for row, u in enumerate(user_ids):
        syms = sorted(set(features[u]))
        mean_vec = sym_bge_np[syms].mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            sem[row] = herb_bge @ (mean_vec / norm)
    return np.concatenate([base, sem], axis=1)


def main():
    torch.set_num_threads(8)
    features, labels, test, herbs, symptoms = load_data(ROOT / 'data')
    n_herbs, n_symptoms = len(herbs), len(symptoms)
    all_train = sorted(labels)
    tr_ids, va_ids = all_train[:21610], all_train[21610:]
    va = {u: sorted(labels[u]) for u in va_ids}

    sym_bge = load_name_vectors(ROOT / 'features' / 'symptom_bge.npz',
                                symptoms, 'symptom_ids', 'symptom_names')
    sym_bge_padded = torch.nn.functional.pad(sym_bge, (0, 0, 1, 0))
    herb_bge = load_name_vectors(ROOT / 'features' / 'herb_bge.npz',
                                 herbs, 'item_ids', 'herb_names').numpy()
    herb_bge = herb_bge / np.linalg.norm(herb_bge, axis=1, keepdims=True)
    sym_bge_np = sym_bge.numpy()
    sym_bge_np = sym_bge_np / np.maximum(np.linalg.norm(sym_bge_np, axis=1, keepdims=True), 1e-9)

    t0 = time.time()

    def run_stage(stage_ids, eval_ids, stats, tag, use_sem=True, epochs=30, early_stop=True):
        pop, sym2herb, sym_count, combo2herb = stats
        if use_sem:
            prior_a = build_prior_v2(stage_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs, herb_bge, sym_bge_np)
            prior_b = build_prior_v2(eval_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs, herb_bge, sym_bge_np)
        else:
            prior_a = build_prior_features(stage_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs)
            prior_b = build_prior_features(eval_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs)
        x_a = pad_symptoms(features, stage_ids, 'cpu')
        x_b = pad_symptoms(features, eval_ids, 'cpu')
        p_a, p_b = torch.from_numpy(prior_a), torch.from_numpy(prior_b)
        y_a = torch.zeros((len(stage_ids), n_herbs))
        for row, u in enumerate(stage_ids):
            y_a[row, sorted(labels[u])] = 1.0

        has_truth = set(eval_ids) <= set(labels)  # 只有训练集内的记录才有标签
        va_set = {u: sorted(labels[u]) for u in eval_ids} if has_truth else None

        logits_list, best_f1s = [], []
        for seed, lr, dp in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = HerbNet(n_symptoms, n_herbs, sym_bge_padded, in_prior_dim=prior_a.shape[1], dropout=dp)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.BCEWithLogitsLoss()
            rng = np.random.default_rng(seed)
            best_f1, best_state = 0.0, None
            for ep in range(1, epochs + 1):
                model.train()
                perm = rng.permutation(len(stage_ids))
                for st in range(0, len(perm), 256):
                    idx = torch.as_tensor(perm[st:st + 256], dtype=torch.long)
                    loss = loss_fn(model(x_a[idx], p_a[idx]), y_a[idx])
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                if early_stop and has_truth:
                    model.eval()
                    with torch.inference_mode():
                        logits = model(x_b, p_b).numpy()
                    pred = {u: np.argsort(-logits[i], kind='stable')[:5].tolist() for i, u in enumerate(eval_ids)}
                    f1 = f1_at_5(pred, va_set)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if early_stop and has_truth:
                model.load_state_dict(best_state)
            model.eval()
            with torch.inference_mode():
                logits_list.append(model(x_b, p_b).numpy())
            best_f1s.append(best_f1)
            print(f'{tag} seed={seed} lr={lr} dp={dp}: best={best_f1:.4f}' if has_truth
                  else f'{tag} seed={seed} done ({time.time()-t0:.0f}s)', flush=True)
        return np.mean(logits_list, axis=0), best_f1s

    # ---- 阶段A：顺序协议 ----
    stats = build_statistics(tr_ids, features, labels)
    ens_va, f1s = run_stage(tr_ids, va_ids, stats, 'A')
    if f1s:
        print(f'A: 10-model ensemble (sem feat) model-only = {f1_at_5({u: np.argsort(-ens_va[i], kind="stable")[:5].tolist() for i, u in enumerate(va_ids)}, va):.4f}')

    stat_va = shrink_scores(va_ids, features, stats[0], stats[1], stats[2], stats[3], n_herbs, 30.0)
    cond_va = cond_scores(va_ids, features, stats[1], stats[2], n_herbs)
    stat_n, cond_n, ens_n = normalize_rows(stat_va), normalize_rows(cond_va), normalize_rows(ens_va)

    best = (None, 0.0)
    for w_s in (0.3, 0.4, 0.5):
        for w_m in (0.3, 0.4, 0.5, 0.6):
            for w_c in (0.0, 0.1, 0.2):
                if abs(w_s + w_m + w_c - 1.0) > 1e-9:
                    continue
                sc = w_s * stat_n + w_m * ens_n + w_c * cond_n
                pred = {u: np.argsort(-sc[i], kind='stable')[:5].tolist() for i, u in enumerate(va_ids)}
                f1 = f1_at_5(pred, va)
                if f1 > best[1]:
                    best = ((w_s, w_m, w_c), f1)
    (w_s, w_m, w_c), f1_best = best
    print(f'best fusion: stat={w_s} model={w_m} cond={w_c}  F1@5 = {f1_best:.4f}')

    # ---- 阶段B：全量重训（无早停，固定 15 轮 = v3 相同协议）----
    stats_full = build_statistics(all_train, features, labels)
    test_ids = sorted(test)
    ens_te, _ = run_stage(all_train, test_ids, stats_full, 'B', use_sem=True, epochs=15, early_stop=False)
    stat_te = shrink_scores(test_ids, features, stats_full[0], stats_full[1], stats_full[2], stats_full[3], n_herbs, 30.0)
    cond_te = cond_scores(test_ids, features, stats_full[1], stats_full[2], n_herbs)
    fused_te = w_s * normalize_rows(stat_te) + w_m * normalize_rows(ens_te) + w_c * normalize_rows(cond_te)

    out = ROOT / 'experiments' / 'artifacts_v4'
    out.mkdir(exist_ok=True)
    np.save(out / 'fused_test.npy', fused_te)
    np.save(out / 'test_ids.npy', np.array(test_ids))
    print('v4 saved.', f'total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
