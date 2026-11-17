"""实验12：15 成员集成 + rank/min-max 融合对比（修复 rank 方向 bug）。

顺序协议（前80%训练，后20%评估）。所有成员的验证 logits 落盘，融合调参免重训。
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

# 15 成员：exp10 的 10 个 + 5 个新增（全部 lr5e-4/dp0.1 与 dp0.2 混合）
MEMBERS = [(2026, 5e-4, 0.1), (7, 5e-4, 0.1), (42, 5e-4, 0.1), (123, 5e-4, 0.1), (777, 5e-4, 0.1),
           (314, 5e-4, 0.2), (159, 5e-4, 0.2), (265, 5e-4, 0.2),
           (358, 1e-3, 0.1), (979, 1e-3, 0.1),
           (1111, 5e-4, 0.1), (2222, 5e-4, 0.1), (3333, 5e-4, 0.2), (4444, 5e-4, 0.1), (5555, 5e-4, 0.15)]


def rank_norm(mat):
    """逐记录排名归一化：分数越高值越大（1.0 = 第一名）。"""
    n_rows, n_cols = mat.shape
    order = np.argsort(mat, axis=1, kind='stable')  # 升序：order[i,0] 是最低分
    ranks = np.empty_like(order)
    rows = np.arange(n_rows)[:, None]
    ranks[rows, order] = np.arange(1, n_cols + 1)[None, :]  # 最低分 -> rank 1，最高分 -> rank 805
    return ranks / n_cols


def main():
    torch.set_num_threads(8)
    features, labels, test, herbs, symptoms = load_data(ROOT / 'data')
    n_herbs, n_symptoms = len(herbs), len(symptoms)
    all_train = sorted(labels)
    tr_ids, va_ids = all_train[:21610], all_train[21610:]
    va = {u: sorted(labels[u]) for u in va_ids}

    sym_bge = load_name_vectors(ROOT / 'features' / 'symptom_bge.npz',
                                symptoms, 'symptom_ids', 'symptom_names')
    sym_bge_np = sym_bge.numpy()
    sym_bge_np = sym_bge_np / np.maximum(np.linalg.norm(sym_bge_np, axis=1, keepdims=True), 1e-9)
    sym_bge = torch.nn.functional.pad(sym_bge, (0, 0, 1, 0))
    herb_bge = load_name_vectors(ROOT / 'features' / 'herb_bge.npz',
                                 herbs, 'item_ids', 'herb_names').numpy()
    herb_bge = herb_bge / np.linalg.norm(herb_bge, axis=1, keepdims=True)

    t0 = time.time()
    stats = build_statistics(tr_ids, features, labels)
    pop, sym2herb, sym_count, combo2herb = stats
    prior_a = build_prior_features(tr_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs, herb_bge, sym_bge_np)
    prior_b = build_prior_features(va_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs, herb_bge, sym_bge_np)
    x_a = pad_symptoms(features, tr_ids, 'cpu')
    x_b = pad_symptoms(features, va_ids, 'cpu')
    p_a, p_b = torch.from_numpy(prior_a), torch.from_numpy(prior_b)
    y_a = torch.zeros((len(tr_ids), n_herbs))
    for row, u in enumerate(tr_ids):
        y_a[row, sorted(labels[u])] = 1.0

    logits_list, f1s = [], []
    for seed, lr, dp in MEMBERS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = HerbNet(n_symptoms, n_herbs, sym_bge, in_prior_dim=prior_a.shape[1], dropout=dp)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(seed)
        best_f1, best_state = 0.0, None
        for ep in range(1, 31):
            model.train()
            perm = rng.permutation(len(tr_ids))
            for st in range(0, len(perm), 256):
                idx = torch.as_tensor(perm[st:st + 256], dtype=torch.long)
                loss = loss_fn(model(x_a[idx], p_a[idx]), y_a[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            model.eval()
            with torch.inference_mode():
                logits = model(x_b, p_b).numpy()
            pred = {u: np.argsort(-logits[i], kind='stable')[:5].tolist() for i, u in enumerate(va_ids)}
            f1 = f1_at_5(pred, va)
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        with torch.inference_mode():
            logits_list.append(model(x_b, p_b).numpy())
        f1s.append(best_f1)
        print(f'seed={seed} lr={lr} dp={dp}: best={best_f1:.4f} ({time.time()-t0:.0f}s)', flush=True)

    logits_arr = np.stack(logits_list)  # [15, 5402, 805]
    out = ROOT / 'experiments' / 'artifacts_v5'
    out.mkdir(exist_ok=True)
    np.save(out / 'val_logits_15.npy', logits_arr)

    # 不同集成规模
    for k in (5, 8, 10, 12, 15):
        ens = logits_arr[:k].mean(axis=0)
        f1 = f1_at_5({u: np.argsort(-ens[i], kind='stable')[:5].tolist() for i, u in enumerate(va_ids)}, va)
        print(f'ensemble top-{k} members model-only: {f1:.4f}')
    ens = logits_arr.mean(axis=0)
    f1_ens = f1_at_5({u: np.argsort(-ens[i], kind='stable')[:5].tolist() for i, u in enumerate(va_ids)}, va)
    print(f'15-member ensemble model-only: {f1_ens:.4f}')

    stat_va = shrink_scores(va_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs, 30.0)
    cond_va = cond_scores(va_ids, features, sym2herb, sym_count, n_herbs)

    # min-max 与 rank 两种融合方式 × 权重网格
    for norm_name, fn in (('min-max', normalize_rows), ('rank', rank_norm)):
        best = (None, 0.0)
        for w_s in (0.2, 0.3, 0.4):
            for w_m in (0.4, 0.5, 0.6, 0.7):
                for w_c in (0.0, 0.1, 0.2):
                    if abs(w_s + w_m + w_c - 1.0) > 1e-9:
                        continue
                    sc = w_s * fn(stat_va) + w_m * fn(ens) + w_c * fn(cond_va)
                    pred = {u: np.argsort(-sc[i], kind='stable')[:5].tolist() for i, u in enumerate(va_ids)}
                    f1 = f1_at_5(pred, va)
                    if f1 > best[1]:
                        best = ((w_s, w_m, w_c), f1)
        print(f'{norm_name} best fusion: {best[0]}  F1@5 = {best[1]:.4f}')


if __name__ == '__main__':
    main()
