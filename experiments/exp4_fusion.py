"""实验4：收缩参数细搜 + 分布函数变体 + 与 MLP logits 的多信号融合。"""
import sys
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import load_data, split_train, pad_symptoms  # noqa: E402
from metrics import f1_at_5  # noqa: E402
from exp1_prior_mlp import build_prior_features, HerbNet  # noqa: E402

SEED = 2026


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(8)

    data_dir = ROOT / 'data'
    features, labels, test, herbs, symptoms = load_data(data_dir)
    n_herbs = len(herbs)

    train_ids, val_ids = split_train(labels, ratio=0.2, seed=SEED)
    va = {u: sorted(labels[u]) for u in val_ids}

    pop = Counter()
    sym2herb = defaultdict(Counter)
    sym_count = Counter()
    combo2herb = defaultdict(Counter)
    for u in train_ids:
        pos = sorted(labels[u])
        pop.update(pos)
        combo2herb[tuple(sorted(set(features[u])))].update(pos)
        for s in set(features[u]):
            sym_count[s] += 1
            for h in pos:
                sym2herb[s][h] += 1

    freq = np.zeros(n_herbs, dtype=np.float32)
    for h, c in pop.items():
        freq[h] = c
    pop_prior = freq / (freq.max() + 1e-9)

    def cond_arr(uid):
        arr = np.zeros(n_herbs, dtype=np.float32)
        for s in set(features[uid]):
            for h, c in sym2herb[s].items():
                arr[h] += c / sym_count[s]
        return arr

    def combo_arr(uid, transform):
        c = combo2herb.get(tuple(sorted(set(features[u] if False else features[uid]))))
        if c is None:
            return None, 0
        arr = np.zeros(n_herbs, dtype=np.float32)
        mx = max(c.values())
        for h, v in c.items():
            if transform == 'log':
                arr[h] = math.log(1 + v) / math.log(1 + mx)
            elif transform == 'raw':
                arr[h] = v / mx
            elif transform == 'sqrt':
                arr[h] = math.sqrt(v) / math.sqrt(mx)
            elif transform == 'prob':
                arr[h] = v / sum(c.values())
        return arr, sum(c.values())

    val_cond = np.stack([cond_arr(u) for u in val_ids])
    val_cond_n = val_cond / np.maximum(val_cond.max(axis=1, keepdims=True), 1e-9)

    # --- tau 细搜 + 变体（基于 per-record 循环太慢，向量化评估）---
    best_cfg = None
    for transform in ('log', 'raw', 'sqrt', 'prob'):
        rows = [combo_arr(u, transform) for u in val_ids]
        combo_mat = np.stack([r[0] if r[0] is not None else np.zeros(n_herbs, dtype=np.float32) for r in rows])
        seen = np.array([r[1] for r in rows], dtype=np.float32)
        has = seen > 0
        for tau in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
            alpha = (seen / (seen + tau))[:, None]
            alpha[~has] = 0.0
            sc = alpha * combo_mat + (1 - alpha) * val_cond_n
            pred = {u: np.argsort(-sc[i], kind='stable')[:5].tolist() for i, u in enumerate(val_ids)}
            f1 = f1_at_5(pred, va)
            print(f'{transform} tau={tau:4.1f}  F1@5 = {f1:.4f}')
            if best_cfg is None or f1 > best_cfg[0]:
                best_cfg = (f1, transform, tau)

    f1_best, transform_best, tau_best = best_cfg
    print(f'best: {transform_best} tau={tau_best} F1={f1_best:.4f}')

    # --- 加载 exp1 MLP 验证 logits ---
    art = torch.load(ROOT / 'experiments' / 'artifacts_v1' / 'model.pt', weights_only=True)
    sym_bge = torch.load(ROOT / 'experiments' / 'artifacts_v1' / 'symptom_bge.pt', weights_only=True) \
        if (ROOT / 'experiments' / 'artifacts_v1' / 'symptom_bge.pt').exists() else None
    if sym_bge is None:
        from data import load_name_vectors
        sym_bge = load_name_vectors(ROOT / 'features' / 'symptom_bge.npz',
                                    symptoms, 'symptom_ids', 'symptom_names')
        sym_bge = torch.nn.functional.pad(sym_bge, (0, 0, 1, 0))
        torch.save(sym_bge, ROOT / 'experiments' / 'artifacts_v1' / 'symptom_bge.pt')

    x_va = pad_symptoms(features, val_ids, 'cpu')
    x_tr_prior_cache = None
    # exp1 的先验特征必须与训练时同一构建（由训练部分统计）——重新构建
    x_tr_prior = build_prior_features(train_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs)
    x_va_prior = build_prior_features(val_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs)
    model = HerbNet(len(symptoms), n_herbs, sym_bge, in_prior_dim=x_tr_prior.shape[1]).to('cpu')
    model.load_state_dict(art['state_dict'])
    model.eval()
    with torch.inference_mode():
        logits_va = model(x_va, torch.from_numpy(x_va_prior)).numpy()
    m = logits_va.min(axis=1, keepdims=True)
    M = logits_va.max(axis=1, keepdims=True)
    norm_model = (logits_va - m) / np.maximum(M - m, 1e-9)

    # 最佳统计分数
    rows = [combo_arr(u, transform_best) for u in val_ids]
    combo_mat = np.stack([r[0] if r[0] is not None else np.zeros(n_herbs, dtype=np.float32) for r in rows])
    seen = np.array([r[1] for r in rows], dtype=np.float32)
    alpha = (seen / (seen + tau_best))[:, None]
    stat_sc = alpha * combo_mat + (1 - alpha) * val_cond_n
    stat_n = stat_sc / np.maximum(stat_sc.max(axis=1, keepdims=True), 1e-9)

    # cond 分数也做一路
    cond_n = val_cond_n

    # --- 多信号融合：stat + model + cond ---
    for w_m in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        for w_c in (0.0, 0.1, 0.2, 0.3):
            if w_m + w_c > 1.0:
                continue
            w_s = 1 - w_m - w_c
            sc = w_s * stat_n + w_m * norm_model + w_c * cond_n
            pred = {u: np.argsort(-sc[i], kind='stable')[:5].tolist() for i, u in enumerate(val_ids)}
            print(f'blend stat={w_s:.1f} model={w_m:.1f} cond={w_c:.1f}  F1@5 = {f1_at_5(pred, va):.4f}')


if __name__ == '__main__':
    main()
