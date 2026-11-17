"""实验5：miss 路径改进 + 药语义平滑。

  A. 层次回退：完整组合 -> 2-症状子集组合 -> cond-sum
  B. 药语义平滑：seen 稀少的药向 BGE 语义近邻药扩散分数
  C. A/B 组合 + 三信号融合
"""
import sys
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import load_data, split_train, load_name_vectors  # noqa: E402
from metrics import f1_at_5  # noqa: E402

SEED = 2026


def main():
    np.random.seed(SEED)
    torch.set_num_threads(8)

    data_dir = ROOT / 'data'
    features, labels, test, herbs, symptoms = load_data(data_dir)
    n_herbs, n_symptoms = len(herbs), len(symptoms)

    train_ids, val_ids = split_train(labels, ratio=0.2, seed=SEED)
    va = {u: sorted(labels[u]) for u in val_ids}

    pop = Counter()
    sym2herb = defaultdict(Counter)
    sym_count = Counter()
    combo2herb = defaultdict(Counter)
    pair2herb = defaultdict(Counter)
    for u in train_ids:
        pos = sorted(labels[u])
        pop.update(pos)
        S = sorted(set(features[u]))
        combo2herb[tuple(S)].update(pos)
        for s in S:
            sym_count[s] += 1
            for h in pos:
                sym2herb[s][h] += 1
        if len(S) >= 2:
            for a, b in combinations(S, 2):
                pair2herb[(a, b)].update(pos)

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

    def log_norm(c):
        arr = np.zeros(n_herbs, dtype=np.float32)
        mx = max(c.values())
        for h, v in c.items():
            arr[h] = math.log(1 + v) / math.log(1 + mx)
        return arr, sum(c.values())

    tau = 2.0

    def estimate(uid, use_pair=True):
        """层次收缩估计：combo -> pair -> cond。返回 [n_herbs] 分数。"""
        S = sorted(set(features[uid]))
        c = combo2herb.get(tuple(S))
        if c is not None:
            arr, seen = log_norm(c)
            alpha = seen / (seen + tau)
            return alpha * arr + (1 - alpha) * cond_arr(uid)
        # miss：2-症状子集组合
        if use_pair and len(S) >= 2:
            agg = np.zeros(n_herbs, dtype=np.float32)
            w_sum = 0.0
            for a, b in combinations(S, 2):
                pc = pair2herb.get((a, b) if a < b else (b, a))
                if pc is None:
                    continue
                parr, pseen = log_norm(pc)
                w = pseen / (pseen + tau)
                agg += w * parr
                w_sum += w
            if w_sum > 0:
                agg /= w_sum
                # pair 聚合与 cond 加权
                return 0.7 * agg + 0.3 * cond_arr(uid)
        return cond_arr(uid)

    # --- 基线（exp4 最佳统计）---
    pred = {}
    for u in val_ids:
        pred[u] = np.argsort(-estimate(u, use_pair=False), kind='stable')[:5].tolist()
    print('0. shrunk combo tau=2     F1@5 =', round(f1_at_5(pred, va), 4))

    # --- A. pair 回退 ---
    pred = {}
    for u in val_ids:
        pred[u] = np.argsort(-estimate(u, use_pair=True), kind='stable')[:5].tolist()
    print('A. + pair fallback        F1@5 =', round(f1_at_5(pred, va), 4))

    # --- B. 药语义平滑 ---
    herb_bge = load_name_vectors(ROOT / 'features' / 'herb_bge.npz', herbs, 'item_ids', 'herb_names').numpy()
    herb_bge = herb_bge / np.linalg.norm(herb_bge, axis=1, keepdims=True)
    hsim = herb_bge @ herb_bge.T
    np.fill_diagonal(hsim, 0)
    knn = np.argsort(-hsim, axis=1)[:, :10]

    herb_seen = np.zeros(n_herbs, dtype=np.float32)
    for h, c in pop.items():
        herb_seen[h] = c

    def smooth(arr, tau2, k):
        out = arr.copy()
        beta = 1.0 / (1.0 + herb_seen / tau2)  # 稀有药 beta 大，形状 [805]
        nb_mean = arr[knn[:, :k]].mean(axis=1)
        return (1 - beta) * out + beta * nb_mean

    for tau2, k in ((5, 10), (10, 10), (5, 20), (20, 5)):
        pred = {}
        for u in val_ids:
            arr = smooth(estimate(u, use_pair=True), tau2, k)
            pred[u] = np.argsort(-arr, kind='stable')[:5].tolist()
        print(f'B. + herb smooth tau2={tau2} k={k}  F1@5 = {f1_at_5(pred, va):.4f}')

    # --- C. 最佳组合分数归一化后存盘，供融合 ---
    arrs = np.stack([estimate(u, use_pair=True) for u in val_ids])
    np.save(ROOT / 'experiments' / 'artifacts_v1' / 'val_stat_exp5.npy', arrs)
    print('saved val_stat_exp5.npy')


if __name__ == '__main__':
    main()
