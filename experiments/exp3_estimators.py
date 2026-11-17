"""实验3：更好的 P(h|症状) 估计器。

  A. kNN 记录检索（症状 Jaccard 相似，加权聚合处方分布）—— combo 的软版本
  B. 子集回退链：完整组合 -> 去掉一个症状的子集组合 -> cond-sum
  C. IDF 加权 cond（罕见症状特异性高）
  D. combo 向 cond 的连续自适应收缩（替代 seen>=10 硬阈值）
"""
import sys
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import load_data, split_train  # noqa: E402
from metrics import f1_at_5  # noqa: E402

SEED = 2026


def top5(arr):
    return np.argsort(-arr, kind='stable')[:5].tolist()


def main():
    np.random.seed(SEED)
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

    # --- cond-sum 基线 ---
    def cond_arr(uid):
        arr = np.zeros(n_herbs, dtype=np.float32)
        for s in set(features[uid]):
            for h, c in sym2herb[s].items():
                arr[h] += c / sym_count[s]
        return arr

    pred = {u: top5(cond_arr(u)) for u in val_ids}
    print('0. cond-sum               F1@5 =', round(f1_at_5(pred, va), 4))

    # --- A. kNN 记录检索 ---
    tr_sym_sets = {u: set(features[u]) for u in train_ids}
    val_sym_sets = {u: set(features[u]) for u in val_ids}
    # 预索引：症状 -> 训练记录列表（只比对共享症状的记录）
    inv = defaultdict(list)
    for u in train_ids:
        for s in tr_sym_sets[u]:
            inv[s].append(u)

    for K in (20, 50, 100):
        pred = {}
        for u in val_ids:
            S = val_sym_sets[u]
            cand = Counter()
            for s in S:
                cand.update(inv[s])
            # 候选打分：Jaccard
            scored = []
            for t, shared in cand.items():
                union = len(S) + len(tr_sym_sets[t]) - shared
                scored.append((shared / union, t))
            scored.sort(reverse=True)
            agg = np.zeros(n_herbs, dtype=np.float32)
            w_sum = 0.0
            for sim, t in scored[:K]:
                if sim <= 0:
                    break
                w = sim * sim  # 锐化
                agg[sorted(labels[t])] += w
                w_sum += w
            if w_sum > 0:
                pred[u] = top5(agg)
            else:
                pred[u] = top5(cond_arr(u) + 0.2 * pop_prior)
        print(f'A. kNN records K={K}       F1@5 = {f1_at_5(pred, va):.4f}')

    # --- B. 子集回退链 ---
    pred = {}
    for u in val_ids:
        S = sorted(set(features[u]))
        key = tuple(S)
        c = combo2herb.get(key)
        source = 'combo'
        if c is None and len(S) > 1:
            # 去掉一个症状的所有子集，加权合并（大子集优先）
            merged = Counter()
            for i in range(len(S)):
                sub = combo2herb.get(tuple(S[:i] + S[i + 1:]))
                if sub:
                    for h, v in sub.items():
                        merged[h] += v
            if merged:
                c, source = merged, 'subset'
        if c is None:
            pred[u] = top5(cond_arr(u))
            continue
        arr = np.zeros(n_herbs, dtype=np.float32)
        mx = max(c.values())
        for h, v in c.items():
            arr[h] = math.log(1 + v) / math.log(1 + mx)
        seen = sum(c.values())
        if seen < 10:
            arr = (arr + cond_arr(u)) / 2
        pred[u] = top5(arr)
    print('B. subset fallback        F1@5 =', round(f1_at_5(pred, va), 4))

    # --- C. IDF 加权 cond ---
    idf = {s: math.log(script_den := len(train_ids) / (1 + sym_count[s])) for s in sym_count}
    pred = {}
    for u in val_ids:
        arr = np.zeros(n_herbs, dtype=np.float32)
        for s in set(features[u]):
            w = idf.get(s, math.log(len(train_ids)))
            for h, c in sym2herb[s].items():
                arr[h] += w * c / sym_count[s]
        mx = arr.max()
        if mx > 0:
            arr = arr / mx + 0.2 * pop_prior
        pred[u] = top5(arr)
    print('C. IDF-weighted cond      F1@5 =', round(f1_at_5(pred, va), 4))

    # --- D. combo 连续收缩 ---
    # 收缩系数 alpha = seen / (seen + tau)，combo 与 cond 按 alpha 加权
    for tau in (2, 5, 10, 20):
        pred = {}
        for u in val_ids:
            c = combo2herb.get(tuple(sorted(set(features[u]))))
            cond = cond_arr(u)
            if c is None:
                pred[u] = top5(cond)
                continue
            arr = np.zeros(n_herbs, dtype=np.float32)
            mx = max(c.values())
            for h, v in c.items():
                arr[h] = math.log(1 + v) / math.log(1 + mx)
            seen = sum(c.values())
            alpha = seen / (seen + tau)
            pred[u] = top5(alpha * arr + (1 - alpha) * cond)
        print(f'D. shrunk combo tau={tau}    F1@5 = {f1_at_5(pred, va):.4f}')

    # --- E. D 的最优 + subset fallback 结合 ---
    for tau in (5, 10):
        pred = {}
        for u in val_ids:
            S = sorted(set(features[u]))
            c = combo2herb.get(tuple(S))
            if c is None and len(S) > 1:
                merged = Counter()
                for i in range(len(S)):
                    sub = combo2herb.get(tuple(S[:i] + S[i + 1:]))
                    if sub:
                        for h, v in sub.items():
                            merged[h] += v
                if merged:
                    c = merged
            cond = cond_arr(u)
            if c is None:
                pred[u] = top5(cond)
                continue
            arr = np.zeros(n_herbs, dtype=np.float32)
            mx = max(c.values())
            for h, v in c.items():
                arr[h] = math.log(1 + v) / math.log(1 + mx)
            seen = sum(c.values())
            alpha = seen / (seen + tau)
            pred[u] = top5(alpha * arr + (1 - alpha) * cond)
        print(f'E. subset + shrunk tau={tau}  F1@5 = {f1_at_5(pred, va):.4f}')


if __name__ == '__main__':
    main()
