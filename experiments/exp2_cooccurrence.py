"""实验2：药-药共现贪心选择 + 症状近邻扩展 + 组合回退链，统一验证集评估。

信号分解：
  A. 症状->药 条件概率（单症状/组合统计）
  B. 药-药配伍共现（处方内搭配规律，独立于症状信号）
  C. BGE 症状语义近邻（扩展稀疏症状的统计覆盖）
"""
import sys
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import load_data, load_name_vectors, split_train  # noqa: E402
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
    print(f'train={len(train_ids)}, val={len(val_ids)}')

    # ---------- 统计量 ----------
    pop = Counter()
    sym2herb = defaultdict(Counter)
    sym_count = Counter()
    combo2herb = defaultdict(Counter)
    pair_co = Counter()          # 药-药共现计数
    script_count = len(train_ids)
    for u in train_ids:
        pos = sorted(labels[u])
        pop.update(pos)
        combo2herb[tuple(sorted(set(features[u])))].update(pos)
        for s in set(features[u]):
            sym_count[s] += 1
            for h in pos:
                sym2herb[s][h] += 1
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                pair_co[(pos[i], pos[j])] += 1

    # 共现矩阵（对称）：lift = P(共现) / (P(h)*P(h'))，加平滑
    co_mat = np.zeros((n_herbs, n_herbs), dtype=np.float32)
    for (a, b), c in pair_co.items():
        co_mat[a, b] = c
        co_mat[b, a] = c
    freq = np.zeros(n_herbs, dtype=np.float32)
    for h, c in pop.items():
        freq[h] = c
    p_h = (freq + 1.0) / (script_count + n_herbs)
    exp_co = np.outer(p_h, p_h) * script_count
    lift = (co_mat + 1.0) / (exp_co + 1.0)
    np.fill_diagonal(lift, 1.0)
    lift = np.log(lift)

    # ---------- 评分函数 ----------
    def cond_scores(uid):
        sc = np.zeros(n_herbs, dtype=np.float32)
        for s in set(features[uid]):
            total = sym_count[s]
            for h, c in sym2herb[s].items():
                sc[h] += c / total
        return sc

    def combo_dist(uid):
        key = tuple(sorted(set(features[uid])))
        return combo2herb.get(key)

    def base_top5(sc):
        return np.argsort(-sc, kind='stable')[:5].tolist()

    # 0. 诊断：combo 覆盖率
    hit = sum(1 for u in val_ids if tuple(sorted(set(features[u]))) in combo2herb)
    print(f'combo coverage in val: {hit}/{len(val_ids)} = {hit/len(val_ids)*100:.1f}%')

    # 1. cond-sum top5
    pred = {}
    for u in val_ids:
        pred[u] = base_top5(cond_scores(u))
    print('1. cond-sum top5          F1@5 =', round(f1_at_5(pred, va), 4))

    # 2. combo 统计 + cond 回退
    pred = {}
    pop_order = np.argsort(-freq, kind='stable')
    for u in val_ids:
        c = combo_dist(u)
        if c is None:
            pred[u] = base_top5(cond_scores(u))
        else:
            arr = np.zeros(n_herbs, dtype=np.float32)
            for h, v in c.items():
                arr[h] = math.log(1 + v) / math.log(1 + max(c.values()))
            # combo 稀疏时混入 cond 信号
            seen = sum(c.values())
            pred[u] = base_top5(arr if seen >= 10 else 0.5 * arr + 0.5 * cond_scores(u))
    print('2. combo(+cond fallback)  F1@5 =', round(f1_at_5(pred, va), 4))

    # 3. 药药共现贪心（在 cond-sum 基础上）
    def greedy_select(base, lam, k=5, cand=None):
        cand = cand if cand is not None else np.argsort(-base, kind='stable')[:30]
        selected = []
        for _ in range(k):
            best_h, best_v = None, -1e18
            for h in cand:
                if h in selected:
                    continue
                co_v = lift[h, selected].mean() if selected else 0.0
                v = base[h] + lam * co_v
                if v > best_v:
                    best_v, best_h = v, h
            selected.append(best_h)
        return selected

    for lam in (0.2, 0.5, 1.0, 2.0, 4.0):
        pred = {}
        for u in val_ids:
            pred[u] = greedy_select(cond_scores(u), lam)
        print(f'3. cond + co-greedy lam={lam}  F1@5 = {f1_at_5(pred, va):.4f}')

    # 4. BGE 近邻症状聚合 + cond
    feat = np.load(ROOT / 'features' / 'symptom_bge.npz', allow_pickle=False)
    vecs = feat['vectors'].astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    sim = vecs @ vecs.T
    np.fill_diagonal(sim, 0)

    def cond_scores_nn(uid, nb, w_floor):
        sc = np.zeros(n_herbs, dtype=np.float32)
        for s in set(features[uid]):
            for nb_s, w in [(s, 1.0)] + [(int(j), float(sim[s, j])) for j in np.argsort(-sim[s])[:nb]]:
                if nb_s != s and (w < w_floor or nb_s not in sym2herb):
                    continue
                total = sym_count[nb_s]
                for h, c in sym2herb[nb_s].items():
                    sc[h] += w * c / total
        return sc

    for nb, floor in ((4, 0.5), (8, 0.4), (8, 0.6), (16, 0.5)):
        pred = {}
        for u in val_ids:
            sc = cond_scores_nn(u, nb, floor)
            mx = sc.max()
            if mx <= 0:
                pred[u] = pop_order[:5].tolist()
            else:
                pred[u] = base_top5(sc / mx + 0.2 * freq / (freq.max() + 1e-9))
        print(f'4. nn-sym nb={nb} floor={floor}  F1@5 = {f1_at_5(pred, va):.4f}')

    # 5. 组合分 + 共现贪心
    for lam in (0.5, 1.0, 2.0):
        pred = {}
        for u in val_ids:
            c = combo_dist(u)
            arr = cond_scores(u)
            if c is not None:
                seen = sum(c.values())
                c_arr = np.zeros(n_herbs, dtype=np.float32)
                for h, v in c.items():
                    c_arr[h] = math.log(1 + v) / math.log(1 + max(c.values()))
                arr = c_arr if seen >= 10 else 0.5 * c_arr + 0.5 * arr
            pred[u] = greedy_select(arr, lam)
        print(f'5. combo/cond + co-greedy lam={lam}  F1@5 = {f1_at_5(pred, va):.4f}')


if __name__ == '__main__':
    main()
