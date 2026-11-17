"""快速统计基线：无需训练，在训练集内部 8:2 划分验证集测 F1@5。

目的：确立"症状 -> 处方"映射的统计方法能到什么水平，为深度模型提供参照与融合信号。
"""
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r'D:/文档/2026-11/web搜索与推荐算法/中药处方推荐算法竞赛')
DATA = ROOT / 'data'


def load(fn):
    with (DATA / fn).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def f1_at_5(pred, truth):
    scores = []
    for uid, items in truth.items():
        target = set(items)
        hits = len(target & set(pred[uid]))
        scores.append(2 * hits / (5 + len(target)))
    return sum(scores) / len(truth)


def main():
    features = {int(r['user_id']): json.loads(r['symptom_ids']) for r in load('user_features.csv')}
    labels = defaultdict(set)
    for r in load('train_interactions.csv'):
        labels[int(r['user_id'])].add(int(r['item_id']))
    all_train = sorted(labels)

    # 内部 8:2 划分：tr 统计，va 验证
    rng = __import__('numpy').random.default_rng(2026)
    ids = np_ids = rng.permutation(len(all_train))
    cut = int(round(len(all_train) * 0.8))
    tr_ids = sorted(all_train[i] for i in ids[:cut])
    va_ids = sorted(all_train[i] for i in ids[cut:])
    va = {u: sorted(labels[u]) for u in va_ids}
    print(f'train={len(tr_ids)}, val={len(va_ids)}')

    # ---- 1. 全局流行度 Top5 ----
    pop = Counter()
    for u in tr_ids:
        pop.update(labels[u])
    top5_pop = [h for h, _ in pop.most_common(5)]
    print('popularity Top5 F1@5 =', round(f1_at_5({u: top5_pop for u in va}, va), 4))

    # ---- 2. 症状条件概率：对每个症状独立打分，取 Top5 ----
    sym2herb = defaultdict(Counter)
    sym_count = Counter()
    for u in tr_ids:
        for s in features[u]:
            sym_count[s] += 1
            for h in labels[u]:
                sym2herb[s][h] += 1

    def cond_score(uid):
        scores = Counter()
        syms = features[uid]
        for s in syms:
            total = sym_count[s]
            for h, c in sym2herb[s].items():
                scores[h] += c / total  # P(h|s)
        for h in range(805):
            scores[h] += 0.0
        return scores

    def top_k(scores, k=5):
        return [h for h, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]]

    pred = {u: top_k(cond_score(u)) for u in va_ids}
    print('symptom conditional P(h|s) F1@5 =', round(f1_at_5(pred, va), 4))

    # ---- 3. 症状组合 tuple 统计（带回退）----
    combo2herb = defaultdict(Counter)
    for u in tr_ids:
        key = tuple(sorted(set(features[u])))
        combo2herb[key].update(labels[u])

    pred = {}
    miss_full = miss_single = miss_zero = 0
    for u in va_ids:
        key = tuple(sorted(set(features[u])))
        if key in combo2herb:
            c = combo2herb[key]
        else:
            miss_full += 1
            c = Counter()
            for s in key:
                if s in sym2herb:
                    c.update(sym2herb[s])
            if not c:
                miss_zero += 1
                c = pop
        pred[u] = top_k(c)
    print(f'symptom-combo F1@5 = {f1_at_5(pred, va):.4f}  (combo miss: {miss_full}/{len(va_ids)}, zero: {miss_zero})')

    # ---- 4. 条件概率 + 全局流行度平滑（interp）----
    for alpha in (0.0, 0.1, 0.2, 0.3, 0.5):
        pred = {}
        for u in va_ids:
            scores = cond_score(u)
            mx = max(scores.values()) if scores else 0
            if mx == 0:
                pred[u] = top5_pop
                continue
            for h in range(805):
                scores[h] = scores.get(h, 0) / mx + alpha * pop.get(h, 0) / pop.most_common(1)[0][1]
            pred[u] = top_k(scores)
        print(f'cond + alpha={alpha} popularity F1@5 = {f1_at_5(pred, va):.4f}')

    # ---- 5. 症状近邻扩展：利用 BGE 名称向量，相似症状加权投票 ----
    import numpy as np
    feat = np.load(ROOT / '基线代码/features/symptom_bge.npz', allow_pickle=False)
    vecs = feat['vectors'].astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    # 390 个症状两两余弦，取 top-8 邻居
    sim = vecs @ vecs.T
    np.fill_diagonal(sim, 0)
    neighbors = np.argsort(-sim, axis=1)[:, :8]

    def cond_score_nb(uid):
        scores = Counter()
        for s in features[uid]:
            w = 1.0
            for j, nb in enumerate([s] + neighbors[s].tolist()):
                weight = w if j == 0 else float(sim[s, nb])
                if weight <= 0.3:
                    continue
                total = sym_count[nb]
                for h, c in sym2herb[nb].items():
                    scores[h] += weight * c / total
        return scores

    pred = {}
    for u in va_ids:
        scores = cond_score_nb(u)
        mx = max(scores.values()) if scores else 0
        if mx == 0:
            pred[u] = top5_pop
            continue
        for h in range(805):
            scores[h] = scores.get(h, 0) / mx + 0.3 * pop.get(h, 0) / pop.most_common(1)[0][1]
        pred[u] = top_k(scores)
    print('neighbor-symptom cond F1@5 =', round(f1_at_5(pred, va), 4))


if __name__ == '__main__':
    main()
