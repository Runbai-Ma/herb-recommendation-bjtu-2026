"""实验13：基于 exp12 保存的验证 logits 做免重训融合增强。

  A. 不同 tau 的 stat 信号 × rank 融合
  B. 按症状数分组的融合权重（单症状/多症状分开调 w）
  C. tau × 分组组合的最优搜索
"""
import sys
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import load_data  # noqa: E402
from metrics import f1_at_5  # noqa: E402
from final_pipeline import build_statistics, shrink_scores, cond_scores, normalize_rows  # noqa: E402

SEED = 2026


def rank_norm(mat):
    n_rows, n_cols = mat.shape
    order = np.argsort(mat, axis=1, kind='stable')
    ranks = np.empty_like(order)
    rows = np.arange(n_rows)[:, None]
    ranks[rows, order] = np.arange(1, n_cols + 1)[None, :]
    return ranks / n_cols


def main():
    np.random.seed(SEED)
    features, labels, test, herbs, symptoms = load_data(ROOT / 'data')
    n_herbs = len(herbs)
    all_train = sorted(labels)
    va_ids = all_train[21610:]
    va = {u: sorted(labels[u]) for u in va_ids}

    logits_arr = np.load(ROOT / 'experiments' / 'artifacts_v5' / 'val_logits_15.npy')
    ens = logits_arr.mean(axis=0)

    stats = build_statistics(all_train[:21610], features, labels)
    pop, sym2herb, sym_count, combo2herb = stats

    stat_by_tau = {tau: shrink_scores(va_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs, tau)
                   for tau in (15, 20, 30, 50, 100)}
    cond_va = cond_scores(va_ids, features, sym2herb, sym_count, n_herbs)

    ens_r, cond_r = rank_norm(ens), rank_norm(cond_va)
    stat_r = {tau: rank_norm(s) for tau, s in stat_by_tau.items()}

    n_sym = np.array([len(set(features[u])) for u in va_ids])
    single_mask = n_sym == 1
    multi_mask = ~single_mask

    def eval_fused(w_s, w_m, w_c, tau, group_weights=None):
        """group_weights: (w_s_single, w_m_single, w_s_multi, w_m_multi) 覆盖式分组权重。"""
        sc = w_s * stat_r[tau] + w_m * ens_r + w_c * cond_r
        if group_weights is not None:
            s1, m1, s2, m2 = group_weights
            c1 = s1 + m1
            c2 = s2 + m2
            sc_single = s1 * stat_r[tau] + m1 * ens_r + (c1 and w_c or 0) * cond_r
            sc_multi = s2 * stat_r[tau] + m2 * ens_r + (c2 and w_c or 0) * cond_r
            # 归一化组内权重到 1
            sc_single = (s1 * stat_r[tau] + m1 * ens_r) / c1 if c1 else sc
            sc_multi = (s2 * stat_r[tau] + m2 * ens_r) / c2 if c2 else sc
            sc = np.where(single_mask[:, None], sc_single, sc_multi)
        pred = {u: np.argsort(-sc[i], kind='stable')[:5].tolist() for i, u in enumerate(va_ids)}
        return f1_at_5(pred, va)

    # A. tau × 权重
    best = (None, 0.0)
    for tau in stat_r:
        for w_s in (0.2, 0.3, 0.4, 0.5):
            for w_m in (0.4, 0.5, 0.6, 0.7):
                w_c = round(1 - w_s - w_m, 2)
                if w_c < 0 or w_c > 0.2:
                    continue
                f1 = eval_fused(w_s, w_m, w_c, tau)
                if f1 > best[1]:
                    best = ((w_s, w_m, w_c, tau), f1)
    print('A. tau x rank fusion best:', best[0], f'F1@5 = {best[1]:.4f}')

    # 基准：全局权重 tau=30
    base = eval_fused(0.4, 0.6, 0.0, 30)
    print('   baseline (0.4/0.6/0 tau30):', round(base, 4))

    # 分组统计信号强度：单/多症状分别统计
    for name, mask in (('单症状', single_mask), ('多症状', multi_mask)):
        sub = [u for u, m in zip(va_ids, mask) if m]
        f1 = sum(2 * len(set(labels[u]) & set(np.argsort(-(0.4 * stat_r[30] + 0.6 * ens_r)[i], kind='stable')[:5])) / (5 + len(labels[u])) for i, u in enumerate(va_ids) if mask[i]) / len(sub)
        print(f'   {name}({len(sub)}条) 当前融合 F1@5 = {f1:.4f}')

    # B/C. 分组权重搜索：单症状组与多症状组分别调
    best_g = (None, 0.0)
    for tau in (20, 30, 50):
        for s1 in (0.3, 0.4, 0.5, 0.6):
            sc_single = s1 * stat_r[tau] + (1 - s1) * ens_r
            pred = {u: np.argsort(-sc_single[i], kind='stable')[:5].tolist()
                    for i, u in enumerate(va_ids) if single_mask[i]}
            f1 = sum(2 * len(set(labels[u]) & set(pred[u])) / (5 + len(labels[u])) for u in pred) / len(pred)
            if f1 > best_g[1]:
                best_g = ((tau, 'single', s1, 1 - s1), f1)
    print('B. 单症状组最优:', best_g[0], f'F1@5 = {best_g[1]:.4f}')

    best_m = (None, 0.0)
    for tau in (20, 30, 50):
        for s2 in (0.0, 0.1, 0.2, 0.3, 0.4):
            for c2 in (0.0, 0.1, 0.2):
                m2 = 1 - s2 - c2
                if m2 < 0.3:
                    continue
                idx_multi = [i for i, u in enumerate(va_ids) if multi_mask[i]]
                sc_multi = s2 * stat_r[tau] + m2 * ens_r + c2 * cond_r
                f1 = sum(2 * len(set(labels[u]) & set(np.argsort(-sc_multi[i], kind='stable')[:5])) / (5 + len(labels[u]))
                         for i, u in enumerate(va_ids) if multi_mask[i]) / len(idx_multi)
                if f1 > best_m[1]:
                    best_m = ((tau, 'multi', s2, m2, c2), f1)
    print('B. 多症状组最优:', best_m[0], f'F1@5 = {best_m[1]:.4f}')

    # C. 组装最终分组融合
    (tau1, _, s1, m1) = best_g[0]
    (tau2, _, s2, m2, c2) = best_m[0]
    tau = (tau1 + tau2) / 2 if tau1 != tau2 else tau1
    sc_single = s1 * stat_r[tau1] + m1 * ens_r
    sc_multi = s2 * stat_r[tau2] + m2 * ens_r + c2 * cond_r
    sc = np.where(single_mask[:, None], sc_single, sc_multi)
    pred = {u: np.argsort(-sc[i], kind='stable')[:5].tolist() for i, u in enumerate(va_ids)}
    print(f'C. grouped fusion (tau {tau1}/{tau2}, single {s1}/{m1}, multi {s2}/{m2}/{c2}): F1@5 = {f1_at_5(pred, va):.4f}')


if __name__ == '__main__':
    main()
