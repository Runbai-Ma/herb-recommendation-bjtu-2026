"""实验11：10-model 多样性集成（无语义特征）+ 融合方式对比。

问题：exp10 显示语义匹配特征单模型层面负向（-0.004），但 10-model 集成净增益 +0.0014。
本实验分离变量：同样 10-model 配置，不加语义特征，看集成与融合是否更高。
附加：rank 归一化融合 vs min-max 归一化融合。
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
from exp10_semfeat_ensemble import SEEDS  # noqa: E402

SEED = 2026


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
    stats = build_statistics(tr_ids, features, labels)
    pop, sym2herb, sym_count, combo2herb = stats
    prior_a = build_prior_features(tr_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs)
    prior_b = build_prior_features(va_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs)
    x_a = pad_symptoms(features, tr_ids, 'cpu')
    x_b = pad_symptoms(features, va_ids, 'cpu')
    p_a, p_b = torch.from_numpy(prior_a), torch.from_numpy(prior_b)
    y_a = torch.zeros((len(tr_ids), n_herbs))
    for row, u in enumerate(tr_ids):
        y_a[row, sorted(labels[u])] = 1.0

    logits_list, f1s = [], []
    for seed, lr, dp in SEEDS:
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

    ens = np.mean(logits_list, axis=0)
    f1_ens = f1_at_5({u: np.argsort(-ens[i], kind='stable')[:5].tolist() for i, u in enumerate(va_ids)}, va)
    print(f'10-model ensemble (NO sem feat) model-only = {f1_ens:.4f}')

    stat_va = shrink_scores(va_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs, 30.0)
    cond_va = cond_scores(va_ids, features, sym2herb, sym_count, n_herbs)

    def rank_norm(mat):
        # 逐记录排名归一化（numpy 实现，1=最高分）
        order = np.argsort(-mat, axis=1, kind='stable')
        ranks = np.empty_like(order)
        rows = np.arange(mat.shape[0])[:, None]
        ranks[rows, order] = np.arange(1, mat.shape[1] + 1)[None, :]
        return ranks / mat.shape[1]

    # min-max 融合网格
    best = (None, 0.0)
    for w_s in (0.2, 0.3, 0.4, 0.5):
        for w_m in (0.3, 0.4, 0.5, 0.6):
            for w_c in (0.0, 0.1, 0.2):
                if abs(w_s + w_m + w_c - 1.0) > 1e-9:
                    continue
                sc = w_s * normalize_rows(stat_va) + w_m * normalize_rows(ens) + w_c * normalize_rows(cond_va)
                pred = {u: np.argsort(-sc[i], kind='stable')[:5].tolist() for i, u in enumerate(va_ids)}
                f1 = f1_at_5(pred, va)
                if f1 > best[1]:
                    best = ((w_s, w_m, w_c), f1)
    print(f'min-max best fusion: {best[0]}  F1@5 = {best[1]:.4f}')

    # rank 融合网格
    best_r = (None, 0.0)
    for w_s in (0.2, 0.3, 0.4, 0.5):
        for w_m in (0.3, 0.4, 0.5, 0.6):
            for w_c in (0.0, 0.1, 0.2):
                if abs(w_s + w_m + w_c - 1.0) > 1e-9:
                    continue
                sc = w_s * rank_norm(stat_va) + w_m * rank_norm(ens) + w_c * rank_norm(cond_va)
                pred = {u: np.argsort(-sc[i], kind='stable')[:5].tolist() for i, u in enumerate(va_ids)}
                f1 = f1_at_5(pred, va)
                if f1 > best_r[1]:
                    best_r = ((w_s, w_m, w_c), f1)
    print(f'rank best fusion:    {best_r[0]}  F1@5 = {best_r[1]:.4f}')


if __name__ == '__main__':
    main()
