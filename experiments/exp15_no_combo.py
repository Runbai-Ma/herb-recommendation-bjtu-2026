"""实验15：V6 核心验证——10 成员全部使用 no_combo 特征（+no_pop 对照）。

exp14 发现：no_combo 消融成员单模型 0.2315，远超全特征成员（<=0.2242）。
本实验：no_combo 10 成员集成（顺序验证 + 跨划分 75/25），融合网格。
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
from final_pipeline import build_statistics, shrink_scores, cond_scores, normalize_rows  # noqa: E402
from exp14_v6_recipes import HerbNet, build_prior_subset  # noqa: E402

MEMBERS = [(2026, 5e-4, 0.1), (7, 5e-4, 0.1), (42, 5e-4, 0.1), (123, 5e-4, 0.1), (777, 5e-4, 0.1),
           (314, 5e-4, 0.2), (159, 5e-4, 0.2), (265, 5e-4, 0.2),
           (358, 1e-3, 0.1), (979, 1e-3, 0.1)]


def run_split(tr_ids, va_ids, features, labels, herbs, symptoms, sym_bge, tag):
    n_herbs, n_symptoms = len(herbs), len(symptoms)
    va = {u: sorted(labels[u]) for u in va_ids}
    stats = build_statistics(tr_ids, features, labels)
    pop, sym2herb, sym_count, combo2herb = stats

    x_a = pad_symptoms(features, tr_ids, 'cpu')
    x_va = pad_symptoms(features, va_ids, 'cpu')
    y_a = torch.zeros((len(tr_ids), n_herbs))
    for row, u in enumerate(tr_ids):
        y_a[row, sorted(labels[u])] = 1.0

    pa = build_prior_subset(tr_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs,
                            use_cond=True, use_combo=False, use_pop=True)
    pb = build_prior_subset(va_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs,
                            use_cond=True, use_combo=False, use_pop=True)
    p_a, p_b = torch.from_numpy(pa), torch.from_numpy(pb)

    t0 = time.time()
    logits_list, f1s = [], []
    for seed, lr, dp in MEMBERS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = HerbNet(n_symptoms, n_herbs, sym_bge, in_prior_dim=pa.shape[1], dropout=dp)
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
                lg = model(x_va, p_b).numpy()
            pred = {u: np.argsort(-lg[i], kind='stable')[:5].tolist() for i, u in enumerate(va_ids)}
            f1 = f1_at_5(pred, va)
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        with torch.inference_mode():
            logits_list.append(model(x_va, p_b).numpy())
        f1s.append(best_f1)
        print(f'[{tag}] seed={seed}: {best_f1:.4f} ({time.time()-t0:.0f}s)', flush=True)

    ens = np.mean(logits_list, axis=0)
    f1_ens = f1_at_5({u: np.argsort(-ens[i], kind='stable')[:5].tolist() for i, u in enumerate(va_ids)}, va)
    print(f'[{tag}] 10-member no_combo ensemble model-only: {f1_ens:.4f}')

    stat_va = shrink_scores(va_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs, 30.0)
    cond_va = cond_scores(va_ids, features, sym2herb, sym_count, n_herbs)
    best = (None, 0.0)
    for w_s in (0.2, 0.3, 0.4, 0.5):
        for w_m in (0.4, 0.5, 0.6, 0.7):
            for w_c in (0.0, 0.1, 0.2):
                if abs(w_s + w_m + w_c - 1.0) > 1e-9:
                    continue
                sc = w_s * normalize_rows(stat_va) + w_m * normalize_rows(ens) + w_c * normalize_rows(cond_va)
                pred = {u: np.argsort(-sc[i], kind='stable')[:5].tolist() for i, u in enumerate(va_ids)}
                f1 = f1_at_5(pred, va)
                if f1 > best[1]:
                    best = ((w_s, w_m, w_c), f1)
    print(f'[{tag}] best fusion {best[0]}: {best[1]:.4f}')
    return ens, best, np.array(va_ids), stats


def main():
    torch.set_num_threads(8)
    features, labels, test, herbs, symptoms = load_data(ROOT / 'data')
    all_train = sorted(labels)
    sym_bge = load_name_vectors(ROOT / 'features' / 'symptom_bge.npz',
                                symptoms, 'symptom_ids', 'symptom_names')
    sym_bge = torch.nn.functional.pad(sym_bge, (0, 0, 1, 0))

    cut = int(round(len(all_train) * 0.80))
    ens, best, va_ids, stats = run_split(all_train[:cut], all_train[cut:], features, labels, herbs, symptoms, sym_bge, '主验证80/20')

    # 跨划分：75/25
    cut2 = int(round(len(all_train) * 0.75))
    ens2, best2, va_ids2, stats2 = run_split(all_train[:cut2], all_train[cut2:], features, labels, herbs, symptoms, sym_bge, '跨划分75/25')

    out = ROOT / 'experiments' / 'artifacts_v6'
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / 'no_combo_val.npz',
                        ens80=ens, ens75=ens2,
                        va_ids80=va_ids, va_ids75=va_ids2,
                        fusion80=np.array(best[0]), fusion75=np.array(best2[0]))
    print('saved artifacts_v6/no_combo_val.npz')


if __name__ == '__main__':
    main()
