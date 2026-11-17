"""实验16b：打磨组合验证——纯cond + dim=128 的 3 成员稳定性 + 跨划分确认。"""
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
from final_pipeline import build_statistics  # noqa: E402
from exp14_v6_recipes import HerbNet, build_prior_subset  # noqa: E402

SEEDS = (2026, 7, 42)


def main():
    torch.set_num_threads(8)
    features, labels, test, herbs, symptoms = load_data(ROOT / 'data')
    n_herbs, n_symptoms = len(herbs), len(symptoms)
    all_train = sorted(labels)
    sym_bge = load_name_vectors(ROOT / 'features' / 'symptom_bge.npz',
                                symptoms, 'symptom_ids', 'symptom_names')
    sym_bge = torch.nn.functional.pad(sym_bge, (0, 0, 1, 0))

    for cut_ratio, tag in ((0.80, '80/20'), (0.75, '75/25')):
        cut = int(round(len(all_train) * cut_ratio))
        tr_ids, va_ids = all_train[:cut], all_train[cut:]
        va = {u: sorted(labels[u]) for u in va_ids}
        stats = build_statistics(tr_ids, features, labels)
        pop, sym2herb, sym_count, combo2herb = stats
        x_a = pad_symptoms(features, tr_ids, 'cpu')
        x_va = pad_symptoms(features, va_ids, 'cpu')
        y_a = torch.zeros((len(tr_ids), n_herbs))
        for row, u in enumerate(tr_ids):
            y_a[row, sorted(labels[u])] = 1.0
        pa = build_prior_subset(tr_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs,
                                use_cond=True, use_combo=False, use_pop=False)
        pb = build_prior_subset(va_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs,
                                use_cond=True, use_combo=False, use_pop=False)
        p_a, p_b = torch.from_numpy(pa), torch.from_numpy(pb)

        t0 = time.time()
        logits_list, f1s = [], []
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = HerbNet(n_symptoms, n_herbs, sym_bge, in_prior_dim=pa.shape[1], dim=128)
            opt = torch.optim.Adam(model.parameters(), lr=5e-4)
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
        print(f'[{tag}] pure-cond dim128 x3 ensemble: {f1_ens:.4f}', flush=True)


if __name__ == '__main__':
    main()
