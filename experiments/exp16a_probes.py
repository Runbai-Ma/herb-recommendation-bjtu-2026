"""实验16a：V6 后打磨探针（快速，3 成员/单成员）。

  P1. 纯 cond 特征（no_pop + no_combo）—— pop 是否也可去？
  P2. no_combo 配方下的超参探针：h=128 / h=384 / lr3e-4+ep40 / dim=128
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
from final_pipeline import build_statistics  # noqa: E402
from exp14_v6_recipes import HerbNet, build_prior_subset  # noqa: E402

SEED = 2026


def main():
    torch.set_num_threads(8)
    features, labels, test, herbs, symptoms = load_data(ROOT / 'data')
    n_herbs, n_symptoms = len(herbs), len(symptoms)
    all_train = sorted(labels)
    cut = int(round(len(all_train) * 0.80))
    tr_ids, va_ids = all_train[:cut], all_train[cut:]
    va = {u: sorted(labels[u]) for u in va_ids}
    stats = build_statistics(tr_ids, features, labels)
    pop, sym2herb, sym_count, combo2herb = stats

    sym_bge = load_name_vectors(ROOT / 'features' / 'symptom_bge.npz',
                                symptoms, 'symptom_ids', 'symptom_names')
    sym_bge = torch.nn.functional.pad(sym_bge, (0, 0, 1, 0))
    x_a = pad_symptoms(features, tr_ids, 'cpu')
    x_va = pad_symptoms(features, va_ids, 'cpu')
    y_a = torch.zeros((len(tr_ids), n_herbs))
    for row, u in enumerate(tr_ids):
        y_a[row, sorted(labels[u])] = 1.0

    def train_eval(tag, prior_kw, seed=SEED, lr=5e-4, dp=0.1, dim=64, hidden=256, epochs=30):
        pa = build_prior_subset(tr_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs, **prior_kw)
        pb = build_prior_subset(va_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs, **prior_kw)
        p_a, p_b = torch.from_numpy(pa), torch.from_numpy(pb)
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = HerbNet(n_symptoms, n_herbs, sym_bge, in_prior_dim=pa.shape[1],
                        dim=dim, hidden=hidden, dropout=dp)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(seed)
        best_f1, best_state = 0.0, None
        for ep in range(1, epochs + 1):
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
        print(f'{tag:<40} best={best_f1:.4f}', flush=True)
        return best_f1

    t0 = time.time()
    # 参照：v6 配方（cond+pop，无 combo）
    train_eval('P0 参照 cond+pop (seed2026)', dict(use_cond=True, use_combo=False, use_pop=True))
    # P1. 纯 cond
    train_eval('P1 纯cond (no_pop no_combo)', dict(use_cond=True, use_combo=False, use_pop=False))
    # P2. no_combo 超参探针
    train_eval('P2a h=128', dict(use_cond=True, use_combo=False, use_pop=True), hidden=128)
    train_eval('P2b h=384', dict(use_cond=True, use_combo=False, use_pop=True), hidden=384)
    train_eval('P2c dim=128', dict(use_cond=True, use_combo=False, use_pop=True), dim=128)
    train_eval('P2d lr=3e-4 ep=40', dict(use_cond=True, use_combo=False, use_pop=True), lr=3e-4, epochs=40)
    print(f'total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
