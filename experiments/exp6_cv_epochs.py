"""CV 确认 MLP 最佳训练轮数（每折统计量折内构建，避免泄漏）。"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import load_data, split_train, load_name_vectors, pad_symptoms  # noqa: E402
from metrics import f1_at_5  # noqa: E402
from exp1_prior_mlp import HerbNet, build_prior_features  # noqa: E402

SEEDS = (2026, 7, 42)


def train_one_split(seed, features, labels, herbs, symptoms, sym_bge, n_herbs):
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_ids, val_ids = split_train(labels, ratio=0.2, seed=seed)
    va = {u: sorted(labels[u]) for u in val_ids}

    pop, sym2herb, sym_count, combo2herb = Counter(), defaultdict(Counter), Counter(), defaultdict(Counter)
    for u in train_ids:
        pos = labels[u]
        pop.update(pos)
        combo2herb[tuple(sorted(set(features[u])))].update(pos)
        for s in set(features[u]):
            sym_count[s] += 1
            for h in pos:
                sym2herb[s][h] += 1

    x_tr_prior = build_prior_features(train_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs)
    x_va_prior = build_prior_features(val_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs)
    x_tr = pad_symptoms(features, train_ids, 'cpu')
    x_va = pad_symptoms(features, val_ids, 'cpu')
    p_tr = torch.from_numpy(x_tr_prior)
    p_va = torch.from_numpy(x_va_prior)

    y_tr = torch.zeros((len(train_ids), n_herbs))
    for row, u in enumerate(train_ids):
        y_tr[row, sorted(labels[u])] = 1.0

    model = HerbNet(len(symptoms), n_herbs, sym_bge, in_prior_dim=x_tr_prior.shape[1]).to('cpu')
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    torch.set_num_threads(8)

    f1s = []
    for ep in range(1, 25):
        model.train()
        perm = rng.permutation(len(train_ids))
        for st in range(0, len(perm), 256):
            idx = torch.as_tensor(perm[st:st + 256], dtype=torch.long)
            loss = loss_fn(model(x_tr[idx], p_tr[idx]), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.inference_mode():
            logits = model(x_va, p_va).numpy()
        pred = {u: np.argsort(-logits[i], kind='stable')[:5].tolist() for i, u in enumerate(val_ids)}
        f1s.append(f1_at_5(pred, va))
    return f1s


def main():
    features, labels, test, herbs, symptoms = load_data(ROOT / 'data')
    n_herbs, n_symptoms = len(herbs), len(symptoms)
    from data import load_name_vectors
    sym_bge = load_name_vectors(ROOT / 'features' / 'symptom_bge.npz',
                                symptoms, 'symptom_ids', 'symptom_names')
    sym_bge = torch.nn.functional.pad(sym_bge, (0, 0, 1, 0))

    all_f1 = []
    for seed in SEEDS:
        f1s = train_one_split(seed, features, labels, herbs, symptoms, sym_bge, n_herbs)
        all_f1.append(f1s)
        print(f'seed={seed}: best epoch {int(np.argmax(f1s))+1}, best F1={max(f1s):.4f}')
        print('  ', ' '.join(f'{f:.4f}' for f in f1s))

    mean_f1 = np.mean(all_f1, axis=0)
    print('mean best epoch:', int(np.argmax(mean_f1)) + 1, 'mean F1:', round(float(mean_f1.max()), 4))
    print('epochs 10-22 mean:', ' '.join(f'{e+1}:{v:.4f}' for e, v in enumerate(mean_f1[9:22], start=9)))


if __name__ == '__main__':
    main()
