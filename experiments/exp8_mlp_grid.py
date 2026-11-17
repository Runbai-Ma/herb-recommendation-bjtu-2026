"""实验8：顺序协议下 MLP 超参网格（model-only 顺序 val F1@5）。

统一协议：前 80% 训练（统计量同源），后 20% 评估。
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
from final_pipeline import build_statistics, build_prior_features, HerbNet  # noqa: E402

SEED = 2026


def eval_config(tag, dim, hidden, dropout, lr, epochs, pos_weight=None, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(8)
    features, labels, test, herbs, symptoms = DATA['features'], DATA['labels'], DATA['test'], DATA['herbs'], DATA['symptoms']
    n_herbs, n_symptoms = len(herbs), len(symptoms)
    all_train = sorted(labels)
    train_ids, val_ids = all_train[:21610], all_train[21610:]
    va = {u: sorted(labels[u]) for u in val_ids}

    pop, sym2herb, sym_count, combo2herb = DATA['stats']
    prior_train = build_prior_features(train_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs)
    prior_val = build_prior_features(val_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs)
    x_train = pad_symptoms(features, train_ids, 'cpu')
    x_val = pad_symptoms(features, val_ids, 'cpu')
    p_train = torch.from_numpy(prior_train)
    p_val = torch.from_numpy(prior_val)
    y_train = torch.zeros((len(train_ids), n_herbs))
    for row, u in enumerate(train_ids):
        y_train[row, sorted(labels[u])] = 1.0

    model = HerbNet(n_symptoms, n_herbs, DATA['sym_bge'], in_prior_dim=prior_train.shape[1],
                    dim=dim, hidden=hidden, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pw = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight])) if pos_weight else nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)

    best_f1, best_ep = 0.0, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(len(train_ids))
        for st in range(0, len(perm), 256):
            idx = torch.as_tensor(perm[st:st + 256], dtype=torch.long)
            loss = pw(model(x_train[idx], p_train[idx]), y_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if ep >= 10:  # 10轮后每轮评估
            model.eval()
            with torch.inference_mode():
                logits = model(x_val, p_val).numpy()
            pred = {u: np.argsort(-logits[i], kind='stable')[:5].tolist() for i, u in enumerate(val_ids)}
            f1 = f1_at_5(pred, va)
            if f1 > best_f1:
                best_f1, best_ep = f1, ep
    print(f'{tag:<44} best F1@5 = {best_f1:.4f} @ep{best_ep}', flush=True)
    return best_f1


def main():
    global DATA
    from data import load_name_vectors
    features, labels, test, herbs, symptoms = load_data(ROOT / 'data')
    sym_bge = load_name_vectors(ROOT / 'features' / 'symptom_bge.npz',
                                symptoms, 'symptom_ids', 'symptom_names')
    sym_bge = torch.nn.functional.pad(sym_bge, (0, 0, 1, 0))
    all_train = sorted(labels)
    DATA = dict(features=features, labels=labels, test=test, herbs=herbs, symptoms=symptoms,
                sym_bge=sym_bge, stats=build_statistics(all_train[:21610], features, labels))

    t0 = time.time()
    eval_config('v2 base dim=64 h=256 lr1e-3 ep20', 64, 256, 0.1, 1e-3, 20)
    eval_config('dim=128 h=256', 128, 256, 0.1, 1e-3, 20)
    eval_config('dim=64 h=512', 64, 512, 0.1, 1e-3, 20)
    eval_config('dim=128 h=512', 128, 512, 0.1, 1e-3, 25)
    eval_config('dim=64 h=256 lr5e-4 ep30', 64, 256, 0.1, 5e-4, 30)
    eval_config('dim=64 h=256 ep30', 64, 256, 0.1, 1e-3, 30)
    eval_config('dim=64 h=256 dropout=0.2', 64, 256, 0.2, 1e-3, 20)
    eval_config('dim=64 h=256 dropout=0.0', 64, 256, 0.0, 1e-3, 20)
    eval_config('dim=64 h=256 posw=2', 64, 256, 0.1, 1e-3, 20, pos_weight=2.0)
    eval_config('dim=64 h=256 posw=5', 64, 256, 0.1, 1e-3, 20, pos_weight=5.0)
    eval_config('dim=64 h=256 posw=10', 64, 256, 0.1, 1e-3, 20, pos_weight=10.0)
    print(f'total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
