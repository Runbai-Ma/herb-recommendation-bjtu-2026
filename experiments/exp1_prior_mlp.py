"""实验1：统计先验 + BGE 语义 + ID 协同 -> 宽 MLP，统一验证集评估。

数据划分复用方案/data.py 的 split_train（numpy default_rng(2026), ratio=0.2），
保证与后续所有实验可比。
"""
import sys
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import load_data, load_name_vectors, split_train, pad_symptoms  # noqa: E402
from metrics import f1_at_5  # noqa: E402

SEED = 2026
DEVICE = 'cpu'


def stat_priors(train_ids, features, labels, n_herbs):
    """从训练部分统计：症状->药条件概率、症状组合->药分布、全局流行度。"""
    pop = Counter()
    sym2herb = defaultdict(Counter)
    sym_count = Counter()
    combo2herb = defaultdict(Counter)
    for u in train_ids:
        pos = labels[u]
        pop.update(pos)
        key = tuple(sorted(set(features[u])))
        combo2herb[key].update(pos)
        for s in features[u]:
            sym_count[s] += 1
            for h in pos:
                sym2herb[s][h] += 1
    return pop, sym2herb, sym_count, combo2herb


def build_prior_features(user_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs):
    """每条记录构造 805 维条件概率先验 + 组合先验 + log 流行度，形状 [n, 3*805]。"""
    pop_arr = np.zeros(n_herbs, dtype=np.float32)
    for h, c in pop.items():
        pop_arr[h] = math.log(1 + c)
    if pop_arr.max() > 0:
        pop_arr /= pop_arr.max()

    n = len(user_ids)
    cond = np.zeros((n, n_herbs), dtype=np.float32)
    combo = np.zeros((n, n_herbs), dtype=np.float32)
    for row, u in enumerate(user_ids):
        syms = features[u]
        for s in syms:
            total = sym_count[s]
            for h, c in sym2herb[s].items():
                cond[row, h] += c / total
        mx = cond[row].max()
        if mx > 0:
            cond[row] /= mx
        key = tuple(sorted(set(syms)))
        if key in combo2herb:
            combo[row, list(combo2herb[key].keys())] = 1.0
    return np.concatenate([cond, combo, np.tile(pop_arr, (n, 1))], axis=1)


def topk_from_scores(scores_mat, user_ids, k=5):
    pred = {}
    order = np.argsort(-scores_mat, axis=1, kind='stable')
    for row, u in enumerate(user_ids):
        pred[u] = order[row, :k].tolist()
    return pred


def combo_predict(user_ids, features, pop, sym2herb, sym_count, combo2herb):
    pred = {}
    pop_order = [h for h, _ in pop.most_common(805)]
    for u in user_ids:
        key = tuple(sorted(set(features[u])))
        c = combo2herb.get(key)
        if c is None:
            c = Counter()
            for s in key:
                c.update(sym2herb.get(s, {}))
            if not c:
                c = pop
        ranked = [h for h, _ in sorted(c.items(), key=lambda x: (-x[1], x[0]))[:5]]
        # 组合内药种不足5时用全局流行度补齐
        if len(ranked) < 5:
            for h in pop_order:
                if h not in ranked:
                    ranked.append(h)
                    if len(ranked) == 5:
                        break
        pred[u] = ranked
    return pred


class HerbNet(nn.Module):
    def __init__(self, n_symptoms, n_herbs, symptom_bge, in_prior_dim, dim=64, hidden=256, dropout=0.1):
        super().__init__()
        self.symptom_embedding = nn.Embedding(n_symptoms + 1, dim, padding_idx=0)
        self.register_buffer('symptom_bge', symptom_bge)  # [n_symptoms+1? -> padded at load]
        in_dim = dim + symptom_bge.shape[1] + in_prior_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_herbs),
        )

    def forward(self, symptoms, prior):
        mask = symptoms.ne(0).unsqueeze(-1)
        emb = self.symptom_embedding(symptoms)
        bge = self.symptom_bge[symptoms]
        pooled_emb = (emb * mask).sum(1) / mask.sum(1).clamp_min(1)
        pooled_bge = (bge * mask).sum(1) / mask.sum(1).clamp_min(1)
        x = torch.cat([pooled_emb, pooled_bge, prior], dim=-1)
        return self.mlp(x)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(8)

    data_dir = ROOT / 'data'
    feat_dir = ROOT / 'features'
    features, labels, test, herbs, symptoms = load_data(data_dir)
    n_herbs, n_symptoms = len(herbs), len(symptoms)
    print(f'herbs={n_herbs}, symptoms={n_symptoms}')

    train_ids, val_ids = split_train(labels, ratio=0.2, seed=SEED)
    va = {u: sorted(labels[u]) for u in val_ids}
    print(f'train={len(train_ids)}, val={len(val_ids)}')

    pop, sym2herb, sym_count, combo2herb = stat_priors(train_ids, features, labels, n_herbs)

    # --- 统计基线（同一划分）---
    print('popularity   F1@5 =', round(f1_at_5({u: [h for h, _ in pop.most_common(5)] for u in val_ids}, va), 4))
    print('combo-stats  F1@5 =', round(f1_at_5(combo_predict(val_ids, features, pop, sym2herb, sym_count, combo2herb), va), 4))

    # --- 先验特征 ---
    x_tr_prior = build_prior_features(train_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs)
    x_va_prior = build_prior_features(val_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs)

    sym_bge = load_name_vectors(feat_dir / 'symptom_bge.npz', symptoms, 'symptom_ids', 'symptom_names')
    sym_bge = torch.nn.functional.pad(sym_bge, (0, 0, 1, 0))  # 与 pad_symptoms 的 +1 偏移对齐

    x_tr = pad_symptoms(features, train_ids, DEVICE)
    x_va = pad_symptoms(features, val_ids, DEVICE)
    p_tr = torch.from_numpy(x_tr_prior).to(DEVICE)
    p_va = torch.from_numpy(x_va_prior).to(DEVICE)

    y_tr = torch.zeros((len(train_ids), n_herbs))
    for row, u in enumerate(train_ids):
        y_tr[row, sorted(labels[u])] = 1.0

    model = HerbNet(n_symptoms, n_herbs, sym_bge, in_prior_dim=x_tr_prior.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(SEED)

    epochs, batch = 40, 256
    best_f1, best_state, best_epoch = 0.0, None, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(len(train_ids))
        loss_sum = 0.0
        for st in range(0, len(perm), batch):
            idx = torch.as_tensor(perm[st:st + batch], dtype=torch.long)
            logits = model(x_tr[idx], p_tr[idx])
            loss = loss_fn(logits, y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(idx)
        model.eval()
        with torch.inference_mode():
            logits = model(x_va, p_va).numpy()
        pred = topk_from_scores(logits, val_ids)
        f1 = f1_at_5(pred, va)
        print(f'epoch {ep:02d}  loss={loss_sum/len(train_ids):.4f}  valF1@5={f1:.4f}  ({time.time()-t0:.0f}s)')
        if f1 > best_f1:
            best_f1, best_epoch = f1, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f'best model F1@5 = {best_f1:.4f} @ epoch {best_epoch}')
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        logits_va = model(x_va, p_va).numpy()

    # --- 模型分 + 组合统计分融合 ---
    stat = np.zeros((len(val_ids), n_herbs), dtype=np.float32)
    for row, u in enumerate(val_ids):
        key = tuple(sorted(set(features[u])))
        c = combo2herb.get(key)
        if c is None:
            c = Counter()
            for s in key:
                c.update(sym2herb.get(s, {}))
        if not c:
            c = pop
        mx = max(c.values())
        for h, v in c.items():
            stat[row, h] = math.log(1 + v) / math.log(1 + mx)

    m = logits_va.min(axis=1, keepdims=True)
    M = logits_va.max(axis=1, keepdims=True)
    norm_model = (logits_va - m) / np.maximum(M - m, 1e-9)

    for w in (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
        blended = w * norm_model + (1 - w) * stat
        pred = topk_from_scores(blended, val_ids)
        print(f'blend w(model)={w:.1f}  F1@5 = {f1_at_5(pred, va):.4f}')

    # 保存模型与先验，供最终全量训练复用
    out = ROOT / 'experiments' / 'artifacts_v1'
    out.mkdir(exist_ok=True)
    torch.save({'state_dict': best_state, 'dim': 64, 'hidden': 256, 'dropout': 0.1}, out / 'model.pt')
    print('saved to', out)


if __name__ == '__main__':
    main()
