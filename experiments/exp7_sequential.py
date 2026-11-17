"""实验7：顺序块协议下的 MLP 评估 + 三信号融合重调 + v2 提交原料。

协议：train 前 80%（0-21609）训练/统计，后 20%（21610-27011）评估——模拟平台按顺序划分。
产出：最佳融合配置、测试集三路分数（用全量 train 重建统计量 + 全量训练 MLP）。
"""
import sys
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import load_data, load_name_vectors, pad_symptoms  # noqa: E402
from metrics import f1_at_5  # noqa: E402
from exp1_prior_mlp import HerbNet, build_prior_features  # noqa: E402

SEED = 2026


def build_stats(ids, features, labels):
    pop, sym2herb, sym_count, combo2herb = Counter(), defaultdict(Counter), Counter(), defaultdict(Counter)
    for u in ids:
        pos = labels[u]
        pop.update(pos)
        combo2herb[tuple(sorted(set(features[u])))].update(pos)
        for s in set(features[u]):
            sym_count[s] += 1
            for h in pos:
                sym2herb[s][h] += 1
    return pop, sym2herb, sym_count, combo2herb


def shrink_and_cond(ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs, tau):
    n = len(ids)
    stat = np.zeros((n, n_herbs), dtype=np.float32)
    cond_all = np.zeros((n, n_herbs), dtype=np.float32)
    for row, u in enumerate(ids):
        cond = np.zeros(n_herbs, dtype=np.float32)
        for s in set(features[u]):
            for h, c in sym2herb[s].items():
                cond[h] += c / sym_count[s]
        cond_n = cond / max(cond.max(), 1e-9)
        cond_all[row] = cond_n
        c = combo2herb.get(tuple(sorted(set(features[u]))))
        if c is not None:
            arr = np.zeros(n_herbs, dtype=np.float32)
            mx = max(c.values())
            for h, v in c.items():
                arr[h] = math.log(1 + v) / math.log(1 + mx)
            seen = sum(c.values())
            alpha = seen / (seen + tau)
            stat[row] = alpha * arr + (1 - alpha) * cond_n
        else:
            stat[row] = cond_n
    return stat, cond_all


def norm_rows(mat):
    mn = mat.min(axis=1, keepdims=True)
    mx = mat.max(axis=1, keepdims=True)
    return (mat - mn) / np.maximum(mx - mn, 1e-9)


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(8)

    features, labels, test, herbs, symptoms = load_data(ROOT / 'data')
    n_herbs, n_symptoms = len(herbs), len(symptoms)
    all_train = sorted(labels)
    tr_ids, va_ids = all_train[:21610], all_train[21610:]
    va = {u: sorted(labels[u]) for u in va_ids}

    sym_bge = load_name_vectors(ROOT / 'features' / 'symptom_bge.npz',
                                symptoms, 'symptom_ids', 'symptom_names')
    sym_bge = torch.nn.functional.pad(sym_bge, (0, 0, 1, 0))

    def train_mlp(stat_train_ids):
        pop, sym2herb, sym_count, combo2herb = build_stats(stat_train_ids, features, labels)
        x_tr_prior = build_prior_features(stat_train_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs)
        x_tr = pad_symptoms(features, stat_train_ids, 'cpu')
        y_tr = torch.zeros((len(stat_train_ids), n_herbs))
        for row, u in enumerate(stat_train_ids):
            y_tr[row, sorted(labels[u])] = 1.0
        model = HerbNet(n_symptoms, n_herbs, sym_bge, in_prior_dim=x_tr_prior.shape[1]).to('cpu')
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(SEED)
        model.train()
        for ep in range(1, 21):
            perm = rng.permutation(len(stat_train_ids))
            for st in range(0, len(perm), 256):
                idx = torch.as_tensor(perm[st:st + 256], dtype=torch.long)
                loss = loss_fn(model(x_tr[idx], torch.from_numpy(x_tr_prior)[idx]), y_tr[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        model.eval()
        return model, (pop, sym2herb, sym_count, combo2herb)

    # ---- 顺序协议：前80%训练 ----
    t0 = time.time()
    model, stats = train_mlp(tr_ids)
    print(f'MLP trained on first 80% ({time.time()-t0:.0f}s)', flush=True)

    pop, sym2herb, sym_count, combo2herb = stats
    stat_va, cond_va = shrink_and_cond(va_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs, tau=30)
    x_va = pad_symptoms(features, va_ids, 'cpu')
    p_va_prior = build_prior_features(va_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs)
    with torch.inference_mode():
        logits_va = model(x_va, torch.from_numpy(p_va_prior)).numpy()

    stat_n, cond_n, model_n = norm_rows(stat_va), norm_rows(cond_va), norm_rows(logits_va)

    def fused(w_s, w_m, w_c):
        return w_s * stat_n + w_m * model_n + w_c * cond_n

    best = (None, 0.0)
    for w_s in (0.3, 0.4, 0.5, 0.6, 0.7, 1.0):
        for w_m in (0.0, 0.1, 0.2, 0.3, 0.4):
            for w_c in (0.0, 0.1, 0.2, 0.3):
                if abs(w_s + w_m + w_c - 1.0) > 1e-9:
                    continue
                sc = fused(w_s, w_m, w_c)
                pred = {u: np.argsort(-sc[i], kind='stable')[:5].tolist() for i, u in enumerate(va_ids)}
                f1 = f1_at_5(pred, va)
                if f1 > best[1]:
                    best = ((w_s, w_m, w_c), f1)
                if w_m == 0 and w_c == 0:
                    print(f'stat-only w_s={w_s}: {f1:.4f}')
                elif w_s == 0 and w_c == 0:
                    print(f'model-only w_m={w_m}: {f1:.4f}')
    (w_s, w_m, w_c), f1_best = best
    print(f'best fusion: stat={w_s} model={w_m} cond={w_c}  F1@5 = {f1_best:.4f}')

    # ---- 全量重建 + 预测 test ----
    model_full, stats_full = train_mlp(all_train)
    pop, sym2herb, sym_count, combo2herb = stats_full
    test_ids = sorted(test)
    stat_te, cond_te = shrink_and_cond(test_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs, tau=30)
    x_te = pad_symptoms(features, test_ids, 'cpu')
    p_te_prior = build_prior_features(test_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs)
    with torch.inference_mode():
        logits_te = model_full(x_te, torch.from_numpy(p_te_prior)).numpy()

    fused_te = w_s * norm_rows(stat_te) + w_m * norm_rows(logits_te) + w_c * norm_rows(cond_te)
    out = ROOT / 'experiments' / 'artifacts_v2'
    out.mkdir(exist_ok=True)
    np.save(out / 'fused_test.npy', fused_te)
    np.save(out / 'test_ids.npy', np.array(test_ids))
    torch.save({'state_dict': model_full.state_dict()}, out / 'model_full.pt')
    print('v2 test scores saved.', f'({time.time()-t0:.0f}s total)')


if __name__ == '__main__':
    main()
