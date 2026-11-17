"""实验14：V6 集成配方探索——双塔成员 + 特征消融成员 + 跨划分验证。

成员池（全部无语义特征，v5 已验证特征集）：
  - 10 x MLP（v5 同款：lr5e-4/lr1e-3 x dp0.1/0.2）
  - 2 x 双塔（SymTower/HerbTower 内积，无先验特征，错误模式互补）
  - 2 x MLP 特征消融（无 pop / 无 combo）
评估：顺序验证（前80%训/后20%评）+ 跨划分确认（前75%训/后25%评）。
"""
import sys
import math
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

MLP_MEMBERS = [(2026, 5e-4, 0.1), (7, 5e-4, 0.1), (42, 5e-4, 0.1), (123, 5e-4, 0.1), (777, 5e-4, 0.1),
               (314, 5e-4, 0.2), (159, 5e-4, 0.2), (265, 5e-4, 0.2),
               (358, 1e-3, 0.1), (979, 1e-3, 0.1)]
TWIN_MEMBERS = [(601, 64, 128), (602, 64, 64)]           # (seed, dim, hidden)
ABLATION_MEMBERS = [(701, 'no_pop'), (702, 'no_combo')]  # (seed, subset)


class TwinTower(nn.Module):
    """双塔：症状塔/药塔各自 ID emb + BGE -> FC -> 内积。无先验特征（保持与 MLP 互补）。"""

    def __init__(self, n_symptoms, n_herbs, symptom_bge, herb_bge, dim=64, hidden=128):
        super().__init__()
        self.symptom_embedding = nn.Embedding(n_symptoms + 1, dim, padding_idx=0)
        self.herb_embedding = nn.Embedding(n_herbs, dim)
        self.register_buffer('symptom_bge', symptom_bge)
        self.register_buffer('herb_bge', herb_bge)
        self.sym_tower = nn.Sequential(
            nn.Linear(dim + symptom_bge.shape[1], hidden), nn.ReLU(),
            nn.Linear(hidden, dim),
        )
        self.herb_tower = nn.Sequential(
            nn.Linear(dim + herb_bge.shape[1], hidden), nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, symptoms):
        mask = symptoms.ne(0).unsqueeze(-1)
        sym = torch.cat([self.symptom_embedding(symptoms), self.symptom_bge[symptoms]], dim=-1)
        pooled = (sym * mask).sum(1) / mask.sum(1).clamp_min(1)
        sym_vec = self.sym_tower(pooled)
        herb_in = torch.cat([self.herb_embedding.weight, self.herb_bge], dim=-1)
        herb_vec = self.herb_tower(herb_in)
        return sym_vec @ herb_vec.T


class HerbNet(nn.Module):
    """MLP 成员（支持特征子集：subset dict 指定 cond/combo/pop/sem 哪些进输入）。"""

    def __init__(self, n_symptoms, n_herbs, symptom_bge, in_prior_dim, dim=64, hidden=256, dropout=0.1):
        super().__init__()
        self.symptom_embedding = nn.Embedding(n_symptoms + 1, dim, padding_idx=0)
        self.register_buffer('symptom_bge', symptom_bge)
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
        return self.mlp(torch.cat([pooled_emb, pooled_bge, prior], dim=-1))


def build_prior_subset(user_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs,
                       use_cond=True, use_combo=True, use_pop=True):
    pop_arr = np.zeros(n_herbs, dtype=np.float32)
    for h, c in pop.items():
        pop_arr[h] = math.log(1 + c)
    if pop_arr.max() > 0:
        pop_arr /= pop_arr.max()
    n = len(user_ids)
    cond = np.zeros((n, n_herbs), dtype=np.float32)
    combo = np.zeros((n, n_herbs), dtype=np.float32)
    for row, u in enumerate(user_ids):
        for s in set(features[u]):
            total = sym_count[s]
            for h, c in sym2herb[s].items():
                cond[row, h] += c / total
        mx = cond[row].max()
        if mx > 0:
            cond[row] /= mx
        key = tuple(sorted(set(features[u])))
        if key in combo2herb:
            combo[row, list(combo2herb[key].keys())] = 1.0
    parts = []
    if use_cond:
        parts.append(cond)
    if use_combo:
        parts.append(combo)
    if use_pop:
        parts.append(np.tile(pop_arr, (n, 1)))
    return np.concatenate(parts, axis=1)


def main():
    torch.set_num_threads(8)
    features, labels, test, herbs, symptoms = load_data(ROOT / 'data')
    n_herbs, n_symptoms = len(herbs), len(symptoms)
    all_train = sorted(labels)

    sym_bge = load_name_vectors(ROOT / 'features' / 'symptom_bge.npz',
                                symptoms, 'symptom_ids', 'symptom_names')
    sym_bge = torch.nn.functional.pad(sym_bge, (0, 0, 1, 0))
    herb_bge = load_name_vectors(ROOT / 'features' / 'herb_bge.npz',
                                 herbs, 'item_ids', 'herb_names')

    for cut_ratio, tag in ((0.80, '主验证80/20'), (0.75, '跨划分75/25')):
        t0 = time.time()
        cut = int(round(len(all_train) * cut_ratio))
        tr_ids, va_ids = all_train[:cut], all_train[cut:]
        va = {u: sorted(labels[u]) for u in va_ids}
        stats = build_statistics(tr_ids, features, labels)
        pop, sym2herb, sym_count, combo2herb = stats

        x_va = pad_symptoms(features, va_ids, 'cpu')
        logits = {}  # name -> [n_va, 805]

        # ---- MLP 10 成员（全特征）----
        prior_a = build_prior_subset(tr_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs)
        prior_b = build_prior_subset(va_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs)
        x_a = pad_symptoms(features, tr_ids, 'cpu')
        p_a, p_b = torch.from_numpy(prior_a), torch.from_numpy(prior_b)
        y_a = torch.zeros((len(tr_ids), n_herbs))
        for row, u in enumerate(tr_ids):
            y_a[row, sorted(labels[u])] = 1.0
        mlp_logit_list = []
        for seed, lr, dp in MLP_MEMBERS:
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
                    lg = model(x_va, p_b).numpy()
                pred = {u: np.argsort(-lg[i], kind='stable')[:5].tolist() for i, u in enumerate(va_ids)}
                f1 = f1_at_5(pred, va)
                if f1 > best_f1:
                    best_f1 = f1
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            model.eval()
            with torch.inference_mode():
                mlp_logit_list.append(model(x_va, p_b).numpy())
            print(f'[{tag}] mlp seed={seed}: {best_f1:.4f} ({time.time()-t0:.0f}s)', flush=True)
        logits['mlp10'] = np.mean(mlp_logit_list, axis=0)

        # ---- 特征消融 2 成员 ----
        for seed, subset in ABLATION_MEMBERS:
            kw = dict(use_cond=True, use_combo=(subset != 'no_combo'), use_pop=(subset != 'no_pop'))
            pa = build_prior_subset(tr_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs, **kw)
            pb = build_prior_subset(va_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs, **kw)
            pa_t, pb_t = torch.from_numpy(pa), torch.from_numpy(pb)
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = HerbNet(n_symptoms, n_herbs, sym_bge, in_prior_dim=pa.shape[1])
            opt = torch.optim.Adam(model.parameters(), lr=5e-4)
            loss_fn = nn.BCEWithLogitsLoss()
            rng = np.random.default_rng(seed)
            best_f1, best_state = 0.0, None
            for ep in range(1, 31):
                model.train()
                perm = rng.permutation(len(tr_ids))
                for st in range(0, len(perm), 256):
                    idx = torch.as_tensor(perm[st:st + 256], dtype=torch.long)
                    loss = loss_fn(model(x_a[idx], pa_t[idx]), y_a[idx])
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                model.eval()
                with torch.inference_mode():
                    lg = model(x_va, pb_t).numpy()
                pred = {u: np.argsort(-lg[i], kind='stable')[:5].tolist() for i, u in enumerate(va_ids)}
                f1 = f1_at_5(pred, va)
                if f1 > best_f1:
                    best_f1 = f1
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            model.eval()
            with torch.inference_mode():
                logits[f'abl_{subset}'] = model(x_va, pb_t).numpy()
            print(f'[{tag}] ablation {subset} seed={seed}: {best_f1:.4f} ({time.time()-t0:.0f}s)', flush=True)

        # ---- 双塔 2 成员 ----
        for seed, dim, hidden in TWIN_MEMBERS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = TwinTower(n_symptoms, n_herbs, sym_bge, herb_bge, dim=dim, hidden=hidden)
            opt = torch.optim.Adam(model.parameters(), lr=5e-4)
            loss_fn = nn.BCEWithLogitsLoss()
            rng = np.random.default_rng(seed)
            best_f1, best_state = 0.0, None
            for ep in range(1, 31):
                model.train()
                perm = rng.permutation(len(tr_ids))
                for st in range(0, len(perm), 256):
                    idx = torch.as_tensor(perm[st:st + 256], dtype=torch.long)
                    loss = loss_fn(model(x_a[idx]), y_a[idx])
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                model.eval()
                with torch.inference_mode():
                    lg = model(x_va).numpy()
                pred = {u: np.argsort(-lg[i], kind='stable')[:5].tolist() for i, u in enumerate(va_ids)}
                f1 = f1_at_5(pred, va)
                if f1 > best_f1:
                    best_f1 = f1
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            model.eval()
            with torch.inference_mode():
                logits[f'twin_{seed}'] = model(x_va).numpy()
            print(f'[{tag}] twin seed={seed} dim={dim}: {best_f1:.4f} ({time.time()-t0:.0f}s)', flush=True)

        # ---- 集成组合对比 + 融合网格 ----
        stat_va = shrink_scores(va_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs, 30.0)
        cond_va = cond_scores(va_ids, features, sym2herb, sym_count, n_herbs)

        def fuse(w_s, w_m, w_c, ens):
            sc = w_s * normalize_rows(stat_va) + w_m * normalize_rows(ens) + w_c * normalize_rows(cond_va)
            pred = {u: np.argsort(-sc[i], kind='stable')[:5].tolist() for i, u in enumerate(va_ids)}
            return f1_at_5(pred, va)

        combos = {
            'v5(mlp10)': logits['mlp10'],
            '+abl2': np.mean([logits['mlp10'], logits['abl_no_pop'], logits['abl_no_combo']], axis=0),
            '+twin2': np.mean([logits['mlp10'], logits['twin_601'], logits['twin_602']], axis=0),
            'v6(10+2+2)': np.mean([logits['mlp10'], logits['abl_no_pop'], logits['abl_no_combo'],
                                   logits['twin_601'], logits['twin_602']], axis=0),
        }
        results = {}
        for name, ens in combos.items():
            f1_ens = f1_at_5({u: np.argsort(-ens[i], kind='stable')[:5].tolist() for i, u in enumerate(va_ids)}, va)
            best = (None, 0.0)
            for w_s in (0.3, 0.4, 0.5):
                for w_m in (0.4, 0.5, 0.6, 0.7):
                    for w_c in (0.0, 0.1, 0.2):
                        if abs(w_s + w_m + w_c - 1.0) > 1e-9:
                            continue
                        f1 = fuse(w_s, w_m, w_c, ens)
                        if f1 > best[1]:
                            best = ((w_s, w_m, w_c), f1)
            results[name] = (f1_ens, best)
            print(f'[{tag}] {name}: model-only={f1_ens:.4f}, best fusion {best[0]} = {best[1]:.4f}', flush=True)

        # 落盘 logits 供最终组装
        out = ROOT / 'experiments' / 'artifacts_v6'
        out.mkdir(exist_ok=True)
        np.savez_compressed(out / f'val_logits_{tag}.npz',
                            **{k: v for k, v in logits.items()},
                            stat=stat_va, cond=cond_va,
                            va_ids=np.array(va_ids))
        print(f'[{tag}] done ({time.time()-t0:.0f}s)\n', flush=True)


if __name__ == '__main__':
    main()
