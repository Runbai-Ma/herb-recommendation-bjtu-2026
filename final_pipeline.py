"""最终方案 v6：层次收缩统计 + 名称语义/ID 协同 MLP 多样性集成的 Top-5 推荐（纯集成模式）。

方法概要
  10 个 MLP 成员（混合 lr 5e-4/1e-3 与 dropout 0.1/0.2）独立训练取 logits 平均。
  每个成员输入 = 症状 ID Embedding + BGE 名称向量均值池化 + 统计先验（条件概率 + log 流行度），
  两层 MLP(256d) 输出 805 维 logits。最终分数 = 集成平均 logits，取 Top-5。

评估协议与配方演进
  数据按 user_id 顺序划分为 train(0-27011) 与 test(27012-33764)，症状->处方映射存在
  顺序漂移（高频症状条件分布 top8 前后段重合率仅 0.66）。本地评估一律用顺序块划分
  （train 前 K% 统计/训练，后 (100-K)% 评估），并在两个切分点（80/20 与 75/25）上
  交叉确认，防止单一验证块的选择偏差。

  配方关键发现（exp15）：
  - 组合命中特征（805 维 multi-hot）在顺序漂移下是误导信号，去掉后单模型
    0.222 -> 0.232（+0.009，跨 10 seed 稳定），集成 0.2335；
  - v6 纯集成（不加统计信号融合）优于任何融合配置：主验证 80/20 上 0.2335 > 融合 0.2312，
    跨划分 75/25 上 0.2292 > 融合 0.2248——stat 信号（含 combo 收缩统计）已成为拖累；
  - 药名语义匹配特征（v4）平台已验证失效，默认关闭。

  平台成绩：v1 0.18145 -> v2 0.18341 -> v3 0.18484 -> v4 0.18255(失效回退) -> v5 0.18505。
  本地顺序校准落差 0.0424-0.0434。v6 本地 0.2335（80/20）/ 0.2292（75/25）。

运行：
  python final_pipeline.py                       # 10 成员，CPU 约 10 分钟
  python final_pipeline.py --output-dir result2  # 指定其他结果目录
"""
import argparse
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn

from data import load_data, load_name_vectors, pad_symptoms, write_submission
from metrics import f1_at_5

ROOT = Path(__file__).resolve().parent


def build_statistics(train_ids, features, labels):
    """全量训练统计量：流行度、症状->药、症状组合->药。"""
    pop = Counter()
    sym2herb = defaultdict(Counter)
    sym_count = Counter()
    combo2herb = defaultdict(Counter)
    for u in train_ids:
        pos = labels[u]
        pop.update(pos)
        combo2herb[tuple(sorted(set(features[u])))].update(pos)
        for s in set(features[u]):
            sym_count[s] += 1
            for h in pos:
                sym2herb[s][h] += 1
    return pop, sym2herb, sym_count, combo2herb


def build_prior_features(user_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs,
                         herb_bge=None, sym_bge_np=None, use_combo=True):
    """MLP 的统计先验输入：[条件概率分布, (组合命中指示), log 流行度] (+可选 药名语义匹配)。

    v6 起默认关闭 combo 命中特征：顺序漂移下它是误导信号（exp15，单模型 +0.009）。
    语义匹配特征单模型略负向（-0.004），但为集成提供错误多样性（v4 验证在平台失效，默认关闭）。
    """
    pop_arr = np.zeros(n_herbs, dtype=np.float32)
    for h, c in pop.items():
        pop_arr[h] = math.log(1 + c)
    if pop_arr.max() > 0:
        pop_arr /= pop_arr.max()

    n = len(user_ids)
    cond = np.zeros((n, n_herbs), dtype=np.float32)
    combo = np.zeros((n, n_herbs), dtype=np.float32) if use_combo else None
    sem = np.zeros((n, n_herbs), dtype=np.float32) if herb_bge is not None else None
    for row, u in enumerate(user_ids):
        syms = sorted(set(features[u]))
        for s in syms:
            total = sym_count[s]
            for h, c in sym2herb[s].items():
                cond[row, h] += c / total
        mx = cond[row].max()
        if mx > 0:
            cond[row] /= mx
        if combo is not None:
            key = tuple(sorted(syms))
            if key in combo2herb:
                combo[row, list(combo2herb[key].keys())] = 1.0
        if sem is not None:
            mean_vec = sym_bge_np[syms].mean(axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                sem[row] = herb_bge @ (mean_vec / norm)
    parts = [cond]
    if combo is not None:
        parts.append(combo)
    parts.append(np.tile(pop_arr, (n, 1)))
    if sem is not None:
        parts.append(sem)
    return np.concatenate(parts, axis=1)


def cond_scores(user_ids, features, sym2herb, sym_count, n_herbs):
    """纯条件概率分数（独立第三信号）：sum_s P(h|s)。"""
    n = len(user_ids)
    out = np.zeros((n, n_herbs), dtype=np.float32)
    for row, u in enumerate(user_ids):
        for s in set(features[u]):
            for h, c in sym2herb[s].items():
                out[row, h] += c / sym_count[s]
    return out


def shrink_scores(user_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs, tau):
    """层次收缩统计分数（主信号）。"""
    n = len(user_ids)
    out = np.zeros((n, n_herbs), dtype=np.float32)
    for row, u in enumerate(user_ids):
        cond = np.zeros(n_herbs, dtype=np.float32)
        for s in set(features[u]):
            for h, c in sym2herb[s].items():
                cond[h] += c / sym_count[s]
        cond_n = cond / max(cond.max(), 1e-9)
        c = combo2herb.get(tuple(sorted(set(features[u]))))
        if c is not None:
            arr = np.zeros(n_herbs, dtype=np.float32)
            mx = max(c.values())
            for h, v in c.items():
                arr[h] = math.log(1 + v) / math.log(1 + mx)
            seen = sum(c.values())
            alpha = seen / (seen + tau)
            out[row] = alpha * arr + (1 - alpha) * cond_n
        else:
            out[row] = cond_n
    return out


class HerbNet(nn.Module):
    """症状 ID Embedding + BGE 名称向量均值池化 + 统计先验 -> 两层 MLP -> 805 维 logits。"""

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


def normalize_rows(mat):
    mn = mat.min(axis=1, keepdims=True)
    mx = mat.max(axis=1, keepdims=True)
    return (mat - mn) / np.maximum(mx - mn, 1e-9)


def run(args):
    if min(args.epochs, args.batch_size, args.threads, args.dim) <= 0 or args.lr <= 0:
        raise ValueError('训练轮数、批量大小、线程数、维度和学习率必须为正数')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('结果目录非空，请用 --output-dir 指定新的结果目录')
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA 不可用，请使用 --device cpu')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)

    t0 = time.time()
    features, labels, test, herbs, symptoms = load_data(args.data_dir)
    n_herbs, n_symptoms = len(herbs), len(symptoms)
    train_ids = sorted(labels)
    test_ids = sorted(test)
    print(f'device={device}, train={len(train_ids)}, test={len(test_ids)}', flush=True)

    # ---- 1. 全量统计量与三路分数的原料 ----
    pop, sym2herb, sym_count, combo2herb = build_statistics(train_ids, features, labels)
    stat_test = shrink_scores(test_ids, features, pop, sym2herb, sym_count, combo2herb, n_herbs, args.tau)
    cond_test = cond_scores(test_ids, features, sym2herb, sym_count, n_herbs)

    # ---- 2. MLP 集成训练（全量数据，多配置多样性集成，固定轮数）----
    sym_bge = load_name_vectors(args.features_dir / 'symptom_bge.npz', symptoms,
                                'symptom_ids', 'symptom_names')
    sym_bge_np = sym_bge.numpy()  # 未 pad 版本：按症状 ID 直接索引
    sym_bge_np = sym_bge_np / np.maximum(np.linalg.norm(sym_bge_np, axis=1, keepdims=True), 1e-9)
    sym_bge = nn.functional.pad(sym_bge, (0, 0, 1, 0))  # pad 后供模型 embedding 索引使用
    herb_bge = load_name_vectors(args.features_dir / 'herb_bge.npz', herbs,
                                 'item_ids', 'herb_names').numpy()
    herb_bge = herb_bge / np.linalg.norm(herb_bge, axis=1, keepdims=True)

    x_train = pad_symptoms(features, train_ids, device)
    x_test = pad_symptoms(features, test_ids, device)
    sem_args = (herb_bge, sym_bge_np) if args.sem_feat else (None, None)
    p_train = torch.from_numpy(build_prior_features(train_ids, features, pop, sym2herb, sym_count,
                                                    combo2herb, n_herbs, *sem_args,
                                                    use_combo=bool(args.use_combo))).to(device)
    p_test = torch.from_numpy(build_prior_features(test_ids, features, pop, sym2herb, sym_count,
                                                   combo2herb, n_herbs, *sem_args,
                                                   use_combo=bool(args.use_combo))).to(device)

    y_train = torch.zeros((len(train_ids), n_herbs))
    for row, u in enumerate(train_ids):
        y_train[row, sorted(labels[u])] = 1.0
    y_train = y_train.to(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    members = [(int(s), float(lr), float(dp)) for s, lr, dp in
               (m.split(':') for m in args.members.split(','))]
    logits_members, history_all = [], []
    for seed, lr, dp in members:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        model = HerbNet(n_symptoms, n_herbs, sym_bge.to(device),
                        in_prior_dim=p_train.shape[1], dim=args.dim,
                        hidden=args.hidden, dropout=dp).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(seed)

        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            indices = rng.permutation(len(train_ids))
            loss_sum = 0.0
            for start in range(0, len(indices), args.batch_size):
                idx = torch.as_tensor(indices[start:start + args.batch_size], dtype=torch.long, device=device)
                loss = loss_fn(model(x_train[idx], p_train[idx]), y_train[idx])
                if not torch.isfinite(loss):
                    raise ValueError('训练损失含 NaN/Inf')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(idx)
            record = {'seed': seed, 'epoch': epoch, 'loss': loss_sum / len(train_ids)}
            history.append(record)
        history_all.extend(history)
        print(json.dumps({'seed': seed, 'lr': lr, 'dropout': dp, 'final_loss': history[-1]['loss']}), flush=True)
        model.eval()
        with torch.inference_mode():
            logits_members.append(model(x_test, p_test).cpu().numpy())
    (args.output_dir / 'history.json').write_text(json.dumps(history_all, indent=2), encoding='utf-8')
    torch.save({'members': members, 'epochs': args.epochs, 'dim': args.dim,
                'hidden': args.hidden, 'tau': args.tau,
                'fusion': [args.w_stat, args.w_model, args.w_cond],
                'loss_function': 'BCEWithLogitsLoss'}, args.output_dir / 'model.pt')
    logits_test = np.mean(logits_members, axis=0)

    # ---- 3. 三信号融合 ----
    fused_test = (args.w_stat * normalize_rows(stat_test)
                  + args.w_model * normalize_rows(logits_test)
                  + args.w_cond * normalize_rows(cond_test))

    predictions = {}
    for row, uid in enumerate(test_ids):
        predictions[uid] = np.argsort(-fused_test[row], kind='stable')[:5].tolist()
    output = args.output_dir / 'submission.csv'
    write_submission(output, predictions)
    print(f'提交文件已生成: {output}  ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='收缩统计+MLP多样性集成三信号融合的Top-5中药推荐')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--features-dir', type=Path, default=ROOT / 'features')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'result_final')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--members', type=str,
                        default='2026:5e-4:0.1,7:5e-4:0.1,42:5e-4:0.1,123:5e-4:0.1,777:5e-4:0.1,'
                                '314:5e-4:0.2,159:5e-4:0.2,265:5e-4:0.2,358:1e-3:0.1,979:1e-3:0.1',
                        help='逗号分隔的集成成员，格式 seed:lr:dropout')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4, help='默认学习率（被 members 覆盖）')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--sem-feat', type=int, default=0, help='1=启用药名语义匹配特征（v4，平台已验证失效），0=关闭')
    parser.add_argument('--use-combo', type=int, default=0, help='1=启用组合命中特征（v3-v5），0=关闭（v6，顺序漂移下为误导信号）')
    parser.add_argument('--tau', type=float, default=30.0, help='组合统计向条件概率的收缩系数')
    parser.add_argument('--w-stat', type=float, default=0.0, help='融合权重：统计信号（v6 纯集成模式为 0）')
    parser.add_argument('--w-model', type=float, default=1.0, help='融合权重：MLP 集成信号（v6 纯集成模式为 1）')
    parser.add_argument('--w-cond', type=float, default=0.0, help='融合权重：条件概率信号（v6 纯集成模式为 0）')
    run(parser.parse_args())
