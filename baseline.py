import argparse
import json
import os
from pathlib import Path
import random

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from data import load_data, load_name_vectors, make_labels, pad_symptoms, write_submission
from model import HerbRecommender

ROOT = Path(__file__).resolve().parent


@torch.inference_mode()
def recommend(model, features, user_ids, batch_size, device):
    model.eval()
    predictions = {}
    for start in range(0, len(user_ids), batch_size):
        ids = user_ids[start:start + batch_size]
        scores = model(pad_symptoms(features, ids, device)).cpu().numpy()
        if not np.isfinite(scores).all():
            raise ValueError('预测分数含 NaN/Inf')
        # 同分按药品ID升序，推荐列表按得分降序。
        ranked = np.argsort(-scores, axis=1, kind='stable')[:, :5]
        predictions.update({uid: row.tolist() for uid, row in zip(ids, ranked)})
    return predictions


def run(args):
    if min(args.epochs, args.batch_size, args.threads, args.dim) <= 0 or args.lr <= 0:
        raise ValueError('训练轮数、批量大小、线程数、维度和学习率必须为正数')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('结果目录非空，请用 --output-dir 指定新的结果目录')
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA 不可用，请使用 --device cpu')
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)

    features, labels, test, herbs, symptoms = load_data(args.data_dir)
    train_ids = sorted(labels)

    x_train = pad_symptoms(features, train_ids, device)
    symptom_names = load_name_vectors(args.features_dir / 'symptom_bge.npz', symptoms,
                                      'symptom_ids', 'symptom_names')
    herb_names = load_name_vectors(args.features_dir / 'herb_bge.npz', herbs, 'item_ids', 'herb_names')
    model = HerbRecommender(symptom_names, herb_names, args.dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f'device={device}, train={len(train_ids)}, test={len(test)}', flush=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = rng.permutation(len(train_ids))
        loss_sum = 0.0
        for start in range(0, len(indices), args.batch_size):
            batch = indices[start:start + args.batch_size]
            user_ids = [train_ids[int(i)] for i in batch]
            logits = model(x_train[torch.as_tensor(batch, device=device)])
            target = make_labels(user_ids, labels, len(herbs), device)
            # 显式标签：真实处方中的药为1，其余为0。sigmoid由损失函数内部稳定计算。
            loss = loss_fn(logits, target)
            if not torch.isfinite(loss):
                raise ValueError('训练损失含 NaN/Inf')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
        record = {'epoch': epoch, 'loss': loss_sum / len(train_ids)}
        history.append(record)
        print(json.dumps(record), flush=True)
    (args.output_dir / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    torch.save({'state_dict': model.state_dict(), 'epochs': args.epochs,
                'dim': args.dim, 'seed': args.seed, 'loss_function': 'BCEWithLogitsLoss'},
               args.output_dir / 'model.pt')
    predictions = recommend(model, test, sorted(test), args.batch_size, device)
    output = args.output_dir / 'submission.csv'
    write_submission(output, predictions)
    print(f'提交文件已生成: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ID与名称拼接、两层FC双塔的Top-5推荐基线')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--features-dir', type=Path, default=ROOT / 'features')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'result')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--threads', type=int, default=4)
    run(parser.parse_args())
