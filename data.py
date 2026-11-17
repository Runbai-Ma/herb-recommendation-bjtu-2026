"""读取数据和名称向量，构造0/1标签，生成平台提交CSV。"""
import csv
import json
from pathlib import Path

import numpy as np
import torch


def read_csv(path, columns):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, strict=True)
        if reader.fieldnames != columns:
            raise ValueError(f'{path}: 表头应为 {columns}')
        rows = list(reader)
    if not rows or any(None in row or None in row.values() for row in rows):
        raise ValueError(f'{path}: 文件为空或列数错误')
    return rows


def integer(value):
    if not value.isascii() or not value.isdecimal():
        raise ValueError(f'非法编号: {value}')
    return int(value)


def id_list(value, upper):
    ids = json.loads(value)
    if (not isinstance(ids, list) or not ids or
            any(type(i) is not int or not 0 <= i < upper for i in ids)):
        raise ValueError('列表必须是非空、合法编号的 JSON 整数数组')
    if len(set(ids)) != len(ids):
        raise ValueError('列表不能包含重复编号')
    return ids


def vocabulary(path, id_column, name_column):
    values = {}
    for row in read_csv(path, [id_column, name_column]):
        idx, name = integer(row[id_column]), row[name_column]
        if idx in values or not name.strip():
            raise ValueError('词表包含重复编号或空名称')
        values[idx] = name
    if set(values) != set(range(len(values))) or len(set(values.values())) != len(values):
        raise ValueError('词表须从0连续编号，且名称不能重复')
    return [values[i] for i in range(len(values))]


def load_data(directory):
    directory = Path(directory)
    herbs = vocabulary(directory / 'items.csv', 'item_id', 'herb_name')
    symptoms = vocabulary(directory / 'symptoms.csv', 'symptom_id', 'symptom_name')
    features, labels, test = {}, {}, {}
    for filename, target in [('user_features.csv', features), ('test.csv', test)]:
        for row in read_csv(directory / filename, ['user_id', 'symptom_ids']):
            uid = integer(row['user_id'])
            if uid in target:
                raise ValueError(f'{filename}: 重复 user_id {uid}')
            target[uid] = id_list(row['symptom_ids'], len(symptoms))
    for row in read_csv(directory / 'train_interactions.csv', ['user_id', 'item_id']):
        uid, item = integer(row['user_id']), integer(row['item_id'])
        if uid not in features or uid in test or not 0 <= item < len(herbs):
            raise ValueError('训练交互包含未知用户、测试用户或非法药品')
        positive = labels.setdefault(uid, set())
        if item in positive:
            raise ValueError('train_interactions.csv包含重复交互')
        positive.add(item)
    if set(features) != set(labels) | set(test):
        raise ValueError('全局用户必须恰好由训练交互用户和测试用户组成')
    if any(uid not in features or values != features[uid] for uid, values in test.items()):
        raise ValueError('测试症状与user_features.csv不一致')
    if len(labels) < 2 or len(herbs) < 5:
        raise ValueError('需要至少2条训练记录、5种候选药品')
    return features, labels, test, herbs, symptoms


def load_name_vectors(path, names, id_key, name_key):
    """按词表ID顺序加载冻结的名称向量。"""
    with np.load(path, allow_pickle=False) as artifact:
        if artifact[id_key].tolist() != list(range(len(names))) or artifact[name_key].tolist() != names:
            raise ValueError(f'{path}: ID或名称与词表不一致')
        vectors = artifact['vectors'].astype(np.float32)
    if vectors.shape != (len(names), 512) or not np.isfinite(vectors).all():
        raise ValueError(f'{path}: 名称向量应为有效的512维向量')
    return torch.from_numpy(vectors)


def split_train(labels, ratio=0.1, seed=2026):
    if not 0 < ratio < 1:
        raise ValueError('val-ratio 必须在0和1之间')
    ids = np.array(sorted(labels), dtype=np.int64)
    np.random.default_rng(seed).shuffle(ids)
    count = min(len(ids) - 1, max(1, int(round(len(ids) * ratio))))
    return sorted(ids[count:].tolist()), sorted(ids[:count].tolist())


def pad_symptoms(features, user_ids, device):
    tensor = torch.zeros((len(user_ids), max(len(features[u]) for u in user_ids)), dtype=torch.long)
    for row, uid in enumerate(user_ids):
        # 数据中的0是有效症状；整体加1，将模型内部0留作 padding。
        tensor[row, :len(features[uid])] = torch.tensor(features[uid]) + 1
    return tensor.to(device)


def make_labels(user_ids, labels, num_items, device):
    """每条记录构造完整的0/1药品标签：[batch_size, num_items]。"""
    target = torch.zeros((len(user_ids), num_items), dtype=torch.float32, device=device)
    for row, uid in enumerate(user_ids):
        target[row, sorted(labels[uid])] = 1.0
    return target


def write_submission(path, predictions):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['user_id', 'item_ids'])
        for uid, items in predictions.items():
            if len(items) != 5 or len(set(items)) != 5:
                raise ValueError('每条记录必须推荐5个不同中药')
            writer.writerow([uid, json.dumps(items, separators=(',', ':'))])
