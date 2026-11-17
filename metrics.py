"""本地验证使用的逐记录平均 F1@5（macro F1）。"""
import math


def f1_at_5(predictions, truth):
    if not truth or set(predictions) != set(truth):
        raise ValueError('推荐结果必须覆盖全部验证记录')
    scores = []
    for uid, items in truth.items():
        target, ranked = set(items), predictions[uid]
        if not target or len(ranked) != 5 or len(set(ranked)) != 5:
            raise ValueError('真实集合不能为空，推荐列表必须恰好包含5个不同中药')
        hits = len(target.intersection(ranked))
        scores.append(2 * hits / (5 + len(target)))
    return math.fsum(scores) / len(truth)
