"""Keep the audit and handoff tables flat for JSON and CSV consumers."""

from .classifier import Prediction, RoutingModel
from .settings import FIELDS, STATIONS

PREDICTION_FIELDS = {
    "id": FIELDS["id"], "shift_id": FIELDS["shift_id"], "message": FIELDS["message"],
    "predicted_label": "建议故障类别", "confidence": "最高置信分数", "confidence_gap": "前两类分差",
    "score_blade": "刃磨得分", "score_boot": "鞋体得分", "score_fastener": "固定件得分",
    "matched_ngrams": "历史字符片段命中数", "route": "分流方式", "station": "交接工位",
    "reason": "分流依据",
}


def table(key: str, title: str, labels: dict[str, str], rows: list[dict]) -> dict:
    return {
        "key": key, "title": title, "columns": list(labels), "labels": labels,
        "rows": [{field: round(row[field], 6) if isinstance(row[field], float) else row[field]
                  for field in labels} for row in rows],
    }


def prediction_values(entry: Prediction) -> dict:
    return {
        **{key: entry.ticket[key] for key in ("id", "shift_id", "message")},
        "predicted_label": entry.label,
        "confidence": entry.confidence,
        "confidence_gap": entry.gap,
        **{f"score_{label}": score for label, score in entry.probabilities.items()},
        "matched_ngrams": entry.matched_ngrams,
        "route": "manual" if entry.manual_reason else "auto",
        "station": entry.station,
        "reason": entry.manual_reason or "达到自动分站置信线",
    }


def pending_routes(entries: list[Prediction]) -> dict:
    ordered = sorted(entries, key=lambda entry: (
        entry.manual_reason is None, entry.confidence if entry.manual_reason else -entry.confidence,
        entry.ticket["id"],
    ))
    return table("pending_routes", "待修冰鞋交接工位清单", {"position": "交班顺序", **PREDICTION_FIELDS}, [
        {"position": index, **prediction_values(entry)} for index, entry in enumerate(ordered, start=1)
    ])


def holdout_predictions(entries: list[Prediction], baseline_label: str) -> dict:
    labels = {**PREDICTION_FIELDS, "fault_label": FIELDS["fault_label"], "baseline_label": "先验基线类别"}
    return table("holdout_predictions", "留出班次逐单对照", labels, [
        {**prediction_values(entry), "fault_label": entry.ticket["fault_label"], "baseline_label": baseline_label}
        for entry in entries
    ])


def holdout_by_shift(rows: list[dict]) -> dict:
    return table("holdout_by_shift", "整班留出表现", {
        "scope": "班次范围", "tickets": "工单数", "accuracy": "模型准确率",
        "macro_f1": "模型宏 F1", "baseline_accuracy": "先验基线准确率",
        "baseline_macro_f1": "先验基线宏 F1", "manual_tickets": "人工分流数",
        "auto_coverage": "自动分站覆盖率", "auto_accuracy": "自动分站子集准确率",
    }, rows)


def holdout_by_label(rows: list[dict]) -> dict:
    return table("holdout_by_label", "留出类别表现", {
        "fault_label": "类别代码", "station": "负责工位", "support": "确认数", "predicted": "建议数",
        "correct": "建议正确数", "precision": "精确率", "recall": "召回率", "f1": "类别 F1",
        "baseline_f1": "先验基线类别 F1",
    }, rows)


def training_profile(model: RoutingModel) -> dict:
    total = model.training_counts.total()
    return table("training_profile", "冻结训练类别分布", {
        "fault_label": "类别代码", "station": "负责工位", "tickets": "历史确认工单数",
        "prior": "历史类别占比", "baseline_label": "先验基线类别",
    }, [
        {"fault_label": label, "station": station, "tickets": model.training_counts[label],
         "prior": model.training_counts[label] / total, "baseline_label": model.baseline_label}
        for label, station in STATIONS.items()
    ])
