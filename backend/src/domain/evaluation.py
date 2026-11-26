"""Compare the frozen classifier and majority baseline on the same held-out shifts."""

from collections import defaultdict
from statistics import mean

from .classifier import Prediction
from .settings import STATIONS


def class_statistics(truth: list[str], predicted: list[str], label: str) -> dict:
    support = truth.count(label)
    assigned = predicted.count(label)
    correct = sum(actual == guess == label for actual, guess in zip(truth, predicted, strict=True))
    return {
        "support": support,
        "predicted": assigned,
        "correct": correct,
        "precision": correct / assigned if assigned else 0.0,
        "recall": correct / support if support else 0.0,
        "f1": 2 * correct / (support + assigned) if support + assigned else 0.0,
    }


def class_comparison(truth: list[str], predicted: list[str], baseline_label: str) -> list[dict]:
    baseline = [baseline_label] * len(truth)
    return [
        {"fault_label": label, "station": station, **class_statistics(truth, predicted, label),
         "baseline_f1": class_statistics(truth, baseline, label)["f1"]}
        for label, station in STATIONS.items()
    ]


def scope_result(scope: str, entries: list[Prediction], baseline_label: str) -> tuple[dict, list[dict]]:
    truth = [entry.ticket["fault_label"] for entry in entries]
    predicted = [entry.label for entry in entries]
    classes = class_comparison(truth, predicted, baseline_label)
    accepted = [entry for entry in entries if entry.manual_reason is None]
    return {
        "scope": scope,
        "tickets": len(entries),
        "accuracy": sum(entry.label == entry.ticket["fault_label"] for entry in entries) / len(entries),
        "macro_f1": mean(row["f1"] for row in classes),
        "baseline_accuracy": truth.count(baseline_label) / len(entries),
        "baseline_macro_f1": mean(row["baseline_f1"] for row in classes),
        "manual_tickets": len(entries) - len(accepted),
        "auto_coverage": len(accepted) / len(entries),
        "auto_accuracy": (
            sum(entry.label == entry.ticket["fault_label"] for entry in accepted) / len(accepted)
            if accepted else None
        ),
    }, classes


def evaluate_holdout(entries: list[Prediction], baseline_label: str) -> tuple[list[dict], list[dict], list[str]]:
    shifts = defaultdict(list)
    for entry in entries:
        shifts[entry.ticket["shift_id"]].append(entry)
    combined, classes = scope_result("全部留出", entries, baseline_label)
    comparisons = [combined]
    for shift_id, tickets in sorted(shifts.items()):
        comparison, _ = scope_result(shift_id, tickets, baseline_label)
        comparisons.append(comparison)

    warnings = []
    absent = [row["fault_label"] for row in classes if not row["support"]]
    if absent:
        warnings.append(f"留出工单缺少 {', '.join(absent)} 的确认标签，相关召回率按 0 记。")
    if combined["auto_accuracy"] is None:
        warnings.append("本次留出工单均进入人工分流，自动分站准确率留空。")
    return comparisons, classes, warnings
