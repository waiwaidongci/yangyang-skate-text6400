"""Domain entry points used by the API, seed command and background worker."""

from .classifier import fit_routing_model
from .evaluation import evaluate_holdout
from .fixtures import demo_rows
from .records import normalize_rows
from .reports import holdout_by_label, holdout_by_shift, holdout_predictions, pending_routes, training_profile
from .settings import SPEC, parse_settings

__all__ = ["SPEC", "demo_rows", "normalize_rows", "analyze"]


def analyze(rows: list[dict], settings: dict) -> dict:
    options = parse_settings(settings)
    cohorts = {"train": [], "holdout": [], "pending": []}
    for row in sorted(normalize_rows(rows), key=lambda row: row["id"]):
        cohorts[row["split"]].append(row)
    train = cohorts["train"]
    model = fit_routing_model(train, options["alpha"])
    threshold = options["confidence_threshold"]
    holdout = model.score(cohorts["holdout"], threshold)
    pending = model.score(cohorts["pending"], threshold)
    by_shift, by_label, warnings = evaluate_holdout(holdout, model.baseline_label)
    overall = by_shift[0]
    manual = sum(entry.manual_reason is not None for entry in pending)
    if not pending:
        warnings.append("本次台账无待修工单，交接工位名单为空。")
    return {
        "summary": (
            f"使用 {len({row['shift_id'] for row in train})} 个历史班次的 {len(train)} 条已确认工单拟合。"
            f"对 {len(by_shift) - 1} 个留出班次的 {len(holdout)} 条工单逐单盲测。"
            f"本班 {len(pending)} 条待修工单中，{len(pending) - manual} 条建议直达工位，{manual} 条待柜台补问。"
        ),
        "metrics": {
            "冻结词表大小": model.vocabulary_size,
            "留出模型准确率": round(overall["accuracy"], 6),
            "留出类别宏 F1": round(overall["macro_f1"], 6),
            "留出先验基线宏 F1": round(overall["baseline_macro_f1"], 6),
            "留出自动分站覆盖率": round(overall["auto_coverage"], 6),
            "待修人工分流数": manual,
        },
        "tables": [
            pending_routes(pending), holdout_by_shift(by_shift), holdout_by_label(by_label),
            holdout_predictions(holdout, model.baseline_label), training_profile(model),
        ],
        "warnings": warnings,
    }
