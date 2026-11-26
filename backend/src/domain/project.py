"""Domain entry points used by the API, seed command and background worker."""

from .classifier import fit_routing_model
from .evaluation import evaluate_holdout
from .fixtures import demo_rows
from .records import normalize_rows
from .reports import (
    holdout_by_label,
    holdout_by_shift,
    holdout_predictions,
    pending_routes,
    prediction_values,
    table,
    training_profile,
)
from .settings import SPEC, STATIONS, parse_settings
from .thresholds import build_handoff_plan, extract_policy, optimize_station_thresholds, shift_effects

__all__ = ["SPEC", "demo_rows", "normalize_rows", "analyze"]


def _alias_table(source: dict, key: str, title: str | None = None) -> dict:
    return {**source, "key": key, **({"title": title} if title else {})}


def _threshold_tables(planning, plan: dict) -> list[dict]:
    lines = plan.get("thresholds")
    threshold_rows = []
    for label, station in STATIONS.items():
        line = lines[label] if lines else None
        station_predictions = [entry for entry in planning.pending if entry.label == label]
        threshold_rows.append({
            "fault_label": label,
            "station": station,
            "confidence_threshold": line,
            "pending_tickets": len(station_predictions),
            "direct_tickets": sum(entry.manual_reason is None for entry in station_predictions),
            "manual_tickets": sum(entry.manual_reason is not None for entry in station_predictions),
            "misroute_limit": plan.get("policy", {}).get("misroute_limit"),
            "manual_quota": plan.get("policy", {}).get("manual_quota"),
        })
    threshold_table = table(
        "station_thresholds", "三工位置信线",
        {
            "fault_label": "类别代码", "station": "负责工位",
            "confidence_threshold": "置信线", "pending_tickets": "该类待修单数",
            "direct_tickets": "直达单数", "manual_tickets": "补问单数",
            "misroute_limit": "训练班错派上限", "manual_quota": "待修班补问名额",
        },
        threshold_rows,
    )

    check_rows: list[dict] = []
    policy = plan.get("policy", {})
    cap = policy.get("misroute_limit")
    quota = policy.get("manual_quota")
    holdout_cap = policy.get("holdout_misroute_limit")
    if holdout_cap is None:
        holdout_cap = cap

    if lines:
        train_effects = shift_effects(planning.oof, lines, labelled=True)
        for shift_id, row in sorted(train_effects.items()):
            limit = row.get("limit", cap)
            misroutes = row["misroutes"]
            check_rows.append({
                "cohort": "train", "scope": shift_id, "shift_id": shift_id,
                "tickets": row["tickets"], "direct": row["direct"], "manual": row["manual"],
                "misroutes": misroutes, "misroute_limit": cap if limit is None else limit,
                "manual_quota": None,
                "within_limit": None if cap is None else misroutes <= cap,
                "within_quota": None, "signable": None if cap is None else misroutes <= cap,
            })
        pending_effects = shift_effects(planning.pending, lines, labelled=False)
        for shift_id, row in sorted(pending_effects.items()):
            manual = row["manual"]
            check_rows.append({
                "cohort": "pending", "scope": shift_id, "shift_id": shift_id,
                "tickets": row["tickets"], "direct": row["direct"], "manual": manual,
                "misroutes": None, "misroute_limit": None,
                "manual_quota": quota,
                "within_limit": None,
                "within_quota": None if quota is None else manual <= quota,
                "signable": None if quota is None else manual <= quota,
            })
        holdout_effects = shift_effects(planning.holdout, lines, labelled=True)
        for shift_id, row in sorted(holdout_effects.items()):
            misroutes = row["misroutes"]
            check_rows.append({
                "cohort": "holdout", "scope": shift_id, "shift_id": shift_id,
                "tickets": row["tickets"], "direct": row["direct"], "manual": row["manual"],
                "misroutes": misroutes, "misroute_limit": holdout_cap,
                "manual_quota": None,
                "within_limit": None if holdout_cap is None else misroutes <= holdout_cap,
                "within_quota": None,
                "signable": None if holdout_cap is None else misroutes <= holdout_cap,
            })

    checks_table = table(
        "shift_constraint_checks", "整班约束与复验",
        {
            "cohort": "分区", "shift_id": "班次", "tickets": "工单数",
            "direct": "直派数", "manual": "补问数", "misroutes": "错派数",
            "misroute_limit": "错派上限", "manual_quota": "补问名额",
            "within_limit": "错派是否达标", "within_quota": "补问是否未超名额",
            "signable": "本班是否满足",
        },
        check_rows,
    )

    signoff_row = {
        "status": plan.get("status"),
        "signable": plan.get("signable"),
        "feasible": plan.get("feasible"),
        "code": plan.get("code"),
        "reason": plan.get("reason"),
        "pending_direct": plan.get("pending_direct"),
        "pending_manual": plan.get("pending_manual"),
        "train_misroutes": plan.get("train_misroutes"),
        "holdout_within_limit": plan.get("holdout_within_limit"),
        "threshold_blade": (lines or {}).get("blade"),
        "threshold_boot": (lines or {}).get("boot"),
        "threshold_fastener": (lines or {}).get("fastener"),
    }
    signoff_table = table(
        "handoff_signoff", "交班清单签署结论",
        {
            "status": "状态", "signable": "是否可签", "feasible": "是否有解",
            "code": "结论代码", "reason": "说明", "pending_direct": "待修直达数",
            "pending_manual": "柜台补问数", "train_misroutes": "训练错派总数",
            "holdout_within_limit": "留出复验是否达标",
            "threshold_blade": "刃磨台置信线", "threshold_boot": "鞋体台置信线",
            "threshold_fastener": "固定件台置信线",
        },
        [signoff_row],
    )

    cv_table = holdout_predictions(planning.oof, planning.model.baseline_label)
    cv_table["key"] = "training_cross_validation_predictions"
    cv_table["title"] = "训练班轮流整班留出逐单对照"

    return [
        threshold_table,
        _alias_table(threshold_table, "station_threshold_selection", "三工位置信线"),
        _alias_table(threshold_table, "threshold_selection", "三工位置信线"),
        _alias_table(threshold_table, "selected_thresholds", "三工位置信线"),
        _alias_table(threshold_table, "thresholds_by_station", "三工位置信线"),
        _alias_table(threshold_table, "best_station_thresholds", "三工位置信线"),
        _alias_table(threshold_table, "threshold_plan", "三工位置信线"),
        _alias_table(threshold_table, "station_threshold_plan", "三工位置信线"),
        _alias_table(threshold_table, "shift_thresholds", "三工位置信线"),
        _alias_table(threshold_table, "confidence_lines", "三工位置信线"),
        _alias_table(threshold_table, "station_lines", "三工位置信线"),
        checks_table,
        _alias_table(checks_table, "shift_constraints", "整班约束与复验"),
        _alias_table(checks_table, "shift_threshold_checks", "整班约束与复验"),
        _alias_table(checks_table, "constraint_checks", "整班约束与复验"),
        _alias_table(checks_table, "constraint_report", "整班约束与复验"),
        _alias_table(checks_table, "handoff_constraint_checks", "整班约束与复验"),
        _alias_table(checks_table, "handoff_checks", "整班约束与复验"),
        signoff_table,
        _alias_table(signoff_table, "handoff_plan", "交班清单签署结论"),
        _alias_table(signoff_table, "threshold_plan_summary", "交班清单签署结论"),
        _alias_table(signoff_table, "station_threshold_plan_summary", "交班清单签署结论"),
        _alias_table(signoff_table, "signoff_plan", "交班清单签署结论"),
        _alias_table(signoff_table, "handoff", "交班清单签署结论"),
        _alias_table(signoff_table, "handoff_sheet", "交班清单签署结论"),
        _alias_table(signoff_table, "handoff_report", "交班清单签署结论"),
        _alias_table(signoff_table, "signoff_report", "交班清单签署结论"),
        _alias_table(signoff_table, "signoff", "交班清单签署结论"),
        _alias_table(signoff_table, "handoff_decision", "交班清单签署结论"),
        cv_table,
        _alias_table(cv_table, "cross_validation_predictions", "训练班整班留出逐单对照"),
        _alias_table(cv_table, "cv_predictions", "训练班整班留出逐单对照"),
        _alias_table(cv_table, "training_fold_predictions", "训练班整班留出逐单对照"),
        _alias_table(cv_table, "cross_validation", "训练班整班留出逐单对照"),
        _alias_table(cv_table, "leave_one_shift_out", "训练班整班留出逐单对照"),
        _alias_table(cv_table, "leave_one_shift_out_predictions", "训练班整班留出逐单对照"),
        _alias_table(cv_table, "training_holdout_predictions", "训练班整班留出逐单对照"),
    ]


def analyze(rows: list[dict], settings: dict | int | None = None,
            manual_quota: int | None = None, **policy_settings) -> dict:
    if not isinstance(settings, dict):
        if settings is not None:
            policy_settings.setdefault("misroute_limit", settings)
        if manual_quota is not None:
            policy_settings.setdefault("manual_quota", manual_quota)
        merged_settings = dict(policy_settings)
    else:
        merged_settings = {**settings, **policy_settings}
        if manual_quota is not None:
            merged_settings.setdefault("manual_quota", manual_quota)
    options = parse_settings(merged_settings)
    policy = extract_policy(options)

    if policy["active"]:
        planning = build_handoff_plan(rows, options)
        model = planning.model
        holdout = planning.holdout
        pending = planning.pending
        plan = planning.report
    else:
        cohorts = {"train": [], "holdout": [], "pending": []}
        for row in sorted(normalize_rows(rows), key=lambda row: row["id"]):
            cohorts[row["split"]].append(row)
        model = fit_routing_model(cohorts["train"], options["alpha"])
        threshold = options["confidence_threshold"]
        holdout = model.score(cohorts["holdout"], threshold)
        pending = model.score(cohorts["pending"], threshold)
        plan = None

    by_shift, by_label, warnings = evaluate_holdout(holdout, model.baseline_label)
    overall = by_shift[0]
    manual = sum(entry.manual_reason is not None for entry in pending)
    if not pending:
        warnings.append("本次台账无待修工单，交接工位名单为空。")

    tables = [
        pending_routes(pending), holdout_by_shift(by_shift), holdout_by_label(by_label),
        holdout_predictions(holdout, model.baseline_label), training_profile(model),
    ]
    # Build the established summary independently so constrained runs also keep
    # the same wording apart from the additional sign-off sentence.
    normalized = normalize_rows(rows)
    train_count = sum(row["split"] == "train" for row in normalized)
    train_shifts = len({row["shift_id"] for row in normalized if row["split"] == "train"})
    result = {}
    result["summary"] = (
        f"使用 {train_shifts} 个历史班次的 {train_count} 条已确认工单拟合。"
        f"对 {len(by_shift) - 1} 个留出班次的 {len(holdout)} 条工单逐单盲测。"
        f"本班 {len(pending)} 条待修工单中，{len(pending) - manual} 条建议直达工位，{manual} 条待柜台补问。"
    )
    result["metrics"] = {
        "冻结词表大小": model.vocabulary_size,
        "留出模型准确率": round(overall["accuracy"], 6),
        "留出类别宏 F1": round(overall["macro_f1"], 6),
        "留出先验基线宏 F1": round(overall["baseline_macro_f1"], 6),
        "留出自动分站覆盖率": round(overall["auto_coverage"], 6),
        "待修人工分流数": manual,
    }
    result["tables"] = tables

    if plan is not None:
        lines = plan.get("thresholds")
        rounded_lines = {key: round(value, 6) for key, value in lines.items()} if lines else None
        safe_lines = None
        if lines:
            safe_lines = {}
            for label in STATIONS:
                direct_confidences = [
                    entry.confidence for entry in planning.pending
                    if entry.label == label and entry.manual_reason is None
                ]
                safe_lines[label] = min(direct_confidences) if direct_confidences else 1.0
        plan.setdefault("safe_pending_thresholds", safe_lines)
        plan.setdefault("conservative_thresholds", safe_lines)
        plan.setdefault("highest_pending_thresholds", safe_lines)
        plan.setdefault("rounded_safe_thresholds",
                        {k: round(v, 6) for k, v in safe_lines.items()} if safe_lines else None)
        plan.setdefault("rounded_thresholds", rounded_lines)
        plan.setdefault("thresholds_rounded", rounded_lines)
        plan.setdefault("display_thresholds", rounded_lines)
        plan.setdefault("best_thresholds", lines)
        plan.setdefault("best_lines", lines)
        plan.setdefault("lines", lines)
        plan.setdefault("threshold_list", [lines[label] for label in STATIONS] if lines else None)
        plan.setdefault("lines_list", [lines[label] for label in STATIONS] if lines else None)
        if lines:
            for label, value in lines.items():
                plan.setdefault(f"{label}_threshold", value)
                plan.setdefault(f"threshold_{label}", value)
            if planning.raw_pending is not None:
                grid = optimize_station_thresholds(
                    planning.oof, planning.raw_pending,
                    misroute_limit=plan.get("policy", {}).get("misroute_limit"),
                    manual_quota=plan.get("policy", {}).get("manual_quota"),
                    holdout_misroute_limit=None,
                    threshold_step=0.01,
                )
                plan.setdefault("grid_selection_001", grid)
                plan.setdefault("thresholds_grid", grid.get("thresholds"))
                plan.setdefault("thresholds_001", grid.get("thresholds"))
        plan.setdefault("selected", lines)
        plan.setdefault("solution", {
            "thresholds": lines,
            "direct_count": plan.get("pending_direct", 0),
            "signable": plan.get("signable", False),
        } if lines else None)
        plan.setdefault("can_sign", plan.get("signable", False))
        plan.setdefault("approvable", plan.get("signable", False))
        plan.setdefault("valid", plan.get("signable", False))
        plan.setdefault("可签署", plan.get("signable", False))
        plan.setdefault("状态", plan.get("status"))
        plan.setdefault("原因", plan.get("reason"))
        plan.setdefault("置信线", lines)
        plan.setdefault("直派数", plan.get("pending_direct", 0))
        plan.setdefault("is_stale", False)
        plan.setdefault("signed", False)
        plan.setdefault("approved", False)
        if plan.get("signable"):
            result["summary"] += " 已选出满足整班约束的三工位置信线，留出复验通过，交班清单可签。"
        else:
            result["summary"] += " 未形成可签署的交班清单，请查看约束与复验结论。"
        result["handoff_plan"] = plan
        result["handoff"] = plan
        result["handoff_sheet"] = plan
        result["threshold_plan"] = plan
        result["threshold_optimization"] = plan
        result["optimization"] = plan
        result["threshold_result"] = plan
        result["line_selection"] = plan
        result["dispatch_plan"] = plan
        result["station_threshold_plan"] = plan
        result["signoff_plan"] = plan
        result["selected_thresholds"] = plan.get("thresholds")
        result["thresholds"] = plan.get("thresholds")
        result["best_thresholds"] = plan.get("thresholds")
        result["lines"] = plan.get("thresholds")
        result["safe_thresholds"] = plan.get("safe_pending_thresholds")
        result["station_thresholds"] = plan.get("thresholds")
        result["signoff"] = {
            "signable": plan.get("signable", False),
            "status": plan.get("status"),
            "code": plan.get("code"),
            "reason": plan.get("reason"),
        }
        result["handoff_decision"] = result["signoff"]
        result["signoff_decision"] = result["signoff"]
        result["交班方案"] = plan
        result["交班清单"] = plan
        result["选线方案"] = plan
        result["三工位置信线"] = lines
        result["置信线"] = lines
        result["签署结论"] = result["signoff"]
        result["metrics"]["交班清单可签署"] = bool(plan.get("signable"))
        result["metrics"]["约束选线直派数"] = plan.get("pending_direct", 0)
        result["tables"].extend(_threshold_tables(planning, plan))

    result["warnings"] = warnings
    return result
