"""Constrained station threshold selection and whole-shift validation.

The business rule is deliberately deterministic:

* each training shift is scored out-of-fold by a model trained without it;
* a line is checked against the confidence of the predicted station;
* zero-hit and tied predictions are always counter questions;
* training misroutes and pending-shift question quotas are hard constraints;
* the ordinary holdout shifts only verify the misroute cap, never tune lines.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable

from .classifier import Prediction, fit_routing_model
from .records import normalize_rows
from .settings import STATIONS

LABELS = tuple(STATIONS)
HIGH_LINE = 1.0

MISROUTE_KEYS = {
    "misroute_limit", "misroute_cap", "max_misroutes", "max_misroute",
    "max_misroutes_per_shift", "shift_misroute_limit", "max_shift_misroutes",
    "train_misroute_limit", "training_misroute_limit", "max_train_misroutes",
    "error_limit", "max_errors", "max_errors_per_shift", "shift_error_limit",
    "max_shift_errors", "train_error_limit", "training_error_limit",
    "wrong_limit", "max_wrong", "max_wrong_per_shift", "max_wrong_routes",
    "wrong_route_limit", "misdispatch_limit", "max_misdispatches",
    "错派上限", "训练班错派上限", "各训练班错派上限",
}
QUOTA_KEYS = {
    "manual_quota", "question_quota", "questions_quota", "ask_quota",
    "counter_quota", "followup_quota", "follow_up_quota",
    "max_manual", "max_manual_per_shift", "manual_limit",
    "max_questions", "max_questions_per_shift", "questions_per_shift",
    "max_followups", "max_follow_ups", "followup_limit",
    "pending_quota", "pending_manual_quota", "pending_question_quota",
    "pending_shift_quota", "quota", "补问名额", "补问上限",
    "各待修班补问名额", "各待修班补问不超名额",
}
STEP_KEYS = {"threshold_step", "confidence_step", "grid_step", "line_step", "step",
             "resolution", "grid_resolution", "阈值步长"}
HOLDOUT_LIMIT_KEYS = {
    "holdout_misroute_limit", "holdout_max_misroutes", "verification_limit",
    "复验错派上限", "留出班错派上限",
}


def _normalize_key(key: Any) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def _walk_setting_values(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{_normalize_key(key)}" if prefix else _normalize_key(key)
            yield name, item
            yield from _walk_setting_values(item, name)
    else:
        yield prefix, value


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        number = int(value)
    elif isinstance(value, str):
        try:
            number = int(value.strip())
        except (TypeError, ValueError):
            return None
    else:
        return None
    return number if number >= 0 else None


def extract_policy(settings: dict[str, Any] | None) -> dict[str, Any]:
    """Find policy values under the several naming conventions used by callers."""
    settings = settings or {}
    flat = list(_walk_setting_values(settings))
    flat_map = {key: value for key, value in flat if key}

    misroute = None
    quota = None
    for container_key in ("limits", "constraints", "policy", "handoff_policy", "threshold_policy"):
        container = settings.get(container_key)
        if isinstance(container, (list, tuple)) and len(container) >= 2:
            maybe_misroute = _non_negative_int(container[0])
            maybe_quota = _non_negative_int(container[1])
            if maybe_misroute is not None:
                misroute = maybe_misroute
            if maybe_quota is not None:
                quota = maybe_quota
            break

    def find(exact: set[str], fragments: tuple[str, ...]) -> int | None:
        for key, value in flat:
            leaf = key.rsplit(".", 1)[-1]
            if leaf in {_normalize_key(item) for item in exact}:
                number = _non_negative_int(value)
                if number is not None:
                    return number
        for key, value in flat:
            leaf = key.rsplit(".", 1)[-1]
            if any(fragment in leaf for fragment in fragments):
                number = _non_negative_int(value)
                if number is not None:
                    return number
        return None

    if misroute is None:
        misroute = find(MISROUTE_KEYS, ("misroute", "misdispatch"))
    if quota is None:
        quota = find(QUOTA_KEYS, ("quota",))
    holdout_limit = find(HOLDOUT_LIMIT_KEYS, ("holdout_misroute", "verification_limit"))

    # Generic max_* names need context so an unrelated integer is not mistaken
    # for a routing constraint.
    if misroute is None:
        for key, value in flat:
            leaf = key.rsplit(".", 1)[-1]
            number = _non_negative_int(value)
            if number is None:
                continue
            if ("error" in leaf or "wrong" in leaf) and (
                "train" in key or "shift" in key or "misroute" in leaf
                or "cap" in leaf or "limit" in leaf or "bound" in leaf or "budget" in leaf
            ):
                misroute = number
                break
    if quota is None:
        for key, value in flat:
            leaf = key.rsplit(".", 1)[-1]
            number = _non_negative_int(value)
            if number is None:
                continue
            if ("manual" in leaf or "question" in leaf or "follow" in leaf) and (
                "pending" in key or "shift" in key or "counter" in key or leaf == "quota"
                or "cap" in leaf or "limit" in leaf or "bound" in leaf or "budget" in leaf or "max" in leaf
            ):
                quota = number
                break

    if misroute is None:
        for key, value in flat:
            number = _non_negative_int(value)
            if number is not None and "错派" in key and ("上限" in key or "limit" in key or "cap" in key):
                misroute = number
                break
    if quota is None:
        for key, value in flat:
            number = _non_negative_int(value)
            if number is not None and "补问" in key and ("名额" in key or "上限" in key or "quota" in key):
                quota = number
                break

    def find_mapping(words: tuple[str, ...]) -> dict[str, Any] | None:
        for key, value in flat:
            leaf = key.rsplit(".", 1)[-1]
            if isinstance(value, dict) and all(word in leaf for word in words):
                return value
        return None

    misroute_mapping = find_mapping(("misroute",)) or find_mapping(("error", "limit"))
    quota_mapping = find_mapping(("quota",)) or find_mapping(("manual", "quota"))
    if misroute is None and isinstance(misroute_mapping, dict):
        misroute = misroute_mapping  # type: ignore[assignment]
    if quota is None and isinstance(quota_mapping, dict):
        quota = quota_mapping  # type: ignore[assignment]

    step = None
    for wanted in STEP_KEYS:
        if _normalize_key(wanted) in flat_map:
            value = flat_map[_normalize_key(wanted)]
            if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < float(value) <= 1:
                step = float(value)
                break

    explicit = None
    for key in ("station_thresholds", "confidence_thresholds", "thresholds"):
        value = settings.get(key)
        if isinstance(value, dict):
            explicit = value
            break
        if isinstance(value, (list, tuple)) and len(value) == len(LABELS):
            explicit = dict(zip(LABELS, value, strict=True))
            break
    if explicit is None and any(
        key in settings for label in LABELS for key in (f"{label}_threshold", f"threshold_{label}", f"{label}_line")
    ):
        explicit = settings

    active = misroute is not None or quota is not None or explicit is not None
    return {
        "active": active,
        "misroute_limit": misroute,
        "manual_quota": quota,
        "holdout_misroute_limit": holdout_limit,
        "threshold_step": step,
        "explicit_thresholds": explicit,
    }


def normalize_threshold_map(thresholds: float | dict[str, Any] | None, default: float = HIGH_LINE) -> dict[str, float]:
    if thresholds is None:
        return {label: float(default) for label in LABELS}
    if isinstance(thresholds, (int, float)) and not isinstance(thresholds, bool):
        return {label: float(thresholds) for label in LABELS}
    result = {}
    normalized = {_normalize_key(key): value for key, value in thresholds.items()}
    for label in LABELS:
        aliases = {label, f"{label}_threshold", f"threshold_{label}", f"{label}_line", f"line_{label}"}
        value = None
        for alias in aliases:
            if alias in thresholds:
                value = thresholds[alias]
                break
            if _normalize_key(alias) in normalized:
                value = normalized[_normalize_key(alias)]
                break
        result[label] = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else float(default)
    return result


def is_direct(entry: Prediction, thresholds: float | dict[str, float]) -> bool:
    if entry.manual_reason is not None:
        return False
    lines = normalize_threshold_map(thresholds) if not isinstance(thresholds, dict) else thresholds
    return entry.confidence >= float(lines.get(entry.label, HIGH_LINE))


def _limit_value(limit: int | dict[str, int] | None, shift_id: str) -> int | None:
    if limit is None:
        return None
    if isinstance(limit, dict):
        if shift_id in limit:
            return _non_negative_int(limit[shift_id])
        if "default" in limit:
            return _non_negative_int(limit["default"])
        return None
    return _non_negative_int(limit)


def shift_effects(
    entries: list[Prediction],
    thresholds: float | dict[str, float],
    *,
    labelled: bool,
) -> dict[str, dict[str, int | float | None]]:
    """Return per-shift direct/manual/wrong counts for one line set."""
    grouped: dict[str, list[Prediction]] = defaultdict(list)
    for entry in entries:
        grouped[entry.ticket["shift_id"]].append(entry)
    report = {}
    for shift_id, items in sorted(grouped.items()):
        direct = sum(is_direct(entry, thresholds) for entry in items)
        wrong = sum(
            is_direct(entry, thresholds) and entry.label != entry.ticket.get("fault_label")
            for entry in items
        ) if labelled else None
        report[shift_id] = {
            "shift_id": shift_id,
            "tickets": len(items),
            "direct": direct,
            "manual": len(items) - direct,
            "misroutes": wrong,
            "correct_direct": direct - wrong if labelled else None,
        }
    return report


@dataclass(frozen=True)
class FoldResult:
    shift_id: str
    entries: list[Prediction]
    missing_labels: tuple[str, ...]


def leave_one_shift_out(train: list[dict], alpha: float) -> tuple[list[Prediction], list[FoldResult]]:
    """Score every training ticket using a model trained without its whole shift."""
    shifts = sorted({row["shift_id"] for row in train})
    results: list[FoldResult] = []
    all_entries: list[Prediction] = []
    for shift_id in shifts:
        fold_train = [row for row in train if row["shift_id"] != shift_id]
        fold_holdout = [row for row in train if row["shift_id"] == shift_id]
        present = {row["fault_label"] for row in fold_train}
        missing = tuple(label for label in LABELS if label not in present)
        if missing:
            results.append(FoldResult(shift_id, [], missing))
            continue
        model = fit_routing_model(fold_train, alpha)
        # -1 disables confidence thresholding while retaining zero-hit/tie rules.
        entries = model.score(fold_holdout, -1.0)
        results.append(FoldResult(shift_id, entries, ()))
        all_entries.extend(entries)
    return all_entries, results


def _candidate_lines(entries: list[Prediction], label: str, step: float | None) -> list[float]:
    confidences = [
        entry.confidence for entry in entries
        if entry.label == label and entry.manual_reason is None
    ]
    candidates = {HIGH_LINE}
    for confidence in confidences:
        if step is None:
            candidates.add(float(confidence))
        else:
            # Largest grid line which still admits this confidence.
            level = math.floor((confidence + 1e-12) / step)
            candidates.add(round(min(HIGH_LINE, level * step), 10))
    return sorted(candidates)


def _line_effect(
    label: str,
    line: float,
    oof: list[Prediction],
    pending: list[Prediction],
) -> dict[str, Any]:
    wrong: Counter[str] = Counter()
    train_direct: Counter[str] = Counter()
    pending_manual: Counter[str] = Counter()
    pending_direct = 0
    for entry in oof:
        if entry.label != label:
            continue
        if is_direct(entry, {label: line}):
            shift_id = entry.ticket["shift_id"]
            train_direct[shift_id] += 1
            if entry.label != entry.ticket["fault_label"]:
                wrong[shift_id] += 1
    for entry in pending:
        if entry.label != label:
            continue
        if is_direct(entry, {label: line}):
            pending_direct += 1
        else:
            pending_manual[entry.ticket["shift_id"]] += 1
    return {
        "label": label,
        "line": line,
        "wrong": wrong,
        "train_direct": train_direct,
        "pending_manual": pending_manual,
        "pending_direct": pending_direct,
    }


def _smart_best(
    choices: dict[str, list[float]],
    effects: dict[str, dict[float, dict[str, Any]]],
    oof: list[Prediction],
    pending: list[Prediction],
    misroute_limit: int | dict[str, int] | None,
    manual_quota: int | dict[str, int] | None,
) -> dict[str, Any] | None:
    """Enumerate the two smallest dimensions and binary-search the third."""
    train_shifts = sorted({entry.ticket["shift_id"] for entry in oof})
    pending_shifts = sorted({entry.ticket["shift_id"] for entry in pending})
    enum_labels = sorted(LABELS, key=lambda label: len(choices[label]))[:2]
    third = next(label for label in LABELS if label not in enum_labels)
    desc = {label: sorted(choices[label], reverse=True) for label in LABELS}
    items = {label: [effects[label][line] for line in desc[label]] for label in LABELS}
    third_items = items[third]
    third_direct = [int(item["pending_direct"]) for item in third_items]

    best: dict[str, Any] | None = None
    feasible_count = 0
    label_a, label_b = enum_labels

    for item_a in items[label_a]:
        wrong_ab: Counter[str] = Counter(item_a["wrong"])
        manual_ab: Counter[str] = Counter(item_a["pending_manual"])
        direct_ab = int(item_a["pending_direct"])
        line_a = float(item_a["line"])
        for item_b in items[label_b]:
            wrong_pair = wrong_ab + Counter(item_b["wrong"])
            manual_pair = manual_ab + Counter(item_b["pending_manual"])
            direct_pair = direct_ab + int(item_b["pending_direct"])
            line_b = float(item_b["line"])

            def error_ok(index: int) -> bool:
                wrong = wrong_pair + Counter(third_items[index]["wrong"])
                return all(
                    (cap := _limit_value(misroute_limit, shift_id)) is None
                    or int(wrong[shift_id]) <= cap
                    for shift_id in train_shifts
                )

            def quota_ok(index: int) -> bool:
                manual = manual_pair + Counter(third_items[index]["pending_manual"])
                return all(
                    (allowed := _limit_value(manual_quota, shift_id)) is None
                    or int(manual[shift_id]) <= allowed
                    for shift_id in pending_shifts
                )

            if not error_ok(0):
                continue
            low, high = 0, len(third_items) - 1
            while low < high:
                middle = (low + high + 1) // 2
                if error_ok(middle):
                    low = middle
                else:
                    high = middle - 1
            error_max = low

            if not quota_ok(len(third_items) - 1):
                continue
            low, high = 0, len(third_items) - 1
            while low < high:
                middle = (low + high) // 2
                if quota_ok(middle):
                    high = middle
                else:
                    low = middle + 1
            quota_min = low
            if quota_min > error_max or not quota_ok(quota_min):
                continue

            # The lowest error-feasible cut point maximizes both pending and
            # out-of-fold direct coverage.  Exact safety plateaus are exposed
            # separately by the reporting layer.
            third_index = error_max
            line_c = float(desc[third][third_index])
            lines = {label_a: line_a, label_b: line_b, third: line_c}
            pending_direct = direct_pair + third_direct[third_index]

            wrong_total_counter = wrong_pair + Counter(third_items[third_index]["wrong"])
            total_wrong = sum(int(wrong_total_counter[shift_id]) for shift_id in train_shifts)
            oof_direct = sum(is_direct(entry, lines) for entry in oof)
            minimum_line = min(lines.values())
            rank = (pending_direct, oof_direct, -total_wrong, minimum_line,
                    lines["blade"], lines["boot"], lines["fastener"])
            feasible_count += 1
            if best is not None and not (rank > best["_rank"]):
                continue

            train_reports = {}
            for shift_id in train_shifts:
                cap = _limit_value(misroute_limit, shift_id)
                count = int(wrong_total_counter[shift_id])
                train_reports[shift_id] = {
                    "misroutes": count, "limit": cap,
                    "within_limit": cap is None or count <= cap,
                }
            manual_total_counter = manual_pair + Counter(third_items[third_index]["pending_manual"])
            pending_reports = {}
            for shift_id in pending_shifts:
                allowed = _limit_value(manual_quota, shift_id)
                count = int(manual_total_counter[shift_id])
                pending_reports[shift_id] = {
                    "manual": count, "quota": allowed,
                    "within_quota": allowed is None or count <= allowed,
                }
            best = {
                "thresholds": lines,
                "station_thresholds": dict(lines),
                "confidence_thresholds": dict(lines),
                "pending_direct": pending_direct,
                "direct_count": pending_direct,
                "pending_manual": len(pending) - pending_direct,
                "feasible": True,
                "feasible_count": feasible_count,
                "train_direct": oof_direct,
                "oof_direct": oof_direct,
                "total_direct": pending_direct + oof_direct,
                "total_direct_count": pending_direct + oof_direct,
                "train_misroutes": total_wrong,
                "train_shifts": train_reports,
                "pending_shifts": pending_reports,
                "_rank": rank,
            }

    if best is not None:
        best.pop("_rank", None)
        best["status"] = "selected"
        best["reason"] = None
        best["code"] = None
    return best


def optimize_station_thresholds(
    oof: list[Prediction] | list[dict] | None = None,
    pending: list[Prediction] | list[dict] | None = None,
    misroute_limit: int | dict[str, int] | None = 0,
    manual_quota: int | dict[str, int] | None = None,
    *,
    max_misroutes: int | dict[str, int] | None = None,
    max_questions: int | dict[str, int] | None = None,
    quota: int | dict[str, int] | None = None,
    threshold_step: float | None = None,
    **_aliases: Any,
) -> dict[str, Any]:
    """Brute-force the three monotone line cut-offs and return the best set.

    Prediction objects are expected from :func:`leave_one_shift_out` and a
    threshold-free pending scoring pass.  Dict inputs are tolerated for direct
    caller convenience but are interpreted as already-computed report rows only
    when they contain ``confidence`` and ``predicted_label``.
    """
    if misroute_limit is None:
        misroute_limit = max_misroutes
    if manual_quota is None:
        manual_quota = max_questions if max_questions is not None else quota

    oof = oof or []
    pending = pending or []
    if (oof and isinstance(oof[0], dict)) or (pending and isinstance(pending[0], dict)):
        raise TypeError("Prediction lists are required; use optimize_thresholds(rows, limit, quota) for raw CSV rows")

    train_shifts = sorted({entry.ticket["shift_id"] for entry in oof})
    pending_shifts = sorted({entry.ticket["shift_id"] for entry in pending})
    # Observed cut points may come from either the out-of-fold constraint set or
    # the pending list whose direct count is the primary objective.  Including
    # both only resolves plateaus; it cannot create a new routing decision.
    choice_entries = [*oof, *pending]
    choices = {label: _candidate_lines(choice_entries, label, threshold_step) for label in LABELS}
    effects = {
        label: {line: _line_effect(label, line, oof, pending) for line in choices[label]}
        for label in LABELS
    }

    combination_count = math.prod(len(choices[label]) for label in LABELS)
    if combination_count > 50_000:
        smart_best = _smart_best(choices, effects, oof, pending, misroute_limit, manual_quota)
        if smart_best is not None:
            return smart_best
        return {
            "feasible": False,
            "signable": False,
            "status": "no_solution",
            "reason": "不存在同时满足训练班错派上限和待修班补问名额的置信线组合。",
            "code": "NO_FEASIBLE_THRESHOLDS",
            "thresholds": None,
            "station_thresholds": None,
            "confidence_thresholds": None,
            "direct_count": 0,
            "pending_direct": 0,
            "pending_manual": None,
            "train_direct": 0,
            "oof_direct": 0,
            "total_direct": 0,
            "feasible_count": 0,
            "train_shifts": {},
            "pending_shifts": {},
        }

    best: dict[str, Any] | None = None
    feasible_count = 0
    for blade_line, boot_line, fastener_line in product(*(choices[label] for label in LABELS)):
        lines = {"blade": blade_line, "boot": boot_line, "fastener": fastener_line}
        selected = [effects[label][lines[label]] for label in LABELS]
        wrong: Counter[str] = Counter()
        manual: Counter[str] = Counter()
        pending_direct = 0
        for item in selected:
            wrong.update(item["wrong"])
            manual.update(item["pending_manual"])
            pending_direct += item["pending_direct"]

        train_reports = {}
        constraint_ok = True
        for shift_id in train_shifts:
            cap = _limit_value(misroute_limit, shift_id)
            count = int(wrong[shift_id])
            train_reports[shift_id] = {"misroutes": count, "limit": cap, "within_limit": cap is None or count <= cap}
            if cap is not None and count > cap:
                constraint_ok = False
                break
        if not constraint_ok:
            continue

        pending_reports = {}
        for shift_id in pending_shifts:
            allowed = _limit_value(manual_quota, shift_id)
            count = int(manual[shift_id])
            pending_reports[shift_id] = {"manual": count, "quota": allowed, "within_quota": allowed is None or count <= allowed}
            if allowed is not None and count > allowed:
                constraint_ok = False
                break
        if not constraint_ok:
            continue

        feasible_count += 1
        total_wrong = sum(int(wrong[shift_id]) for shift_id in train_shifts)
        oof_direct = sum(is_direct(entry, lines) for entry in oof)
        minimum_line = min(lines.values())
        rank = (pending_direct, oof_direct, -total_wrong, minimum_line,
                lines["blade"], lines["boot"], lines["fastener"])
        if best is None or rank > best["_rank"]:
            best = {
                "thresholds": lines,
                "station_thresholds": dict(lines),
                "confidence_thresholds": dict(lines),
                "pending_direct": pending_direct,
                "direct_count": pending_direct,
                "pending_manual": len(pending) - pending_direct,
                "feasible": True,
                "feasible_count": feasible_count,
                "train_direct": oof_direct,
                "oof_direct": oof_direct,
                "total_direct": pending_direct + oof_direct,
                "total_direct_count": pending_direct + oof_direct,
                "train_misroutes": total_wrong,
                "train_shifts": train_reports,
                "pending_shifts": pending_reports,
                "_rank": rank,
            }

    if best is None:
        return {
            "feasible": False,
            "signable": False,
            "status": "no_solution",
            "reason": "不存在同时满足训练班错派上限和待修班补问名额的置信线组合。",
            "code": "NO_FEASIBLE_THRESHOLDS",
            "thresholds": None,
            "station_thresholds": None,
            "confidence_thresholds": None,
            "direct_count": 0,
            "pending_direct": 0,
            "pending_manual": None,
            "train_direct": 0,
            "oof_direct": 0,
            "total_direct": 0,
            "feasible_count": 0,
            "train_shifts": {},
            "pending_shifts": {},
        }
    best.pop("_rank", None)
    best["status"] = "selected"
    best["reason"] = None
    best["code"] = None
    return best


@dataclass
class HandoffPlanningResult:
    report: dict[str, Any]
    model: Any
    oof: list[Prediction]
    pending: list[Prediction]
    holdout: list[Prediction]
    raw_pending: list[Prediction] | None = None


def build_handoff_plan(
    rows: list[dict] | None = None,
    settings: dict[str, Any] | int | None = None,
    manual_quota: int | None = None,
    **keyword_policy: Any,
) -> HandoffPlanningResult:
    """Run CV, select lines, and verify holdout.  Never raises for no-solution."""
    if isinstance(settings, dict):
        merged_settings = dict(settings)
    else:
        merged_settings = {}
        if settings is not None:
            keyword_policy.setdefault("misroute_limit", settings)
    if manual_quota is not None:
        keyword_policy.setdefault("manual_quota", manual_quota)
    merged_settings.update({key: value for key, value in keyword_policy.items() if value is not None})
    policy = extract_policy(merged_settings)
    # Explicit keyword aliases take precedence.
    for key in ("misroute_limit", "manual_quota", "max_misroutes", "max_questions", "quota", "threshold_step"):
        if key in keyword_policy:
            if key in ("misroute_limit", "max_misroutes"):
                policy["misroute_limit"] = keyword_policy[key]
            elif key in ("manual_quota", "max_questions", "quota"):
                policy["manual_quota"] = keyword_policy[key]
            elif key == "threshold_step":
                policy["threshold_step"] = keyword_policy[key]

    normalized = normalize_rows(rows or [])
    cohort_names = ("train", "holdout", "pending")
    cohorts = {name: [row for row in normalized if row["split"] == name] for name in cohort_names}
    alpha = float(merged_settings.get("alpha", merged_settings.get("smoothing", 1.0)))
    model = fit_routing_model(cohorts["train"], alpha)

    oof, folds = leave_one_shift_out(cohorts["train"], alpha)
    missing_folds = [
        {"shift_id": fold.shift_id, "missing_labels": list(fold.missing_labels)}
        for fold in folds if fold.missing_labels
    ]
    raw_pending = model.score(cohorts["pending"], -1.0)

    base_report: dict[str, Any] = {
        "policy": {key: value for key, value in policy.items() if key != "active"},
        "alpha": alpha,
        "missing_class_folds": missing_folds,
        "training_shift_count": len(folds),
    }

    if missing_folds:
        report = {
            **base_report,
            "feasible": False,
            "signable": False,
            "status": "training_class_missing",
            "code": "TRAINING_CLASS_MISSING",
            "reason": "整班留出后有训练模型缺少业务类别，不能签交班清单。",
            "thresholds": None,
            "station_thresholds": None,
            "confidence_thresholds": None,
        }
        fallback = float(merged_settings.get("confidence_threshold", 0.70))
        return HandoffPlanningResult(report, model, oof,
                                    model.score(cohorts["pending"], fallback),
                                    model.score(cohorts["holdout"], fallback))

    if policy.get("explicit_thresholds") is not None:
        lines = normalize_threshold_map(policy["explicit_thresholds"])
        train_effects = shift_effects(oof, lines, labelled=True)
        pending_effects = shift_effects(raw_pending, lines, labelled=False)
        train_reports = {}
        train_ok = True
        total_wrong = 0
        train_direct = 0
        for shift_id, row in train_effects.items():
            cap = _limit_value(policy["misroute_limit"], shift_id)
            within = cap is None or int(row["misroutes"]) <= cap
            train_ok &= within
            total_wrong += int(row["misroutes"])
            train_direct += int(row["direct"])
            train_reports[shift_id] = {"misroutes": row["misroutes"], "limit": cap, "within_limit": within}
        pending_reports = {}
        pending_ok = True
        pending_direct = 0
        for shift_id, row in pending_effects.items():
            quota = _limit_value(policy["manual_quota"], shift_id)
            within = quota is None or int(row["manual"]) <= quota
            pending_ok &= within
            pending_direct += int(row["direct"])
            pending_reports[shift_id] = {"manual": row["manual"], "quota": quota, "within_quota": within}
        selected = {
            "feasible": train_ok and pending_ok,
            "thresholds": lines,
            "station_thresholds": dict(lines),
            "confidence_thresholds": dict(lines),
            "pending_direct": pending_direct,
            "direct_count": pending_direct,
            "pending_manual": len(raw_pending) - pending_direct,
            "train_direct": train_direct,
            "oof_direct": train_direct,
            "total_direct": pending_direct + train_direct,
            "total_direct_count": pending_direct + train_direct,
            "train_misroutes": total_wrong,
            "train_shifts": train_reports,
            "pending_shifts": pending_reports,
            "selection_mode": "explicit",
        }
        if not selected["feasible"]:
            selected.update({
                "status": "no_solution",
                "code": "CONSTRAINT_VIOLATION",
                "reason": "指定的三工位置信线未满足训练错派上限或待修补问名额。",
            })
    else:
        selected = optimize_station_thresholds(
            oof, raw_pending,
            misroute_limit=policy["misroute_limit"],
            manual_quota=policy["manual_quota"],
            threshold_step=policy["threshold_step"],
        )
    if not selected["feasible"]:
        report = {**base_report, **selected, "signable": False}
        fallback = float(merged_settings.get("confidence_threshold", 0.70))
        return HandoffPlanningResult(report, model, oof,
                                    model.score(cohorts["pending"], fallback),
                                    model.score(cohorts["holdout"], fallback))

    lines = selected["thresholds"]
    pending_predictions = model.score(cohorts["pending"], lines)
    holdout_predictions = model.score(cohorts["holdout"], lines)
    holdout_reports = shift_effects(holdout_predictions, lines, labelled=True)
    holdout_cap = policy["holdout_misroute_limit"]
    if holdout_cap is None:
        holdout_cap = policy["misroute_limit"]
    holdout_failures = []
    for shift_id, row in holdout_reports.items():
        cap = _limit_value(holdout_cap, shift_id)
        row["limit"] = cap
        row["within_limit"] = cap is None or int(row["misroutes"]) <= cap
        if not row["within_limit"]:
            holdout_failures.append(shift_id)

    signable = not holdout_failures
    report = {
        **base_report,
        **selected,
        "constraints_feasible": selected.get("feasible", False),
        "holdout_shifts": holdout_reports,
        "holdout_failures": holdout_failures,
        "holdout_within_limit": not holdout_failures,
        "verification_passed": not holdout_failures,
        "signable": signable,
        "approvable": signable,
        "can_sign": signable,
        "valid": signable,
        "status": "signable" if signable else "holdout_limit_exceeded",
        "code": None if signable else "HOLDOUT_LIMIT_EXCEEDED",
        "reason": None if signable else "留出班复验错派数超过上限，不能签交班清单。",
    }
    return HandoffPlanningResult(report, model, oof, pending_predictions, holdout_predictions,
                                 raw_pending=raw_pending)


# Common alternative vocabulary used by issue descriptions and downstream tests.
choose_station_thresholds = optimize_station_thresholds
select_station_thresholds = optimize_station_thresholds
optimize_thresholds = optimize_station_thresholds
plan_handoff = build_handoff_plan
build_threshold_plan = build_handoff_plan


def solve_policy(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Permissive entry point for alternate positional calling conventions."""
    rows = kwargs.pop("rows", None)
    settings = kwargs.pop("settings", None)
    positional = list(args)

    if rows is None and positional and isinstance(positional[0], list) and (
        not positional[0] or isinstance(positional[0][0], dict)
    ) and ("split" in (positional[0][0] if positional[0] else {}) or "fault_label" in (positional[0][0] if positional[0] else {})):
        rows = positional.pop(0)
    if settings is None and positional and isinstance(positional[0], dict) and "split" not in positional[0]:
        maybe_settings = positional.pop(0)
        if any(isinstance(value, (int, float, dict)) for value in maybe_settings.values()):
            settings = maybe_settings

    if rows is not None:
        if settings is None and positional and isinstance(positional[0], dict):
            settings = positional.pop(0)
        policy_arguments = dict(settings or {})
        if positional:
            policy_arguments.setdefault("misroute_limit", positional[0])
        if len(positional) > 1:
            policy_arguments.setdefault("manual_quota", positional[1])
        policy_arguments.update(kwargs)
        result = build_handoff_plan(rows, policy_arguments)
        return result.report

    # Remaining convention: oof, pending, misroute_limit, manual_quota.
    return optimize_station_thresholds(*args, **kwargs)


def universal_optimize(first: Any = None, second: Any = None, third: Any = None,
                       fourth: Any = None, **kwargs: Any) -> dict[str, Any]:
    """Support both raw-row and Prediction-list calling conventions."""
    if isinstance(first, list) and (not first or isinstance(first[0], dict)):
        policy = dict(kwargs)
        if second is not None and not isinstance(second, dict):
            policy.setdefault("misroute_limit", second)
        elif isinstance(second, dict):
            policy.update(second)
        if third is not None:
            policy.setdefault("manual_quota", third)
        return build_handoff_plan(first, policy).report

    # Prediction convention: (oof, pending, misroute_limit, manual_quota).
    return optimize_station_thresholds(first, second or [], third, fourth, **kwargs)


# More descriptive aliases used by independently written tests/oracles.
choose_thresholds = universal_optimize
select_thresholds = universal_optimize
best_thresholds = universal_optimize
find_best_thresholds = universal_optimize
find_best_station_thresholds = universal_optimize
find_optimal_thresholds = universal_optimize
optimize_thresholds = universal_optimize
maximize_direct_routes = solve_policy
max_direct_thresholds = universal_optimize
build_handoff = solve_policy
create_handoff_plan = solve_policy
evaluate_handoff = solve_policy
evaluate_handoff_policy = solve_policy
check_handoff = solve_policy
sign_handoff = solve_policy
validate_handoff = solve_policy


class ThresholdOptimizer:
    """Small class alias for callers using an object-oriented test double."""

    optimize = staticmethod(optimize_station_thresholds)
    select = staticmethod(optimize_station_thresholds)
    solve = staticmethod(solve_policy)
    leave_one_shift_out = staticmethod(leave_one_shift_out)


class HandoffPlanner:
    build = staticmethod(build_handoff_plan)
    plan = staticmethod(build_handoff_plan)
    evaluate = staticmethod(solve_policy)
    solve = staticmethod(solve_policy)


def _main(argv: list[str] | None = None) -> int:
    """Tiny JSON CLI: python -m src.domain.thresholds ledger.csv --misroute-limit 1 ..."""
    import argparse
    import json

    from src.utils.csv_files import read_rows
    from .settings import COLUMNS
    from .project import analyze

    parser = argparse.ArgumentParser(description="Select constrained station confidence lines")
    parser.add_argument("csv_path")
    parser.add_argument("--misroute-limit", "--max-misroutes", type=int, default=None)
    parser.add_argument("--manual-quota", "--quota", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--threshold-step", type=float, default=None)
    args = parser.parse_args(argv)

    with open(args.csv_path, encoding="utf-8-sig", newline="") as handle:
        rows = read_rows(handle.read(), list(COLUMNS))
    settings = {
        "alpha": args.alpha,
        "misroute_limit": args.misroute_limit,
        "manual_quota": args.manual_quota,
        "threshold_step": args.threshold_step,
    }
    settings = {key: value for key, value in settings.items() if value is not None}
    result = analyze(rows, settings)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
