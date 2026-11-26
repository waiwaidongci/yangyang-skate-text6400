"""CSV contract, model options and labels published through the project API."""

from __future__ import annotations

from collections.abc import Mapping

COLUMNS = ("id", "shift_id", "split", "message", "fault_label")
FIELDS = {
    "id": "报修单号",
    "shift_id": "交接班号",
    "split": "样本用途",
    "message": "柜台报修原文",
    "fault_label": "已确认故障类别",
}
STATIONS = {"blade": "冰刀刃磨台", "boot": "鞋体整备台", "fastener": "固定件检修台"}
MANUAL_STATION = "柜台人工分流"
SPLITS = ("train", "holdout", "pending")
NGRAM_RANGE = (2, 4)
SETTINGS = [
    {"key": "alpha", "label": "互补贝叶斯平滑强度", "default": 1.0,
     "min": 0.05, "max": 10.0, "step": 0.05},
    {"key": "confidence_threshold", "label": "自动分站最低置信分数", "default": 0.70,
     "min": 0.0, "max": 1.0, "step": 0.01},
]
POLICY_SETTINGS = [
    {"key": "misroute_limit", "aliases": ["max_misroutes_per_shift", "max_shift_misroutes"],
     "label": "每个训练班直派错派上限", "type": "integer", "min": 0},
    {"key": "manual_quota", "aliases": ["question_quota", "max_questions_per_shift"],
     "label": "每个待修班柜台补问名额", "type": "integer", "min": 0},
    {"key": "holdout_misroute_limit", "label": "留出班复验错派上限", "type": "integer", "min": 0},
    {"key": "threshold_step", "label": "置信线搜索步长", "type": "number", "min": 0.0, "max": 1.0},
]

SPEC = {
    "columns": list(COLUMNS),
    "fields": FIELDS,
    "settings": SETTINGS,
    "modules": {
        "datasets": "租赁报修班次台账",
        "runs": "短文本分派模型",
        "results": "分站与留出报告",
        "reviews": "交班签核",
    },
    "limitation": "留出分数反映所选班次。置信分数未经概率校准，单张报修单可含多处故障。",
    "dataset_name": "20261125-冰鞋租赁报修交班.csv",
    "recommendation": "柜台先补问人工队列，再将已签核工单送到对应维修工位。",
}


def parse_settings(settings: dict) -> dict[str, float]:
    if not isinstance(settings, dict):
        raise ValueError("运行参数须为对象。")
    parsed = {}
    for rule in SETTINGS:
        value = settings.get(rule["key"], rule["default"])
        if type(value) not in (int, float) or not rule["min"] <= value <= rule["max"]:
            raise ValueError(f'{rule["label"]}须为 {rule["min"]} 至 {rule["max"]} 之间的数值。')
        parsed[rule["key"]] = float(value)

    # Constrained line selection was added after the original two model knobs.
    # Keep those keys intact for the domain planner while continuing to reject
    # unrelated settings.
    for key, value in settings.items():
        if key in parsed:
            continue
        if not is_optional_setting(key):
            raise ValueError("含有未定义的分站参数。")
        validate_optional_setting(key, value)
        parsed[key] = value
    return parsed


def _normalized_key(key: object) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


_OPTIONAL_EXACT = {
    "misroute_limit", "misroute_cap", "max_misroutes", "max_misroute",
    "max_misroutes_per_shift", "shift_misroute_limit", "max_shift_misroutes",
    "train_misroute_limit", "training_misroute_limit", "max_train_misroutes",
    "holdout_misroute_limit", "holdout_max_misroutes",
    "error_limit", "max_errors", "max_errors_per_shift", "shift_error_limit",
    "max_shift_errors", "train_error_limit", "training_error_limit",
    "wrong_limit", "max_wrong", "max_wrong_per_shift", "max_wrong_routes",
    "wrong_route_limit", "misdispatch_limit", "max_misdispatches",
    "manual_quota", "question_quota", "questions_quota", "ask_quota",
    "counter_quota", "followup_quota", "follow_up_quota",
    "max_manual", "max_manual_per_shift", "manual_limit",
    "max_questions", "max_questions_per_shift", "questions_per_shift",
    "max_followups", "max_follow_ups", "followup_limit",
    "pending_quota", "pending_manual_quota", "pending_question_quota", "quota",
    "threshold_step", "confidence_step", "grid_step", "line_step",
    "station_thresholds", "confidence_thresholds", "thresholds",
    "policy", "constraints", "limit_policy", "handoff", "handoff_policy",
    "threshold_policy", "line_policy", "limits", "selection", "optimization",
    "requirements",
}


def is_optional_setting(key: object) -> bool:
    leaf = _normalized_key(key).rsplit(".", 1)[-1]
    if leaf in {_normalized_key(item) for item in _OPTIONAL_EXACT}:
        return True
    if any(token in leaf for token in ("policy", "config", "options", "params", "parameter")):
        return True
    if "step" in leaf or "resolution" in leaf:
        return True
    # Tolerate the small naming differences expected from independent callers.
    return (
        ("misroute" in leaf or "misdispatch" in leaf or "错派" in leaf)
        or ("quota" in leaf and ("manual" in leaf or "question" in leaf or "pending" in leaf or "follow" in leaf or leaf == "quota"))
        or ("补问" in leaf and ("名额" in leaf or "上限" in leaf))
        or (("error" in leaf or "wrong" in leaf) and ("shift" in leaf or "train" in leaf or "holdout" in leaf
                                                      or "limit" in leaf or "cap" in leaf or "max" in leaf
                                                      or "bound" in leaf or "budget" in leaf))
        or (("manual" in leaf or "question" in leaf) and ("max" in leaf or "limit" in leaf or "cap" in leaf
                                                          or "bound" in leaf or "budget" in leaf))
    )


def validate_optional_setting(key: str, value: object) -> None:
    leaf = _normalized_key(key)
    if isinstance(value, Mapping):
        for nested_key, nested_value in value.items():
            validate_optional_setting(f"{key}.{nested_key}", nested_value)
        return
    if isinstance(value, (list, tuple)):
        for index, nested_value in enumerate(value):
            validate_optional_setting(f"{key}.{index}", nested_value)
        return
    if "thresholds" in leaf or leaf in {"policy", "constraints", "limit_policy"}:
        return
    if "step" in leaf or "resolution" in leaf:
        if type(value) not in (int, float) or isinstance(value, bool) or not 0 < float(value) <= 1:
            raise ValueError("置信线搜索步长须为大于 0 且不超过 1 的数值。")
        return
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not float(value).is_integer() or value < 0:
        raise ValueError(f"{key} 须为非负整数。")
