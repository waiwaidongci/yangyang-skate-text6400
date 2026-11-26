"""CSV contract, model options and labels published through the project API."""

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
    if settings.keys() - parsed.keys():
        raise ValueError("含有未定义的分站参数。")
    return parsed
