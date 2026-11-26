"""Validate a CSV cohort before it enters the model or the job queue."""

import re
import unicodedata
from collections import Counter

from .settings import COLUMNS, FIELDS, SPLITS, STATIONS

CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def parse_code(value, key: str, line: int) -> str:
    if isinstance(value, str) and CODE.fullmatch(value.strip()):
        return value.strip().upper()
    raise ValueError(f"第 {line} 条的{FIELDS[key]}须为 1 至 64 位编号，以英文字母或数字开头。")


def parse_message(value, line: int) -> str:
    if isinstance(value, str):
        original = value.strip()
        cleaned = unicodedata.normalize("NFKC", original)
        if len(original) <= 240 and sum(char.isalnum() for char in cleaned) >= 2:
            if not any(unicodedata.category(char) in ("Cc", "Cs") and char not in "\r\n\t" for char in original):
                return original
    raise ValueError(f"第 {line} 条的报修原文须含至少两个文字或数字，且不超过 240 字符。")


def parse_label(value, split: str, line: int) -> str | None:
    if isinstance(value, str):
        value = value.strip().lower()
    if split == "pending":
        if value is None or value == "":
            return None
        raise ValueError(f"第 {line} 条待分流工单的已确认故障类别须留空。")
    if not isinstance(value, str) or value not in STATIONS:
        raise ValueError(f"第 {line} 条已确认故障类别应填 blade、boot 或 fastener。")
    return value


def normalize_rows(rows: list[dict]) -> list[dict]:
    ids = set()
    shift_splits = {}
    training_counts = Counter()
    row_counts = Counter()
    normalized = []
    for line, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict) or raw.keys() != set(COLUMNS):
            raise ValueError(f"第 {line} 条的列须与模板一致：{', '.join(COLUMNS)}。")
        ticket_id = parse_code(raw["id"], "id", line)
        shift_id = parse_code(raw["shift_id"], "shift_id", line)
        if ticket_id in ids:
            raise ValueError(f"第 {line} 条的报修单号 {ticket_id} 重复。")
        ids.add(ticket_id)
        split = raw["split"].strip() if isinstance(raw["split"], str) else None
        if split not in SPLITS:
            raise ValueError(f"第 {line} 条的样本用途须为 train、holdout 或 pending。")
        if shift_splits.setdefault(shift_id, split) != split:
            raise ValueError(f"交接班 {shift_id} 同时出现在不同样本用途，请整班划分。")
        message = parse_message(raw["message"], line)
        label = parse_label(raw["fault_label"], split, line)
        normalized.append(dict(zip(COLUMNS, (ticket_id, shift_id, split, message, label))))
        row_counts[split] += 1
        if split == "train":
            training_counts[label] += 1

    if sum(split == "train" for split in shift_splits.values()) < 3:
        raise ValueError("历史训练至少需要 3 个完整班次。")
    if any(training_counts[label] < 2 for label in STATIONS):
        raise ValueError("每个故障类别至少需要 2 条已确认的历史训练工单。")
    if row_counts["holdout"] < 2:
        raise ValueError("留出评估至少需要 2 条来自独立班次的已确认工单。")
    return normalized
