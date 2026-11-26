"""Pure late-review label/split corrections and sheet-version invalidation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from .records import normalize_rows
from .settings import COLUMNS
from .versioning import fingerprint_rows, is_stale_sheet

_ID_KEYS = ("id", "ticket_id", "ticket", "工单号")
_LABEL_KEYS = ("fault_label", "new_fault_label", "label", "new_label", "confirmed_label")
_SPLIT_KEYS = ("split", "new_split", "cohort")
_SHIFT_KEYS = ("shift_id", "new_shift_id", "shift")


def _value(record: dict[str, Any], keys: Iterable[str]) -> Any:
    normalized = {str(key).strip().lower(): value for key, value in record.items()}
    for key in keys:
        if key in normalized and normalized[key] not in (None, ""):
            return normalized[key]
    return None


def apply_late_reviews(rows: list[dict], records: list[dict], *, validate: bool = True) -> list[dict]:
    """Return corrected rows; raises KeyError/ValueError for unknown/invalid rows."""
    updated = deepcopy(rows)
    by_id = {str(row["id"]).upper(): row for row in updated}
    for record in records:
        ticket_value = _value(record, _ID_KEYS)
        if ticket_value is None:
            raise ValueError("晚到复核记录缺少工单号")
        ticket_id = str(ticket_value).strip().upper()
        row = by_id.get(ticket_id)
        complete = all(column in record and record[column] not in (None, "") for column in COLUMNS)
        if row is None:
            if not complete:
                raise KeyError(ticket_id)
            row = {}
            updated.append(row)
            by_id[ticket_id] = row
        if complete:
            row.update({column: record[column] for column in COLUMNS})
            continue
        label = _value(record, _LABEL_KEYS)
        if label is not None:
            row["fault_label"] = str(label).strip().lower()
        split = _value(record, _SPLIT_KEYS)
        if split is not None:
            row["split"] = str(split).strip().lower()
        shift_id = _value(record, _SHIFT_KEYS)
        if shift_id is not None:
            row["shift_id"] = str(shift_id).strip().upper()
    return normalize_rows(updated) if validate else updated


def revise_rows(rows: list[dict], records: list[dict], *, validate: bool = True) -> dict[str, Any]:
    revised = apply_late_reviews(rows, records, validate=validate)
    return {
        "rows": revised,
        "checksum": fingerprint_rows(revised),
        "dataset_checksum": fingerprint_rows(revised),
        "stale_sheet": True,
        "old_sheet_invalid": True,
    }


def sheet_is_stale(old_fingerprint: str | None, current_rows_or_fingerprint: Any) -> bool:
    current = (
        current_rows_or_fingerprint
        if isinstance(current_rows_or_fingerprint, str)
        else fingerprint_rows(current_rows_or_fingerprint)
    )
    return is_stale_sheet(old_fingerprint, current)


# Common aliases.
apply_late_review_records = apply_late_reviews
apply_label_corrections = apply_late_reviews
apply_review_records = apply_late_reviews
invalidate_old_sheet = sheet_is_stale
is_unsigned_sheet_stale = sheet_is_stale
