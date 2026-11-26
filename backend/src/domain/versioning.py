"""Small version helpers used by late review-record invalidation tests."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def fingerprint_rows(rows: Any) -> str:
    encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_stale_sheet(run_fingerprint: str | None, current_fingerprint: str | None) -> bool:
    return bool(run_fingerprint and current_fingerprint and run_fingerprint != current_fingerprint)


# Vocabulary aliases.
dataset_fingerprint = fingerprint_rows
rows_checksum = fingerprint_rows
is_sheet_stale = is_stale_sheet
is_old_sheet_invalid = is_stale_sheet
