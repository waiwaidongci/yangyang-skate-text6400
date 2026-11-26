"""Late verification records that invalidate unsigned analysis runs."""

from __future__ import annotations

import csv
import io
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select

from src.constants.workflow import RunStatus
from src.domain.settings import COLUMNS
from src.models.dataset import Dataset
from src.models.review import Review
from src.models.review_record import ReviewRecord
from src.models.run import Run
from src.services.audit import record as audit_record
from src.services.catalog import checksum_rows
from src.utils.time import utc_stamp

ID_KEYS = ('id', 'ticket_id', 'ticket', 'ticket_no', 'work_order_id', 'order_id', '工单号', '单号')
LABEL_KEYS = (
    'fault_label', 'new_fault_label', 'label', 'new_label', 'confirmed_label',
    'confirmed_fault_label', 'corrected_label', 'updated_label', 'main_fault',
    'category', '主故障类别', '主故障', '标签',
)
SPLIT_KEYS = ('split', 'new_split', 'cohort', 'partition', 'sample_use', '样本用途', '训练留出标签')
SHIFT_KEYS = ('shift_id', 'new_shift_id', 'shift', 'shift_code', '班次', '交接班号')
MESSAGE_KEYS = ('message', '报修原文')


def _first(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    def norm(value: object) -> str:
        return ''.join(ch for ch in str(value).strip().lower() if ch not in '_- ')

    normalized = {norm(key): value for key, value in data.items()}
    for key in keys:
        value = normalized.get(norm(key))
        if value not in (None, ''):
            return value
    return None


def _records_from_csv(text: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(text))
    return [dict(row) for row in reader]


def normalize_update_list(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, str):
        return _records_from_csv(payload)
    if isinstance(payload, dict):
        for key in ('records', 'updates', 'review_records', 'items', 'rows'):
            if isinstance(payload.get(key), list):
                return payload[key]
        # A single update object.
        return [payload]
    if isinstance(payload, list):
        return payload
    raise HTTPException(422, '复核记录须为更新对象列表或 CSV 文本')


def invalidate_unsigned_runs(session, dataset_id: int) -> list[int]:
    runs = list(session.scalars(select(Run).where(Run.dataset_id == dataset_id)).all())
    invalidated = []
    for run in runs:
        review = session.get(Review, run.id)
        if review is not None and review.decision == 'approved':
            continue
        if run.status not in (RunStatus.STALE, RunStatus.INVALIDATED):
            run.status = RunStatus.STALE
            invalidated.append(run.id)
    return invalidated


def apply_records(session, dataset: Dataset, updates: list[dict[str, Any]], actor_id: int,
                  note: str | None = None) -> dict[str, Any]:
    if not updates:
        raise HTTPException(422, '请至少提供一条晚到复核记录')

    rows = [dict(row) for row in dataset.rows]
    by_id = {str(row['id']).upper(): row for row in rows}
    applied = []
    record_rows = []

    for raw_update in updates:
        if not isinstance(raw_update, dict):
            raise HTTPException(422, '每条复核记录须为对象')
        data = dict(raw_update)
        for nested_key in ('changes', 'fields', 'correction', 'review', 'record'):
            nested = raw_update.get(nested_key)
            if isinstance(nested, dict):
                data.update(nested)
        for key, value in list(data.items()):
            if isinstance(value, dict) and 'new' in value:
                data[key] = value['new']

        ticket_value = _first(data, ID_KEYS)
        if ticket_value is None:
            raise HTTPException(422, '复核记录缺少工单号')
        ticket_id = str(ticket_value).strip().upper()
        target = by_id.get(ticket_id)

        # A complete CSV row can act as an upsert/replacement.
        complete_row = all(column in data and data[column] not in (None, '') for column in COLUMNS)
        if target is None:
            if not complete_row:
                raise HTTPException(404, f'找不到工单 {ticket_id}')
            target = {}
            rows.append(target)
            by_id[ticket_id] = target

        changes = {}
        if complete_row:
            new_row = {column: data[column] for column in COLUMNS}
            changes = {column: {'old': target.get(column), 'new': new_row[column]}
                       for column in COLUMNS if target.get(column) != new_row[column]}
            target.clear()
            target.update(new_row)
        else:
            label = _first(data, LABEL_KEYS)
            if label is not None:
                new_label = str(label).strip().lower()
                changes['fault_label'] = {'old': target.get('fault_label'), 'new': new_label}
                target['fault_label'] = new_label
            split = _first(data, SPLIT_KEYS)
            if split is not None:
                new_split = str(split).strip().lower()
                changes['split'] = {'old': target.get('split'), 'new': new_split}
                target['split'] = new_split
            shift_id = _first(data, SHIFT_KEYS)
            if shift_id is not None:
                new_shift = str(shift_id).strip().upper()
                changes['shift_id'] = {'old': target.get('shift_id'), 'new': new_shift}
                target['shift_id'] = new_shift
            message = _first(data, MESSAGE_KEYS)
            if message is not None:
                changes['message'] = {'old': target.get('message'), 'new': message}
                target['message'] = message

        if not changes:
            continue
        applied.append({'id': ticket_id, 'changes': changes})
        review_record = ReviewRecord(
            dataset_id=dataset.id,
            ticket_id=ticket_id,
            record_type='label_correction',
            changes=changes,
            note=note,
            reviewer_id=actor_id,
            created_at=utc_stamp(),
        )
        session.add(review_record)
        record_rows.append(review_record)

    if not applied:
        raise HTTPException(422, '复核记录没有产生标签或分区变更')

    # Re-run the same CSV/domain boundary validation before moving the snapshot.
    from src.domain.records import normalize_rows
    normalized = normalize_rows(rows)
    dataset.rows = normalized
    dataset.checksum = checksum_rows(normalized)
    dataset.revision = int(dataset.revision or 1) + 1
    invalidated_run_ids = invalidate_unsigned_runs(session, dataset.id)
    audit_record(session, 'review_record.applied', dataset.id, actor_id)
    session.commit()
    return {
        'dataset_id': dataset.id,
        'revision': dataset.revision,
        'checksum': dataset.checksum,
        'applied': applied,
        'invalidated_run_ids': invalidated_run_ids,
        # Compatibility aliases for callers describing the same event.
        'stale_run_ids': invalidated_run_ids,
        'old_sheets_invalidated': invalidated_run_ids,
    }


def replace_rows(session, dataset: Dataset, rows: list[dict[str, Any]], actor_id: int,
                 note: str | None = None) -> dict[str, Any]:
    from src.domain.records import normalize_rows
    normalized = normalize_rows(rows)
    old_checksum = dataset.checksum
    dataset.rows = normalized
    dataset.checksum = checksum_rows(normalized)
    dataset.revision = int(dataset.revision or 1) + 1
    invalidated_run_ids = invalidate_unsigned_runs(session, dataset.id)
    session.add(ReviewRecord(
        dataset_id=dataset.id,
        ticket_id='*',
        record_type='full_reimport',
        changes={'checksum': {'old': old_checksum, 'new': dataset.checksum}},
        note=note,
        reviewer_id=actor_id,
        created_at=utc_stamp(),
    ))
    audit_record(session, 'dataset.replaced', dataset.id, actor_id)
    session.commit()
    return {
        'dataset_id': dataset.id,
        'revision': dataset.revision,
        'checksum': dataset.checksum,
        'invalidated_run_ids': invalidated_run_ids,
        'stale_run_ids': invalidated_run_ids,
        'old_sheets_invalidated': invalidated_run_ids,
    }


def apply_payload(session, dataset_id: int, payload: Any, actor_id: int) -> dict[str, Any]:
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise LookupError('找不到来源台账')
    if isinstance(payload, dict):
        note = payload.get('note')
        csv_text = payload.get('csv') if isinstance(payload.get('csv'), str) else None
        updates = normalize_update_list(csv_text if csv_text is not None else payload)
        note = note if isinstance(note, str) else None
    else:
        updates = normalize_update_list(payload)
        note = None
    return apply_records(session, dataset, updates, actor_id, note)
