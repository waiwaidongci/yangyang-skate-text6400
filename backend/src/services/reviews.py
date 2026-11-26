from fastapi import HTTPException
from sqlalchemy import select

from src.constants.workflow import RunStatus
from src.models.dataset import Dataset
from src.models.review import Review
from src.models.run import Run
from src.services.audit import record
from src.services.catalog import checksum_rows
from src.utils.time import utc_stamp


def review_info(review):
    return {'run_id': review.run_id, 'decision': review.decision, 'note': review.note, 'created_at': review.created_at}


def submit(session, run_id, payload, actor):
    run = session.scalar(select(Run).where(Run.id == run_id).with_for_update())
    if not run:
        raise LookupError('找不到运行')

    dataset = session.get(Dataset, run.dataset_id)
    current_checksum = checksum_rows(dataset.rows) if dataset is not None else None
    source_changed = bool(
        run.dataset_checksum and current_checksum and run.dataset_checksum != current_checksum
    )
    if source_changed and run.status not in (RunStatus.STALE, RunStatus.INVALIDATED):
        run.status = RunStatus.STALE

    if run.status in (RunStatus.STALE, RunStatus.INVALIDATED) or source_changed:
        session.commit()
        raise HTTPException(409, '台账标签或分区已被晚到复核记录修改，旧交班单已失效；请重算后签核新单。')
    if run.status != RunStatus.COMPLETED or session.get(Review, run_id):
        raise HTTPException(409, '仅可复核一次，且计算必须成功完成')

    handoff = (run.result or {}).get('handoff_plan') if isinstance(run.result, dict) else None
    if payload.decision == 'approved' and handoff is not None and not handoff.get('signable', False):
        reason = handoff.get('reason') or '当前阈值方案未满足整班约束或留出复验要求'
        raise HTTPException(409, f'交班清单不满足签署条件：{reason}')

    review = Review(run_id=run_id, reviewer_id=actor.id, decision=payload.decision,
                    note=payload.note, created_at=utc_stamp())
    session.add(review)
    record(session, 'review.' + payload.decision, run_id, actor.id)
    session.commit()
    return review_info(review)
