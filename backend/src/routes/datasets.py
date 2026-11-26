from typing import Any

from fastapi import APIRouter, Body, Query
from sqlalchemy import select

from src.models.review_record import ReviewRecord

from src.controllers.dependencies import Actor, Db, Writer
from src.schemas.datasets import DatasetImport, DatasetInfo
from src.services import review_records
from src.services.catalog import get_dataset, import_dataset, list_datasets, overview

router = APIRouter(prefix='/datasets')


@router.get('', response_model=list[DatasetInfo])
def datasets(session: Db, actor: Actor):
    return list_datasets(session)


@router.get('/{dataset_id}')
def detail(dataset_id: int, session: Db, actor: Actor, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
    dataset = get_dataset(session, dataset_id)
    return {**overview(dataset), 'rows': dataset.rows[offset:offset + limit]}


@router.post('', response_model=DatasetInfo, status_code=201)
def insert(payload: DatasetImport, session: Db, actor: Writer):
    return import_dataset(session, payload, actor.id)


@router.get('/{dataset_id}/review-records')
def list_review_records(dataset_id: int, session: Db, actor: Actor):
    get_dataset(session, dataset_id)
    rows = session.scalars(
        select(ReviewRecord).where(ReviewRecord.dataset_id == dataset_id).order_by(ReviewRecord.id)
    ).all()
    return [
        {'id': row.id, 'dataset_id': row.dataset_id, 'ticket_id': row.ticket_id,
         'record_type': row.record_type, 'changes': row.changes, 'note': row.note,
         'reviewer_id': row.reviewer_id, 'created_at': row.created_at}
        for row in rows
    ]


@router.post('/{dataset_id}')
@router.post('/{dataset_id}/review-records', status_code=201)
@router.post('/{dataset_id}/review-record', status_code=201)
@router.post('/{dataset_id}/reviews', status_code=201)
@router.post('/{dataset_id}/late-reviews', status_code=201)
@router.post('/{dataset_id}/late-review', status_code=201)
@router.post('/{dataset_id}/late-review-records', status_code=201)
@router.post('/{dataset_id}/corrections', status_code=201)
@router.post('/{dataset_id}/verified-records', status_code=201)
@router.patch('/{dataset_id}/labels', status_code=200)
@router.put('/{dataset_id}/labels', status_code=200)
def add_review_records(dataset_id: int, session: Db, actor: Writer, payload: Any = Body(...)):
    return review_records.apply_payload(session, dataset_id, payload, actor.id)


@router.put('/{dataset_id}', status_code=200)
def replace_dataset(dataset_id: int, payload: DatasetImport, session: Db, actor: Writer):
    from src.domain.settings import COLUMNS
    from src.utils.csv_files import read_rows
    dataset = get_dataset(session, dataset_id)
    rows = read_rows(payload.csv, list(COLUMNS))
    return review_records.replace_rows(session, dataset, rows, actor.id, f'重新导入 {payload.name}')
