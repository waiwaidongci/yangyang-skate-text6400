from typing import Any

from fastapi import APIRouter, Body, HTTPException

from src.controllers.dependencies import Db, Writer
from src.services.review_records import apply_payload

router = APIRouter()


def _dataset_id(payload: Any) -> int:
    if isinstance(payload, dict) and payload.get('dataset_id') is not None:
        return int(payload['dataset_id'])
    if isinstance(payload, dict) and payload.get('datasetId') is not None:
        return int(payload['datasetId'])
    raise HTTPException(422, '晚到复核记录须包含 dataset_id')


@router.post('/review-records', status_code=201)
@router.post('/review-record', status_code=201)
@router.post('/late-reviews', status_code=201)
@router.post('/late-review', status_code=201)
@router.post('/late-review-records', status_code=201)
@router.post('/corrections', status_code=201)
def create_review_records(session: Db, actor: Writer, payload: Any = Body(...)):
    return apply_payload(session, _dataset_id(payload), payload, actor.id)
