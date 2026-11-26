import json
import logging

from sqlalchemy import select, update

from src.config.database import Session
from src.constants.workflow import RunStatus
from src.domain.project import analyze
from src.models.dataset import Dataset
from src.models.run import Run
from src.services.audit import record
from src.services.catalog import checksum_rows


def recover_jobs():
    with Session.begin() as session:
        session.execute(update(Run).where(Run.status == RunStatus.RUNNING).values(status=RunStatus.QUEUED))


def claim():
    with Session.begin() as session:
        run = session.scalar(select(Run).where(Run.status == RunStatus.QUEUED)
                             .order_by(Run.id).limit(1).with_for_update(skip_locked=True))
        if run is None:
            return None
        run.status = RunStatus.RUNNING
        dataset = session.get(Dataset, run.dataset_id)
        record(session, 'run.started', run.id)
        return run.id, dataset.rows, run.settings


def execute(job):
    run_id, rows, settings = job
    result = None
    error_message = None
    try:
        result = analyze(rows, settings)
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except ValueError as error:
        result = None
        error_message = str(error)
    except Exception:
        logging.getLogger(__name__).exception('计算运行 %s 失败', run_id)
        result = None
        error_message = '计算遇到内部错误，请检查台账后重试。'
    with Session.begin() as session:
        run = session.get(Run, run_id)
        dataset = session.get(Dataset, run.dataset_id) if run is not None else None
        source_changed = bool(
            run is not None and dataset is not None and run.dataset_checksum
            and run.dataset_checksum != checksum_rows(dataset.rows)
        )
        if run is None:
            logging.getLogger(__name__).error('计算任务 %s 找不到运行记录', run_id)
        elif source_changed or run.status == RunStatus.STALE:
            run.status = RunStatus.STALE
            run.result = None
            run.error = '计算期间台账被晚到复核记录修改，旧结果已失效，请重新发起运行。'
            record(session, 'run.stale', run.id)
        else:
            run.status = RunStatus.FAILED if error_message is not None else RunStatus.COMPLETED
            run.result = result
            run.error = error_message
            record(session, 'run.' + run.status, run.id)
