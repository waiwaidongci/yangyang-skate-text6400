import math

from fastapi import HTTPException
from sqlalchemy import select

from src.constants.workflow import Decision, Role, RunStatus
from src.domain.project import SPEC
from src.domain.settings import is_optional_setting, validate_optional_setting
from src.models.dataset import Dataset
from src.models.review import Review
from src.models.run import Run
from src.services.audit import record
from src.services.catalog import checksum_rows, get_dataset
from src.services.queue import notify_worker
from src.utils.time import utc_stamp


def run_info(run, review):
    fields = ('id', 'label', 'dataset_id', 'created_at', 'status', 'settings', 'dataset_checksum')
    info = {**{key: getattr(run, key) for key in fields}, 'decision': review.decision if review else None}
    info['source_checksum'] = info['dataset_checksum']
    info['stale'] = (
        run.status in (RunStatus.STALE, RunStatus.INVALIDATED)
        or bool(getattr(run, 'source_changed', False))
    )
    if info['stale'] and run.status == RunStatus.COMPLETED:
        info['status'] = RunStatus.STALE
    info['invalidated'] = info['stale']
    info['obsolete'] = info['stale']
    info['expired'] = info['stale']
    return info


def refresh_staleness(session, run, review=None, dataset=None):
    if run is None:
        return False
    if review is None:
        review = session.get(Review, run.id)
    if review is not None and review.decision == Decision.APPROVED:
        return False
    if dataset is None:
        dataset = session.get(Dataset, run.dataset_id)
    changed = bool(
        dataset is not None and run.dataset_checksum
        and run.dataset_checksum != checksum_rows(dataset.rows)
    )
    if changed and run.status not in (RunStatus.STALE, RunStatus.INVALIDATED):
        run.status = RunStatus.STALE
        session.commit()
    run.source_changed = changed
    return changed


def list_runs(session, actor):
    statement = select(Run, Review).outerjoin(Review, Run.id == Review.run_id).order_by(Run.id.desc()).limit(100)
    if actor.role == Role.VIEWER:
        statement = statement.where(Review.decision == Decision.APPROVED)
    rows = list(session.execute(statement))
    dataset_ids = {run.dataset_id for run, _ in rows}
    datasets = {dataset.id: dataset for dataset in session.scalars(
        select(Dataset).where(Dataset.id.in_(dataset_ids))
    )} if dataset_ids else {}
    infos = []
    for run, review in rows:
        refresh_staleness(session, run, review, datasets.get(run.dataset_id))
        infos.append(run_info(run, review))
    return infos


def get_run(session, run_id, actor):
    run = session.get(Run, run_id)
    review = session.get(Review, run_id)
    if run is None or (actor.role == Role.VIEWER and (not review or review.decision != Decision.APPROVED)):
        raise LookupError('找不到可查看的运行')
    refresh_staleness(session, run, review)
    return run, review


def validate_settings(settings):
    parameters = {item['key']: item for item in SPEC['settings']}
    checked = {}
    for key, parameter in parameters.items():
        value = settings.get(key, parameter['default'])
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{parameter['label']}必须是数值")
        if not parameter['min'] <= value <= parameter['max']:
            raise ValueError(f"{parameter['label']}超出范围")
        if parameter['step'] == 1 and value != int(value):
            raise ValueError(f"{parameter['label']}必须是整数")
        checked[key] = int(value) if parameter['step'] == 1 else value

    for key, value in settings.items():
        if key in parameters:
            continue
        if not is_optional_setting(key):
            raise ValueError('存在本项目未定义的运行参数')
        validate_optional_setting(key, value)
        checked[key] = value
    return checked


def create_run(session, payload, actor):
    dataset = get_dataset(session, payload.dataset_id)
    run = Run(label=payload.label, dataset_id=payload.dataset_id,
              dataset_checksum=dataset.checksum,
              created_by=actor.id,
              created_at=utc_stamp(), status=RunStatus.QUEUED,
              settings=validate_settings(payload.settings))
    session.add(run)
    session.flush()
    record(session, 'run.queued', run.id, actor.id)
    session.commit()
    notify_worker()
    return run_info(run, None)


def recompute_run(session, run_id, actor, overrides: dict | None = None):
    old = session.get(Run, run_id)
    if old is None:
        raise LookupError('找不到运行')
    settings = dict(old.settings or {})
    label = f'{old.label}（重算）'[:140]
    if overrides:
        settings.update(overrides.get('settings') or {})
        if overrides.get('label'):
            label = str(overrides['label'])[:140]
    payload = type('RecomputePayload', (), {
        'label': label,
        'dataset_id': old.dataset_id,
        'settings': settings,
    })()
    return create_run(session, payload, actor)


def export_table(session, run_id, key, actor):
    run, review = get_run(session, run_id, actor)
    if review is None or review.decision != Decision.APPROVED:
        raise HTTPException(409, '复核通过后可下载正式 CSV')
    for table in run.result['tables']:
        if table['key'] == key:
            return table
    raise LookupError('找不到结果表')
