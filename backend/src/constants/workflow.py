from enum import StrEnum


class Role(StrEnum):
    ADMIN = 'admin'
    ANALYST = 'analyst'
    VIEWER = 'viewer'


class RunStatus(StrEnum):
    QUEUED = 'queued'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    STALE = 'stale'
    INVALIDATED = 'invalidated'


class Decision(StrEnum):
    APPROVED = 'approved'
    REJECTED = 'rejected'
