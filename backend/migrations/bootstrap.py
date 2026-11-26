from sqlalchemy import inspect, text

from src.config.database import Base, engine
from src.models import account, audit, dataset, review, review_record, run  # noqa: F401


def _ensure_columns() -> None:
    """Add columns introduced after the first snapshot on persistent volumes."""
    inspector = inspect(engine)
    additions = {
        'analysis_runs': {'dataset_checksum': 'VARCHAR(64)'},
        'datasets': {'revision': 'INTEGER NOT NULL DEFAULT 1'},
    }
    with engine.begin() as connection:
        for table, columns in additions.items():
            if table not in inspector.get_table_names():
                continue
            existing = {column['name'] for column in inspector.get_columns(table)}
            for name, declaration in columns.items():
                if name not in existing:
                    connection.execute(text(f'ALTER TABLE {table} ADD COLUMN {name} {declaration}'))


def bootstrap():
    Base.metadata.create_all(engine)
    _ensure_columns()
