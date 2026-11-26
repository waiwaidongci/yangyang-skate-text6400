from sqlalchemy import JSON, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from src.config.database import Base


class ReviewRecord(Base):
    """A late counter/repair verification that can correct a frozen ledger."""

    __tablename__ = 'review_records'

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey('datasets.id'), index=True)
    ticket_id: Mapped[str] = mapped_column(String(64), index=True)
    record_type: Mapped[str] = mapped_column(String(32), default='label_correction')
    changes: Mapped[dict] = mapped_column(JSON)
    note: Mapped[str | None] = mapped_column(String(500), nullable=True)
    reviewer_id: Mapped[int | None] = mapped_column(ForeignKey('accounts.id'), nullable=True)
    created_at: Mapped[str] = mapped_column(String(32), index=True)
