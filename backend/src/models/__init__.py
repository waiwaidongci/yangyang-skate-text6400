"""Register all SQLAlchemy tables in every entry point using this package."""

from .account import Account
from .audit import Audit
from .dataset import Dataset
from .review import Review
from .review_record import ReviewRecord
from .run import Run

__all__ = ['Account', 'Audit', 'Dataset', 'Review', 'ReviewRecord', 'Run']
