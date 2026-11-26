"""Character features that preserve Chinese repair phrases and normalize width variants."""

import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer

from .settings import NGRAM_RANGE


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def character_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char",
        preprocessor=normalize_text,
        ngram_range=NGRAM_RANGE,
        smooth_idf=True,
        sublinear_tf=True,
        norm="l2",
    )
