"""Fit a frozen historical ComplementNB model and produce per-ticket routing evidence."""

from collections import Counter
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB

from .features import character_vectorizer
from .settings import MANUAL_STATION, STATIONS


@dataclass(frozen=True)
class Prediction:
    ticket: dict
    label: str
    probabilities: dict[str, float]
    gap: float
    matched_ngrams: int
    manual_reason: str | None

    @property
    def confidence(self) -> float:
        return self.probabilities[self.label]

    @property
    def station(self) -> str:
        return MANUAL_STATION if self.manual_reason else STATIONS[self.label]


@dataclass(frozen=True)
class RoutingModel:
    vectorizer: TfidfVectorizer
    classifier: ComplementNB
    training_counts: Counter
    baseline_label: str

    @property
    def vocabulary_size(self) -> int:
        return len(self.vectorizer.vocabulary_)

    def score(self, rows: list[dict], threshold: float | dict[str, float] = 0.70) -> list[Prediction]:
        if not rows:
            return []
        matrix = self.vectorizer.transform([ticket["message"] for ticket in rows])
        predictions = []
        for ticket, values, matches in zip(
            rows, self.classifier.predict_proba(matrix), matrix.getnnz(axis=1), strict=True,
        ):
            scores = {str(label): float(value) for label, value in zip(self.classifier.classes_, values, strict=True)}
            # ComplementNB is trained on all three business classes in normal operation.
            # Keep every score key present even when a custom fold accidentally lacks a class.
            for label in STATIONS:
                scores.setdefault(label, 0.0)
            matches = int(matches)
            ranked = sorted(scores, key=lambda label: (-scores[label], label))
            gap = scores[ranked[0]] - scores[ranked[1]]
            label = ranked[0] if matches else self.baseline_label
            if isinstance(threshold, dict):
                cutoff = float(threshold.get(label, threshold.get("confidence_threshold", 1.0)))
            else:
                cutoff = float(threshold)
            reason = None
            if matches == 0:
                reason = "未命中历史字符片段"
            elif gap <= 1e-12:
                reason = "最高两类分数并列"
            elif scores[label] < cutoff:
                reason = "未达到自动分站置信线"
            predictions.append(Prediction(ticket, label, scores, gap, matches, reason))
        return predictions
        return predictions


def fit_routing_model(train: list[dict], alpha: float) -> RoutingModel:
    vectorizer = character_vectorizer()
    matrix = vectorizer.fit_transform([ticket["message"] for ticket in train])
    labels = [ticket["fault_label"] for ticket in train]
    classifier = ComplementNB(alpha=alpha, norm=False).fit(matrix, labels)
    training_counts = Counter(labels)
    baseline_label = min(training_counts, key=lambda label: (-training_counts[label], label))
    return RoutingModel(vectorizer, classifier, training_counts, baseline_label)
