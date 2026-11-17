"""Core data models.

The models are plain, immutable-where-practical dataclasses. Construction-time
checks guard invariants (e.g. scores are integers from 1 to 5, human scores
carry written feedback); schema validation of raw JSON lives in
:mod:`llm_eval.validation`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from llm_eval.errors import ValidationError
from llm_eval.rubric import VALID_SCORES, Rubric

#: Formats that the instruction-following checker understands.
SUPPORTED_FORMATS = ("bullet_list", "numbered_list", "json", "paragraph")

#: Absolute score difference at which human and automated scores are flagged.
DISAGREEMENT_THRESHOLD = 2


class ScoreSource(str, Enum):
    """Where a score came from."""

    HUMAN = "human"
    AUTOMATED = "automated"


@dataclass(frozen=True)
class Constraints:
    """Machine-checkable instructions attached to a prompt.

    Every field is optional; only declared constraints are checked.
    """

    max_words: int | None = None
    min_words: int | None = None
    max_sentences: int | None = None
    required_keywords: tuple[str, ...] = ()
    forbidden_phrases: tuple[str, ...] = ()
    response_format: str | None = None
    item_count: int | None = None

    def __post_init__(self) -> None:
        if (
            self.max_words is not None
            and self.min_words is not None
            and self.min_words > self.max_words
        ):
            raise ValidationError("min_words cannot be greater than max_words")
        if self.response_format is not None and self.response_format not in SUPPORTED_FORMATS:
            raise ValidationError(
                f"unsupported format {self.response_format!r}; expected one of "
                f"{list(SUPPORTED_FORMATS)}"
            )
        if self.item_count is not None and self.response_format not in (
            "bullet_list",
            "numbered_list",
        ):
            raise ValidationError("item_count requires format 'bullet_list' or 'numbered_list'")

    def is_empty(self) -> bool:
        return self.count() == 0

    def count(self) -> int:
        """Number of individual checks these constraints produce."""
        scalar = [
            self.max_words,
            self.min_words,
            self.max_sentences,
            self.response_format,
            self.item_count,
        ]
        return (
            sum(v is not None for v in scalar)
            + len(self.required_keywords)
            + len(self.forbidden_phrases)
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "max_words": self.max_words,
            "min_words": self.min_words,
            "max_sentences": self.max_sentences,
            "required_keywords": list(self.required_keywords),
            "forbidden_phrases": list(self.forbidden_phrases),
            "format": self.response_format,
            "item_count": self.item_count,
        }
        return {k: v for k, v in data.items() if v not in (None, [])}


@dataclass(frozen=True)
class EvaluationItem:
    """A single prompt/response pair to be evaluated."""

    id: str
    prompt: str
    response: str
    model: str | None = None
    reference_key_points: tuple[str, ...] = ()
    constraints: Constraints = field(default_factory=Constraints)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"id": self.id, "model": self.model, "prompt": self.prompt}
        data["response"] = self.response
        if self.reference_key_points:
            data["reference_key_points"] = list(self.reference_key_points)
        if not self.constraints.is_empty():
            data["constraints"] = self.constraints.to_dict()
        return data


@dataclass(frozen=True)
class Dataset:
    """A named collection of evaluation items with unique ids."""

    items: tuple[EvaluationItem, ...]
    name: str = "unnamed-dataset"
    description: str = ""

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for item in self.items:
            if item.id in seen:
                raise ValidationError(f"duplicate item id {item.id!r}")
            seen.add(item.id)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(i.id for i in self.items)


@dataclass(frozen=True)
class CriterionScore:
    """A score for one criterion, from a single source, with its justification.

    Attributes:
        criterion: Rubric criterion key.
        score: Integer from 1 to 5.
        source: Whether a human or the automated heuristics produced the score.
        feedback: Written explanation of the score. Required for every score.
        evidence: Optional supporting references (URLs, citations, quotes).
        annotator: Identifier of the human reviewer (human scores only).
        signals: Raw measurements behind an automated score, for transparency.
    """

    criterion: str
    score: int
    source: ScoreSource
    feedback: str
    evidence: tuple[str, ...] = ()
    annotator: str | None = None
    signals: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.score, bool) or not isinstance(self.score, int):
            raise ValidationError(f"score must be an integer, got {type(self.score).__name__}")
        if self.score not in VALID_SCORES:
            raise ValidationError(f"score must be between 1 and 5, got {self.score}")
        if not isinstance(self.feedback, str) or not self.feedback.strip():
            raise ValidationError("feedback explaining the score is required")
        if not isinstance(self.source, ScoreSource):
            object.__setattr__(self, "source", ScoreSource(self.source))

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "score": self.score,
            "source": self.source.value,
            "feedback": self.feedback,
        }
        if self.source is ScoreSource.HUMAN:
            data["annotator"] = self.annotator
            data["evidence"] = list(self.evidence)
        else:
            data["signals"] = dict(self.signals)
        return data


@dataclass(frozen=True)
class HumanJudgment:
    """All human scores for a single item from one annotator."""

    item_id: str
    annotator: str
    scores: Mapping[str, CriterionScore]
    comment: str | None = None

    def __post_init__(self) -> None:
        for key, score in self.scores.items():
            if score.source is not ScoreSource.HUMAN:
                raise ValidationError(f"judgment score for {key!r} must have source 'human'")
            if score.criterion != key:
                raise ValidationError(f"score keyed {key!r} is for criterion {score.criterion!r}")


@dataclass(frozen=True)
class JudgmentSet:
    """Human judgments keyed by item id."""

    judgments: Mapping[str, HumanJudgment]
    description: str = ""

    def get(self, item_id: str) -> HumanJudgment | None:
        return self.judgments.get(item_id)

    def __len__(self) -> int:
        return len(self.judgments)


@dataclass(frozen=True)
class CriterionResult:
    """Human and automated scores for one criterion of one item.

    The *final* score follows a simple, explicit policy: a human score always
    takes precedence; an automated score is used only when no human score exists.
    """

    criterion: str
    human: CriterionScore | None = None
    automated: CriterionScore | None = None

    @property
    def final(self) -> CriterionScore | None:
        return self.human if self.human is not None else self.automated

    @property
    def difference(self) -> int | None:
        """Human score minus automated score, when both exist."""
        if self.human is None or self.automated is None:
            return None
        return self.human.score - self.automated.score

    @property
    def needs_review(self) -> bool:
        """True when human and automated scores differ by >= the threshold."""
        diff = self.difference
        return diff is not None and abs(diff) >= DISAGREEMENT_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        final = self.final
        return {
            "final_score": final.score if final else None,
            "final_source": final.source.value if final else None,
            "human": self.human.to_dict() if self.human else None,
            "automated": self.automated.to_dict() if self.automated else None,
            "human_minus_automated": self.difference,
            "needs_review": self.needs_review,
        }


@dataclass(frozen=True)
class OverallScore:
    """Weighted mean of the available criterion scores.

    Attributes:
        score: Weighted mean on the 1-5 scale, or ``None`` if nothing was scored.
        criteria_used: Criteria that contributed to the score.
        missing: Criteria with no score from any source.
        sources: How many contributing criteria came from each source.
    """

    score: float | None
    criteria_used: tuple[str, ...]
    missing: tuple[str, ...]
    sources: Mapping[str, int]

    @property
    def complete(self) -> bool:
        return not self.missing

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "complete": self.complete,
            "criteria_used": list(self.criteria_used),
            "missing_criteria": list(self.missing),
            "score_sources": dict(self.sources),
        }


@dataclass(frozen=True)
class ItemResult:
    """Evaluation outcome for one item."""

    item: EvaluationItem
    criteria: Mapping[str, CriterionResult]
    overall: OverallScore
    human_overall: float | None
    automated_overall: float | None
    annotator: str | None = None
    comment: str | None = None

    @property
    def needs_review(self) -> bool:
        return any(r.needs_review for r in self.criteria.values())

    def to_dict(self) -> dict[str, Any]:
        item = self.item.to_dict()
        return {
            "item_id": item.pop("id"),
            **item,
            "criteria": {k: v.to_dict() for k, v in self.criteria.items()},
            "overall": self.overall.to_dict(),
            "human_overall": self.human_overall,
            "automated_overall": self.automated_overall,
            "annotator": self.annotator,
            "reviewer_comment": self.comment,
            "needs_review": self.needs_review,
        }


@dataclass(frozen=True)
class EvaluationRun:
    """The complete result of evaluating a dataset."""

    dataset: Dataset
    rubric: Rubric
    results: tuple[ItemResult, ...]
    automated_enabled: bool = True
    summary: Mapping[str, Any] = field(default_factory=dict)
