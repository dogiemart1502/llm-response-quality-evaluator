"""Scoring rubric definitions.

Each criterion is scored on an integer 1-5 scale. Level descriptors are written
so that a human reviewer can apply them consistently; the ``automated_proxy``
field documents exactly what the (much weaker) automated heuristic measures for
that criterion, so nobody mistakes one for the other.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from llm_eval.errors import ValidationError

SCORE_MIN = 1
SCORE_MAX = 5
VALID_SCORES = tuple(range(SCORE_MIN, SCORE_MAX + 1))


@dataclass(frozen=True)
class Criterion:
    """A single rubric dimension.

    Attributes:
        key: Stable machine-readable identifier (e.g. ``"accuracy"``).
        name: Display name.
        description: What the criterion measures.
        levels: Descriptor for each score from 1 to 5.
        automated_proxy: Plain-language description of the automated signal
            used for this criterion and its limits.
        default_weight: Relative weight used when computing overall scores.
    """

    key: str
    name: str
    description: str
    levels: Mapping[int, str]
    automated_proxy: str
    default_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.key or not self.key.isidentifier():
            raise ValidationError(f"criterion key {self.key!r} must be a valid identifier")
        missing = [s for s in VALID_SCORES if s not in self.levels]
        if missing:
            raise ValidationError(f"criterion {self.key!r} is missing level descriptors {missing}")
        if self.default_weight <= 0 or not math.isfinite(self.default_weight):
            raise ValidationError(f"criterion {self.key!r} must have a positive weight")
        # Freeze the mapping so the rubric cannot be mutated after creation.
        object.__setattr__(self, "levels", MappingProxyType(dict(self.levels)))

    def describe(self, score: int) -> str:
        """Return the level descriptor for ``score``."""
        if score not in self.levels:
            raise ValidationError(f"score must be one of {list(VALID_SCORES)}, got {score!r}")
        return self.levels[score]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "levels": {str(k): v for k, v in sorted(self.levels.items())},
            "automated_proxy": self.automated_proxy,
        }


ACCURACY = Criterion(
    key="accuracy",
    name="Accuracy",
    description=(
        "Factual correctness of the claims in the response. Human reviewers should "
        "cite evidence (a source, a reference answer, or reasoning) for their score."
    ),
    levels={
        1: "Mostly incorrect or contains serious factual errors / fabrications.",
        2: "Contains significant errors that undermine the answer.",
        3: "Generally correct but with notable errors or omissions.",
        4: "Correct with only minor imprecision.",
        5: "Fully correct; no factual errors identified.",
    },
    automated_proxy=(
        "Coverage of human-supplied reference key points (lexical match). This measures "
        "agreement with the reference, NOT truth: it cannot detect contradictions, negation, "
        "or errors that are absent from the reference. Not scored when no reference is given."
    ),
    default_weight=0.35,
)

RELEVANCE = Criterion(
    key="relevance",
    name="Relevance",
    description="How directly the response addresses the user's prompt.",
    levels={
        1: "Off-topic; does not address the prompt.",
        2: "Mostly unrelated, with only passing reference to the prompt.",
        3: "Partially addresses the prompt or includes substantial tangents.",
        4: "Addresses the prompt with minor digressions.",
        5: "Fully focused on what was asked.",
    },
    automated_proxy=(
        "Fraction of the prompt's content terms that appear in the response. A lexical "
        "proxy: it misses paraphrases and cannot judge whether the topic is handled well."
    ),
    default_weight=0.20,
)

CLARITY = Criterion(
    key="clarity",
    name="Clarity",
    description="How easy the response is to read and understand.",
    levels={
        1: "Very hard to follow; confusing or incoherent.",
        2: "Difficult to follow; poorly organised or overly convoluted.",
        3: "Understandable but could be clearer or better structured.",
        4: "Clear and well organised with minor issues.",
        5: "Exceptionally clear, concise, and well structured.",
    },
    automated_proxy=(
        "Readability heuristics: sentence length, Flesch reading ease, duplicated sentences "
        "and term repetition. Does not assess logical coherence."
    ),
    default_weight=0.20,
)

INSTRUCTION_FOLLOWING = Criterion(
    key="instruction_following",
    name="Instruction Following",
    description="Whether the response obeys explicit constraints in the prompt.",
    levels={
        1: "Ignores the instructions.",
        2: "Follows few of the instructions.",
        3: "Follows some instructions but misses important ones.",
        4: "Follows nearly all instructions with a minor deviation.",
        5: "Follows every instruction.",
    },
    automated_proxy=(
        "Fraction of machine-checkable constraints satisfied (word/sentence limits, required "
        "or forbidden phrases, list/JSON format, item count). Only checks constraints that "
        "were declared in the dataset. Not scored when no constraints are declared."
    ),
    default_weight=0.25,
)

DEFAULT_CRITERIA: tuple[Criterion, ...] = (ACCURACY, RELEVANCE, CLARITY, INSTRUCTION_FOLLOWING)


@dataclass(frozen=True)
class Rubric:
    """An ordered collection of criteria with normalised weights.

    Args:
        criteria: The criteria to include. Defaults to :data:`DEFAULT_CRITERIA`.
        weights: Optional overrides mapping criterion key to a positive weight.
            Weights are normalised to sum to 1.
    """

    criteria: tuple[Criterion, ...] = DEFAULT_CRITERIA
    weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        criteria = tuple(self.criteria)
        if not criteria:
            raise ValidationError("a rubric needs at least one criterion")
        keys = [c.key for c in criteria]
        if len(set(keys)) != len(keys):
            raise ValidationError(f"duplicate criterion keys in rubric: {keys}")

        overrides = dict(self.weights)
        unknown = sorted(set(overrides) - set(keys))
        if unknown:
            raise ValidationError(f"weights given for unknown criteria: {unknown}")

        raw: dict[str, float] = {}
        for c in criteria:
            w = overrides.get(c.key, c.default_weight)
            if isinstance(w, bool) or not isinstance(w, (int, float)):
                raise ValidationError(f"weight for {c.key!r} must be a number")
            if not math.isfinite(w) or w <= 0:
                raise ValidationError(f"weight for {c.key!r} must be a positive finite number")
            raw[c.key] = float(w)

        total = sum(raw.values())
        normalised = {k: v / total for k, v in raw.items()}
        object.__setattr__(self, "criteria", criteria)
        object.__setattr__(self, "weights", MappingProxyType(normalised))

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(c.key for c in self.criteria)

    def get(self, key: str) -> Criterion:
        for c in self.criteria:
            if c.key == key:
                return c
        raise ValidationError(f"unknown criterion {key!r}; expected one of {list(self.keys)}")

    def weight(self, key: str) -> float:
        self.get(key)  # raises for unknown keys
        return self.weights[key]

    def with_weights(self, weights: Mapping[str, float]) -> Rubric:
        """Return a copy of this rubric with some weights overridden."""
        merged = {c.key: c.default_weight for c in self.criteria}
        merged.update(weights)
        return Rubric(self.criteria, merged)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": {"min": SCORE_MIN, "max": SCORE_MAX},
            "criteria": [
                {**c.to_dict(), "weight": round(self.weights[c.key], 4)} for c in self.criteria
            ],
        }


def parse_weights(spec: str | Iterable[str]) -> dict[str, float]:
    """Parse ``"accuracy=0.4,clarity=0.2"`` style weight overrides.

    Raises:
        ValidationError: If the specification is malformed.
    """
    parts = spec.split(",") if isinstance(spec, str) else list(spec)
    weights: dict[str, float] = {}
    for part in parts:
        part = part.strip()
        if not part:
            continue
        key, sep, value = part.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValidationError(f"invalid weight {part!r}; expected the form criterion=number")
        try:
            weights[key] = float(value)
        except ValueError:
            raise ValidationError(
                f"weight for {key!r} is not a number: {value.strip()!r}"
            ) from None
    if not weights:
        raise ValidationError("no weights were provided")
    return weights
