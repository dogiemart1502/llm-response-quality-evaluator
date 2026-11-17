"""Validate raw (JSON-decoded) input and convert it into model objects.

Every error message includes the path of the offending value, e.g.
``items[3].constraints.max_words: must be a positive integer``, so problems in
large files are easy to locate.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from llm_eval.errors import ValidationError
from llm_eval.models import (
    Constraints,
    CriterionScore,
    Dataset,
    EvaluationItem,
    HumanJudgment,
    JudgmentSet,
    ScoreSource,
)
from llm_eval.rubric import Rubric

MAX_TEXT_LENGTH = 100_000

_ITEM_FIELDS = {"id", "prompt", "response", "model", "reference_key_points", "constraints"}
_CONSTRAINT_FIELDS = {
    "max_words",
    "min_words",
    "max_sentences",
    "required_keywords",
    "forbidden_phrases",
    "format",
    "item_count",
}
_SCORE_FIELDS = {"score", "feedback", "evidence"}
_JUDGMENT_FIELDS = {"item_id", "annotator", "scores", "comment"}


# ------------------------------------------------------------------ primitive helpers


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"must be an object, got {type(value).__name__}", path)
    return value


def _reject_unknown(data: dict[str, Any], allowed: Iterable[str], path: str) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ValidationError(
            f"unknown field(s) {unknown}; allowed fields are {sorted(allowed)}", path
        )


def _string(value: Any, path: str, *, required: bool = True, allow_empty: bool = False) -> str:
    if value is None:
        if required:
            raise ValidationError("is required", path)
        return ""
    if not isinstance(value, str):
        raise ValidationError(f"must be a string, got {type(value).__name__}", path)
    if not allow_empty and not value.strip():
        raise ValidationError("must not be empty", path)
    if len(value) > MAX_TEXT_LENGTH:
        raise ValidationError(f"is too long ({len(value)} > {MAX_TEXT_LENGTH} characters)", path)
    return value


def _optional_string(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _positive_int(value: Any, path: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError(f"must be a positive integer, got {value!r}", path)
    return value


def _string_list(value: Any, path: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValidationError(f"must be a list of strings, got {type(value).__name__}", path)
    return tuple(_string(v, f"{path}[{i}]") for i, v in enumerate(value))


def _wrap(path: str, exc: ValidationError) -> ValidationError:
    """Prefix a model-level ValidationError with the location it came from."""
    inner = f"{exc.path}: {exc.message}" if exc.path else exc.message
    return ValidationError(inner, path)


# --------------------------------------------------------------------------- datasets


def parse_constraints(data: Any, path: str = "constraints") -> Constraints:
    if data is None:
        return Constraints()
    data = _require_mapping(data, path)
    _reject_unknown(data, _CONSTRAINT_FIELDS, path)
    fmt = data.get("format")
    if fmt is not None:
        fmt = _string(fmt, f"{path}.format")
    try:
        return Constraints(
            max_words=_positive_int(data.get("max_words"), f"{path}.max_words"),
            min_words=_positive_int(data.get("min_words"), f"{path}.min_words"),
            max_sentences=_positive_int(data.get("max_sentences"), f"{path}.max_sentences"),
            required_keywords=_string_list(
                data.get("required_keywords"), f"{path}.required_keywords"
            ),
            forbidden_phrases=_string_list(
                data.get("forbidden_phrases"), f"{path}.forbidden_phrases"
            ),
            response_format=fmt,
            item_count=_positive_int(data.get("item_count"), f"{path}.item_count"),
        )
    except ValidationError as exc:
        if exc.path:
            raise
        raise ValidationError(exc.message, path) from None


def parse_item(data: Any, path: str = "item") -> EvaluationItem:
    data = _require_mapping(data, path)
    _reject_unknown(data, _ITEM_FIELDS, path)
    item_id = data.get("id")
    if isinstance(item_id, int) and not isinstance(item_id, bool):
        item_id = str(item_id)
    return EvaluationItem(
        id=_string(item_id, f"{path}.id"),
        prompt=_string(data.get("prompt"), f"{path}.prompt"),
        # An empty response is legitimate input: it should score poorly, not crash.
        response=_string(data.get("response"), f"{path}.response", allow_empty=True),
        model=_optional_string(data.get("model"), f"{path}.model"),
        reference_key_points=_string_list(
            data.get("reference_key_points"), f"{path}.reference_key_points"
        ),
        constraints=parse_constraints(data.get("constraints"), f"{path}.constraints"),
    )


def parse_dataset(data: Any) -> Dataset:
    """Validate a dataset document: ``{"name": ..., "items": [...]}``."""
    data = _require_mapping(data, "dataset")
    _reject_unknown(data, {"name", "description", "items"}, "dataset")
    raw_items = data.get("items")
    if not isinstance(raw_items, list):
        raise ValidationError("must be a list of items", "items")
    if not raw_items:
        raise ValidationError("must contain at least one item", "items")
    items = tuple(parse_item(raw, f"items[{i}]") for i, raw in enumerate(raw_items))

    seen: dict[str, int] = {}
    for i, item in enumerate(items):
        if item.id in seen:
            raise ValidationError(
                f"duplicate id {item.id!r} (first used at items[{seen[item.id]}])", f"items[{i}].id"
            )
        seen[item.id] = i

    return Dataset(
        items=items,
        name=_optional_string(data.get("name"), "name") or "unnamed-dataset",
        description=_string(
            data.get("description"), "description", required=False, allow_empty=True
        ),
    )


# -------------------------------------------------------------------------- judgments


def parse_human_score(data: Any, criterion: str, annotator: str, path: str) -> CriterionScore:
    data = _require_mapping(data, path)
    _reject_unknown(data, _SCORE_FIELDS, path)
    score = data.get("score")
    if score is None:
        raise ValidationError("is required", f"{path}.score")
    if isinstance(score, bool) or not isinstance(score, int):
        # Accept 4.0 but not 4.5: the rubric is an integer scale.
        if isinstance(score, float) and score.is_integer():
            score = int(score)
        else:
            raise ValidationError(f"must be an integer from 1 to 5, got {score!r}", f"{path}.score")
    if not 1 <= score <= 5:
        raise ValidationError(f"must be between 1 and 5, got {score}", f"{path}.score")
    feedback = data.get("feedback")
    if feedback is None or (isinstance(feedback, str) and not feedback.strip()):
        raise ValidationError(
            "written feedback explaining the score is required", f"{path}.feedback"
        )
    try:
        return CriterionScore(
            criterion=criterion,
            score=score,
            source=ScoreSource.HUMAN,
            feedback=_string(feedback, f"{path}.feedback").strip(),
            evidence=_string_list(data.get("evidence"), f"{path}.evidence"),
            annotator=annotator,
        )
    except ValidationError as exc:
        raise _wrap(path, exc) from None


def parse_judgment(
    data: Any, path: str, rubric: Rubric, default_annotator: str | None
) -> HumanJudgment:
    data = _require_mapping(data, path)
    _reject_unknown(data, _JUDGMENT_FIELDS, path)
    item_id = data.get("item_id")
    if isinstance(item_id, int) and not isinstance(item_id, bool):
        item_id = str(item_id)
    item_id = _string(item_id, f"{path}.item_id")
    annotator = _optional_string(data.get("annotator"), f"{path}.annotator") or default_annotator
    if not annotator:
        raise ValidationError(
            "an annotator is required (set it here or at the top level)", f"{path}.annotator"
        )
    raw_scores = _require_mapping(data.get("scores"), f"{path}.scores")
    if not raw_scores:
        raise ValidationError("must contain at least one criterion score", f"{path}.scores")
    unknown = sorted(set(raw_scores) - set(rubric.keys))
    if unknown:
        raise ValidationError(
            f"unknown criteria {unknown}; expected some of {list(rubric.keys)}", f"{path}.scores"
        )
    scores = {
        key: parse_human_score(value, key, annotator, f"{path}.scores.{key}")
        for key, value in raw_scores.items()
    }
    return HumanJudgment(
        item_id=item_id,
        annotator=annotator,
        scores=scores,
        comment=_optional_string(data.get("comment"), f"{path}.comment"),
    )


def parse_judgments(
    data: Any, rubric: Rubric | None = None, dataset: Dataset | None = None
) -> JudgmentSet:
    """Validate a human-judgments document.

    Args:
        data: Decoded JSON, ``{"annotator": ..., "judgments": [...]}``.
        rubric: Rubric the criterion keys must belong to.
        dataset: If given, every ``item_id`` must exist in it.
    """
    rubric = rubric or Rubric()
    data = _require_mapping(data, "judgments_file")
    _reject_unknown(data, {"annotator", "description", "judgments"}, "judgments_file")
    default_annotator = _optional_string(data.get("annotator"), "annotator")
    raw = data.get("judgments")
    if not isinstance(raw, list):
        raise ValidationError("must be a list", "judgments")

    judgments: dict[str, HumanJudgment] = {}
    known_ids = set(dataset.ids) if dataset is not None else None
    for i, entry in enumerate(raw):
        path = f"judgments[{i}]"
        judgment = parse_judgment(entry, path, rubric, default_annotator)
        if judgment.item_id in judgments:
            raise ValidationError(
                f"duplicate judgment for item {judgment.item_id!r}", f"{path}.item_id"
            )
        if known_ids is not None and judgment.item_id not in known_ids:
            raise ValidationError(
                f"item {judgment.item_id!r} does not exist in the dataset", f"{path}.item_id"
            )
        judgments[judgment.item_id] = judgment

    return JudgmentSet(
        judgments=judgments,
        description=_string(
            data.get("description"), "description", required=False, allow_empty=True
        ),
    )
