import copy

import pytest

from llm_eval.errors import ValidationError
from llm_eval.models import Constraints, CriterionScore, ScoreSource
from llm_eval.validation import parse_dataset, parse_judgments


def mutate(base, fn):
    data = copy.deepcopy(base)
    fn(data)
    return data


# ------------------------------------------------------------------------ dataset


def test_valid_dataset_parses(minimal_dataset):
    ds = parse_dataset(minimal_dataset)
    assert ds.name == "mini"
    assert ds.items[0].constraints.max_words == 50
    assert ds.items[0].reference_key_points == ("converts light energy into chemical energy",)


def test_numeric_ids_are_accepted_as_strings(minimal_dataset):
    ds = parse_dataset(mutate(minimal_dataset, lambda d: d["items"][0].update(id=7)))
    assert ds.items[0].id == "7"


def test_empty_response_is_allowed(minimal_dataset):
    ds = parse_dataset(mutate(minimal_dataset, lambda d: d["items"][0].update(response="")))
    assert ds.items[0].response == ""


@pytest.mark.parametrize(
    "change, path_fragment, message",
    [
        (lambda d: d.update(items=[]), "items", "at least one"),
        (lambda d: d.update(items="nope"), "items", "list"),
        (lambda d: d["items"][0].pop("prompt"), "items[0].prompt", "required"),
        (lambda d: d["items"][0].update(prompt="  "), "items[0].prompt", "empty"),
        (lambda d: d["items"][0].update(response=None), "items[0].response", "required"),
        (lambda d: d["items"][0].update(response=42), "items[0].response", "string"),
        (lambda d: d["items"][0].update(extra=1), "items[0]", "unknown field"),
        (lambda d: d["items"][0].update(reference_key_points="x"), "reference_key_points", "list"),
        (lambda d: d["items"][0]["constraints"].update(max_words=0), "max_words", "positive"),
        (lambda d: d["items"][0]["constraints"].update(max_words=True), "max_words", "positive"),
        (
            lambda d: d["items"][0]["constraints"].update(format="xml"),
            "constraints",
            "unsupported format",
        ),
        (lambda d: d["items"][0]["constraints"].update(min_words=100), "constraints", "min_words"),
        (lambda d: d["items"][0]["constraints"].update(item_count=3), "constraints", "item_count"),
        (lambda d: d["items"].append(copy.deepcopy(d["items"][0])), "items[1].id", "duplicate"),
    ],
)
def test_invalid_datasets_are_rejected_with_location(
    minimal_dataset, change, path_fragment, message
):
    with pytest.raises(ValidationError) as exc_info:
        parse_dataset(mutate(minimal_dataset, change))
    assert path_fragment in str(exc_info.value)
    assert message in str(exc_info.value)


def test_dataset_must_be_an_object():
    with pytest.raises(ValidationError, match="must be an object"):
        parse_dataset([1, 2])


def test_overlong_text_is_rejected(minimal_dataset):
    data = mutate(minimal_dataset, lambda d: d["items"][0].update(response="x" * 100_001))
    with pytest.raises(ValidationError, match="too long"):
        parse_dataset(data)


# ---------------------------------------------------------------------- judgments


def test_valid_judgments_parse(minimal_dataset, minimal_judgments):
    ds = parse_dataset(minimal_dataset)
    js = parse_judgments(minimal_judgments, dataset=ds)
    j = js.get("q1")
    assert j.annotator == "tester"
    assert j.scores["accuracy"].source is ScoreSource.HUMAN
    assert j.scores["accuracy"].evidence == ("textbook",)
    assert len(js) == 1


def test_integral_float_score_is_accepted(minimal_judgments):
    data = mutate(
        minimal_judgments, lambda d: d["judgments"][0]["scores"]["clarity"].update(score=4.0)
    )
    assert parse_judgments(data).get("q1").scores["clarity"].score == 4


def test_per_judgment_annotator_overrides_default(minimal_judgments):
    data = mutate(minimal_judgments, lambda d: d["judgments"][0].update(annotator="alice"))
    assert parse_judgments(data).get("q1").scores["relevance"].annotator == "alice"


def _score(d):
    return d["judgments"][0]["scores"]


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda d: _score(d)["accuracy"].update(score=6), "between 1 and 5"),
        (lambda d: _score(d)["accuracy"].update(score=0), "between 1 and 5"),
        (lambda d: _score(d)["accuracy"].update(score=3.5), "integer"),
        (lambda d: _score(d)["accuracy"].update(score="5"), "integer"),
        (lambda d: _score(d)["accuracy"].update(score=True), "integer"),
        (lambda d: _score(d)["accuracy"].pop("score"), "required"),
        (lambda d: _score(d)["accuracy"].pop("feedback"), "feedback"),
        (lambda d: _score(d)["accuracy"].update(feedback="   "), "feedback"),
        (lambda d: _score(d)["accuracy"].update(evidence="x"), "list"),
        (lambda d: _score(d).update(fluency={"score": 3, "feedback": "x"}), "unknown criteria"),
        (lambda d: d["judgments"][0].update(scores={}), "at least one"),
        (lambda d: d.pop("annotator"), "annotator"),
        (lambda d: d["judgments"].append(copy.deepcopy(d["judgments"][0])), "duplicate"),
        (lambda d: d.update(judgments={}), "list"),
    ],
)
def test_invalid_judgments_are_rejected(minimal_judgments, change, message):
    with pytest.raises(ValidationError, match=message):
        parse_judgments(mutate(minimal_judgments, change))


def test_judgment_for_unknown_item_is_rejected(minimal_dataset, minimal_judgments):
    data = mutate(minimal_judgments, lambda d: d["judgments"][0].update(item_id="missing"))
    with pytest.raises(ValidationError, match="does not exist"):
        parse_judgments(data, dataset=parse_dataset(minimal_dataset))


# ------------------------------------------------------------- model invariants


def test_criterion_score_invariants():
    with pytest.raises(ValidationError):
        CriterionScore("accuracy", 7, ScoreSource.HUMAN, "x")
    with pytest.raises(ValidationError):
        CriterionScore("accuracy", 3, ScoreSource.HUMAN, "")
    assert CriterionScore("accuracy", 3, "human", "ok").source is ScoreSource.HUMAN


def test_constraints_count():
    c = Constraints(max_words=10, required_keywords=("a", "b"), response_format="json")
    assert c.count() == 4
    assert Constraints().is_empty()
