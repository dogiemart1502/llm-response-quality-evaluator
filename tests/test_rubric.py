import math

import pytest

from llm_eval.errors import ValidationError
from llm_eval.rubric import DEFAULT_CRITERIA, Criterion, Rubric, parse_weights


def test_default_rubric_has_the_four_required_criteria():
    assert Rubric().keys == ("accuracy", "relevance", "clarity", "instruction_following")


def test_default_weights_are_normalised():
    rubric = Rubric()
    assert math.isclose(sum(rubric.weights.values()), 1.0)
    assert rubric.weight("accuracy") > rubric.weight("clarity")


@pytest.mark.parametrize("criterion", DEFAULT_CRITERIA, ids=lambda c: c.key)
def test_every_criterion_describes_each_level(criterion):
    for score in range(1, 6):
        assert criterion.describe(score)
    assert criterion.automated_proxy


def test_describe_rejects_out_of_range_score():
    with pytest.raises(ValidationError):
        DEFAULT_CRITERIA[0].describe(6)


def test_custom_weights_are_normalised():
    rubric = Rubric().with_weights(
        {"accuracy": 2, "relevance": 1, "clarity": 1, "instruction_following": 0.0001}
    )
    assert math.isclose(sum(rubric.weights.values()), 1.0)
    assert math.isclose(rubric.weight("accuracy"), 2 * rubric.weight("relevance"))


@pytest.mark.parametrize(
    "weights, message",
    [
        ({"nonsense": 1}, "unknown criteria"),
        ({"accuracy": 0}, "positive"),
        ({"accuracy": -1}, "positive"),
        ({"accuracy": float("nan")}, "positive"),
        ({"accuracy": "high"}, "number"),
        ({"accuracy": True}, "number"),
    ],
)
def test_invalid_weights_are_rejected(weights, message):
    with pytest.raises(ValidationError, match=message):
        Rubric().with_weights(weights)


def test_unknown_criterion_lookup_raises():
    with pytest.raises(ValidationError, match="unknown criterion"):
        Rubric().get("fluency")


def test_criterion_requires_all_levels():
    with pytest.raises(ValidationError, match="missing level"):
        Criterion("x", "X", "d", {1: "a", 2: "b"}, "proxy")


def test_rubric_rejects_duplicate_keys():
    with pytest.raises(ValidationError, match="duplicate"):
        Rubric(criteria=(DEFAULT_CRITERIA[0], DEFAULT_CRITERIA[0]))


def test_rubric_to_dict_contains_scale_and_weights():
    data = Rubric().to_dict()
    assert data["scale"] == {"min": 1, "max": 5}
    assert {c["key"] for c in data["criteria"]} == set(Rubric().keys)
    assert all("weight" in c and "levels" in c for c in data["criteria"])


def test_parse_weights():
    assert parse_weights("accuracy=0.5, clarity=1") == {"accuracy": 0.5, "clarity": 1.0}


@pytest.mark.parametrize("spec", ["accuracy", "accuracy=abc", "=1", "", " , "])
def test_parse_weights_rejects_malformed_input(spec):
    with pytest.raises(ValidationError):
        parse_weights(spec)
