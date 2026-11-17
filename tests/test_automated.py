import pytest

from llm_eval.automated import (
    AutomatedScorer,
    check_constraints,
    fraction_to_score,
    score_accuracy,
    score_clarity,
    score_instruction_following,
    score_relevance,
)
from llm_eval.models import Constraints, ScoreSource
from llm_eval.rubric import Rubric
from tests.conftest import make_item

KEY_POINTS = (
    "Converts light energy into chemical energy",
    "Uses carbon dioxide and water",
    "Produces glucose and releases oxygen",
    "Takes place in chloroplasts",
)


@pytest.mark.parametrize(
    "fraction, score",
    [(0, 1), (0.25, 2), (0.5, 3), (0.75, 4), (1, 5), (-1, 1), (2, 5), (0.124, 1), (0.125, 2)],
)
def test_fraction_to_score(fraction, score):
    assert fraction_to_score(fraction) == score


# ----------------------------------------------------------------------- accuracy


def test_accuracy_is_not_scored_without_reference_evidence():
    assert score_accuracy(make_item("Anything at all.")) is None


def test_accuracy_full_key_point_coverage():
    response = (
        "Photosynthesis takes place in chloroplasts. Plants convert light energy into "
        "chemical energy, using carbon dioxide and water to produce glucose and release "
        "oxygen."
    )
    result = score_accuracy(make_item(response, key_points=KEY_POINTS))
    assert result.score == 5
    assert result.source is ScoreSource.AUTOMATED
    assert result.signals["key_points_matched"] == 4


def test_accuracy_partial_coverage_lists_missing_points():
    result = score_accuracy(
        make_item("It produces glucose and releases oxygen.", key_points=KEY_POINTS)
    )
    assert result.score == 2
    assert "Uses carbon dioxide and water" in result.feedback
    assert "does not detect contradictions" in result.feedback


def test_accuracy_known_limitation_negation_is_not_detected():
    """Documents a deliberate limitation: lexical overlap cannot see negation.

    The response asserts the opposite of the reference point, yet matches it.
    This is why human accuracy judgments always take precedence.
    """
    item = make_item(
        "Seasons are caused by distance from the Sun.",
        key_points=("Seasons are not caused by distance from the Sun",),
    )
    assert score_accuracy(item).score == 5


# ---------------------------------------------------------------------- relevance


def test_relevance_on_topic_vs_off_topic():
    prompt = "How do I reverse a list in Python?"
    on_topic = make_item("In Python, call my_list.reverse() to reverse it.", prompt=prompt)
    off_topic = make_item("Bananas are a good source of potassium.", prompt=prompt)
    assert score_relevance(on_topic).score == 5
    assert score_relevance(off_topic).score == 1
    assert "python" in score_relevance(off_topic).signals["missing_terms"]


def test_relevance_ignores_instruction_words_and_numbers():
    item = make_item(
        "Photosynthesis turns light into sugar.",
        prompt="Explain photosynthesis in no more than 80 words.",
    )
    assert score_relevance(item).signals["prompt_terms"] == ["photosy"]


def test_relevance_empty_response_scores_one():
    assert score_relevance(make_item("")).score == 1


def test_relevance_prompt_without_topical_terms_is_neutral():
    assert score_relevance(make_item("Hello!", prompt="Please explain.")).score == 3


# ------------------------------------------------------------------------ clarity


def test_clarity_clean_text_scores_five():
    item = make_item("Photosynthesis makes food for plants. It uses light. It releases oxygen.")
    result = score_clarity(item)
    assert result.score == 5
    assert result.signals["sentence_count"] == 3


def test_clarity_penalises_run_on_repetitive_text():
    run_on = " and ".join(["the gadget is good because the gadget helps"] * 12) + "."
    result = score_clarity(make_item(run_on))
    assert result.score <= 2
    assert "long sentences" in result.feedback


def test_clarity_penalises_duplicate_sentences():
    item = make_item("Water is wet. Water is wet. The sky is blue.")
    result = score_clarity(item)
    assert result.signals["duplicate_sentences"] == 1
    assert result.score == 4


def test_clarity_empty_response():
    assert score_clarity(make_item("   ")).score == 1


# ---------------------------------------------------------- instruction following


def test_instruction_following_not_scored_without_constraints():
    assert score_instruction_following(make_item("text")) is None


def test_word_limits():
    c = Constraints(max_words=3, min_words=2)
    assert all(ch["passed"] for ch in check_constraints(c, "one two three"))
    assert not all(ch["passed"] for ch in check_constraints(c, "one two three four"))
    assert not all(ch["passed"] for ch in check_constraints(c, "one"))


def test_sentence_limit():
    c = Constraints(max_sentences=2)
    assert check_constraints(c, "One. Two.")[0]["passed"]
    assert not check_constraints(c, "One. Two. Three.")[0]["passed"]


def test_required_and_forbidden_phrases():
    c = Constraints(required_keywords=("capital",), forbidden_phrases=("let me know",))
    checks = check_constraints(c, "The capital is Paris. Let me know if you need more.")
    assert [ch["passed"] for ch in checks] == [True, False]


@pytest.mark.parametrize(
    "fmt, good, bad",
    [
        ("bullet_list", "- a\n- b", "a, b"),
        ("numbered_list", "1. a\n2. b", "- a\n- b"),
        ("json", '{"a": 1}', 'Here: {"a": 1}'),
        ("paragraph", "Just prose.", "- a\n- b"),
    ],
)
def test_format_checks(fmt, good, bad):
    c = Constraints(response_format=fmt)
    assert check_constraints(c, good)[0]["passed"]
    assert not check_constraints(c, bad)[0]["passed"]


def test_item_count():
    c = Constraints(response_format="bullet_list", item_count=3)
    assert all(ch["passed"] for ch in check_constraints(c, "- a\n- b\n- c"))
    assert not all(ch["passed"] for ch in check_constraints(c, "- a\n- b"))


def test_instruction_following_score_reflects_fraction_passed():
    c = Constraints(
        response_format="json", required_keywords=("name", "capital"), forbidden_phrases=("sure",)
    )
    item = make_item('Sure! {"name": "France", "capital": "Paris"}', constraints=c)
    result = score_instruction_following(item)
    assert result.signals["checks_passed"] == 2
    assert result.signals["checks_total"] == 4
    assert result.score == 3
    assert "Failed" in result.feedback


# ------------------------------------------------------------------------- scorer


def test_automated_scorer_only_returns_scorable_criteria():
    scores = AutomatedScorer().score(make_item("Photosynthesis uses light."))
    assert set(scores) == {"relevance", "clarity"}


def test_automated_scorer_accepts_custom_scorers():
    from llm_eval.models import CriterionScore

    def always_three(item):
        return CriterionScore("clarity", 3, ScoreSource.AUTOMATED, "custom")

    scores = AutomatedScorer(Rubric(), {"clarity": always_three}).score(make_item("x"))
    assert scores["clarity"].feedback == "custom"
