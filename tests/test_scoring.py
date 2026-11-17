import pytest

from llm_eval.automated import AutomatedScorer
from llm_eval.models import (
    Constraints,
    CriterionResult,
    CriterionScore,
    Dataset,
    HumanJudgment,
    JudgmentSet,
    ScoreSource,
)
from llm_eval.rubric import Rubric
from llm_eval.scoring import (
    agreement_statistics,
    compute_overall,
    evaluate_dataset,
    evaluate_item,
    weighted_mean,
)
from tests.conftest import make_item


def human(criterion, score, feedback="because"):
    return CriterionScore(criterion, score, ScoreSource.HUMAN, feedback, annotator="r1")


def auto(criterion, score):
    return CriterionScore(criterion, score, ScoreSource.AUTOMATED, "heuristic")


def full_judgment(item_id="item", scores=(5, 5, 5, 5)):
    keys = Rubric().keys
    return HumanJudgment(item_id, "r1", {k: human(k, s) for k, s in zip(keys, scores, strict=True)})


EQUAL = Rubric().with_weights({k: 1 for k in Rubric().keys})


def test_weighted_mean_equal_weights():
    assert (
        weighted_mean(
            {"accuracy": 5, "relevance": 3, "clarity": 4, "instruction_following": 2}, EQUAL
        )
        == 3.5
    )


def test_weighted_mean_uses_weights():
    rubric = Rubric().with_weights(
        {"accuracy": 3, "relevance": 1, "clarity": 1, "instruction_following": 1}
    )
    # (3*5 + 1*1 + 1*1 + 1*1) / 6 = 3.0
    assert (
        weighted_mean(
            {"accuracy": 5, "relevance": 1, "clarity": 1, "instruction_following": 1}, rubric
        )
        == 3.0
    )


def test_weighted_mean_renormalises_over_present_criteria():
    assert weighted_mean({"accuracy": 4, "clarity": 2}, EQUAL) == 3.0
    assert weighted_mean({}, EQUAL) is None


def test_human_score_takes_precedence():
    r = CriterionResult("accuracy", human=human("accuracy", 2), automated=auto("accuracy", 5))
    assert r.final.score == 2
    assert r.final.source is ScoreSource.HUMAN
    assert r.difference == -3
    assert r.needs_review


def test_automated_score_fills_gap():
    r = CriterionResult("clarity", automated=auto("clarity", 4))
    assert r.final.source is ScoreSource.AUTOMATED
    assert r.difference is None
    assert not r.needs_review


def test_small_disagreement_is_not_flagged():
    r = CriterionResult("clarity", human=human("clarity", 4), automated=auto("clarity", 5))
    assert not r.needs_review


def test_overall_marks_missing_criteria():
    criteria = {k: CriterionResult(k) for k in EQUAL.keys}
    criteria["clarity"] = CriterionResult("clarity", automated=auto("clarity", 4))
    criteria["relevance"] = CriterionResult("relevance", human=human("relevance", 2))
    overall = compute_overall(criteria, EQUAL)
    assert overall.score == 3.0
    assert not overall.complete
    assert set(overall.missing) == {"accuracy", "instruction_following"}
    assert overall.sources == {"human": 1, "automated": 1}


def test_overall_none_when_nothing_scored():
    overall = compute_overall({k: CriterionResult(k) for k in EQUAL.keys}, EQUAL)
    assert overall.score is None


def test_evaluate_item_with_human_and_automated():
    item = make_item(
        "Photosynthesis converts light energy into chemical energy.",
        key_points=("converts light energy into chemical energy",),
        constraints=Constraints(max_words=20),
    )
    result = evaluate_item(item, EQUAL, full_judgment(scores=(4, 4, 4, 4)), AutomatedScorer(EQUAL))
    assert result.overall.score == 4.0
    assert result.overall.complete
    assert result.human_overall == 4.0
    assert result.automated_overall is not None
    assert result.overall.sources["human"] == 4
    assert result.annotator == "r1"


def test_evaluate_item_without_scorer_is_human_only():
    result = evaluate_item(make_item("x"), EQUAL, full_judgment(), scorer=None)
    assert all(c.automated is None for c in result.criteria.values())
    assert result.automated_overall is None


def test_automated_overall_requires_every_criterion():
    # No key points or constraints -> accuracy and instruction following are unscorable.
    result = evaluate_item(
        make_item("Photosynthesis uses light."), EQUAL, None, AutomatedScorer(EQUAL)
    )
    assert result.automated_overall is None
    assert not result.overall.complete


def test_evaluate_item_rejects_mismatched_judgment():
    with pytest.raises(ValueError):
        evaluate_item(make_item("x", item_id="a"), EQUAL, full_judgment(item_id="b"))


def test_agreement_statistics():
    r1 = evaluate_item(make_item("x", item_id="a"), EQUAL, full_judgment("a"), None)
    # Inject automated scores manually via CriterionResult to control the differences.
    criteria = {
        "accuracy": CriterionResult("accuracy", human("accuracy", 5), auto("accuracy", 5)),
        "relevance": CriterionResult("relevance", human("relevance", 4), auto("relevance", 5)),
        "clarity": CriterionResult("clarity", human("clarity", 1), auto("clarity", 4)),
        "instruction_following": CriterionResult("instruction_following"),
    }
    r2 = type(r1)(
        item=r1.item,
        criteria=criteria,
        overall=r1.overall,
        human_overall=None,
        automated_overall=None,
    )
    stats = agreement_statistics([r1, r2], EQUAL)
    pooled = stats["pooled"]
    assert pooled["pairs"] == 3
    assert pooled["exact_agreement"] == pytest.approx(1 / 3, abs=0.01)
    assert pooled["within_one"] == pytest.approx(2 / 3, abs=0.01)
    assert pooled["mean_absolute_difference"] == pytest.approx(4 / 3, abs=0.01)
    assert pooled["mean_signed_difference"] == pytest.approx(-4 / 3, abs=0.01)
    assert stats["per_criterion"]["instruction_following"]["pairs"] == 0
    assert r2.needs_review


def test_evaluate_dataset_summary():
    items = (
        make_item("Photosynthesis uses light.", item_id="a"),
        make_item("Totally unrelated words.", item_id="b"),
    )
    dataset = Dataset(items=items, name="t")
    judgments = JudgmentSet({"a": full_judgment("a", (5, 5, 5, 5))})
    run = evaluate_dataset(dataset, EQUAL, judgments)
    s = run.summary
    assert s["items_evaluated"] == 2
    assert s["items_with_human_judgment"] == 1
    assert s["items_complete"] == 1
    assert s["criteria"]["accuracy"]["scored_items"] == 1
    assert "unspecified" in s["by_model"]


def test_evaluate_dataset_without_automated():
    dataset = Dataset(items=(make_item("x", item_id="a"),))
    run = evaluate_dataset(dataset, EQUAL, None, use_automated=False)
    assert run.results[0].overall.score is None
    assert not run.automated_enabled
