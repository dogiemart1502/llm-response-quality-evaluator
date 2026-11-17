"""End-to-end checks on the bundled sample data.

These pin down the behaviours the sample data is meant to demonstrate, so the
README's claims stay true as the code evolves.
"""

import json

import pytest

from llm_eval.io import load_dataset, load_judgments
from llm_eval.scoring import evaluate_dataset
from tests.conftest import ROOT


@pytest.fixture(scope="module")
def run():
    dataset = load_dataset(ROOT / "data" / "sample_responses.json")
    judgments = load_judgments(ROOT / "data" / "human_judgments.json", dataset=dataset)
    return evaluate_dataset(dataset, judgments=judgments)


def result(run, item_id):
    return next(r for r in run.results if r.item.id == item_id)


def test_all_items_evaluated(run):
    assert len(run.results) == 10
    assert run.summary["items_with_human_judgment"] == 9


def test_every_human_score_has_feedback(run):
    for r in run.results:
        for crit in r.criteria.values():
            if crit.human is not None:
                assert crit.human.feedback.strip()


def test_factual_error_missed_by_heuristic_is_flagged(run):
    """photosynthesis-b swaps O2 and CO2: heuristics miss it, the human catches it."""
    acc = result(run, "photosynthesis-b").criteria["accuracy"]
    assert acc.automated.score >= 4
    assert acc.human.score <= 2
    assert acc.final.source.value == "human"
    assert acc.needs_review


def test_negation_misconception_is_flagged(run):
    acc = result(run, "seasons-b").criteria["accuracy"]
    assert acc.human.score == 1
    assert acc.needs_review


def test_unjudged_item_has_no_accuracy_score_without_evidence(run):
    r = result(run, "python-reverse-a")
    assert r.criteria["accuracy"].final is None
    assert "accuracy" in r.overall.missing
    assert not r.overall.complete


def test_instruction_violations_detected_automatically(run):
    assert result(run, "exercise-b").criteria["instruction_following"].automated.score == 1
    assert result(run, "json-capital-b").criteria["instruction_following"].automated.score < 5


def test_better_responses_outscore_worse_ones(run):
    pairs = [
        ("photosynthesis-a", "photosynthesis-b"),
        ("exercise-a", "exercise-b"),
        ("seasons-a", "seasons-b"),
        ("json-capital-a", "json-capital-b"),
    ]
    for good, bad in pairs:
        assert result(run, good).overall.score > result(run, bad).overall.score


def test_example_results_file_is_in_sync(run):
    """examples/sample_results.json should match a fresh run (ignoring the timestamp)."""
    from llm_eval.io import run_to_dict

    committed = json.loads((ROOT / "examples" / "sample_results.json").read_text("utf-8"))
    fresh = json.loads(json.dumps(run_to_dict(run)))
    committed.pop("generated_at")
    fresh.pop("generated_at")
    assert committed == fresh, "regenerate with: llm-eval evaluate ... -o examples/..."
