"""Combine human and automated scores into per-item and dataset-level results.

Scoring policy (also written into every exported file):

1. For each criterion, a human score takes precedence over an automated one.
   The automated score is still reported alongside it for comparison.
2. If neither source scored a criterion, it is reported as *missing* and the
   overall score is computed over the remaining criteria with re-normalised
   weights; the result is marked ``complete: false``.
3. ``human_overall`` and ``automated_overall`` are reported separately and are
   only computed when that source scored *every* criterion, so they are always
   like-for-like.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

from llm_eval.automated import AutomatedScorer
from llm_eval.models import (
    CriterionResult,
    CriterionScore,
    Dataset,
    EvaluationItem,
    EvaluationRun,
    HumanJudgment,
    ItemResult,
    JudgmentSet,
    OverallScore,
    ScoreSource,
)
from llm_eval.rubric import Rubric

SCORING_POLICY = (
    "Human scores take precedence over automated scores for each criterion. Automated scores "
    "fill gaps only where no human score exists and are reported separately for comparison. "
    "Overall = weighted mean of available final criterion scores (weights re-normalised over "
    "scored criteria); 'complete' is false when any criterion has no score."
)


def _round(value: float) -> float:
    return round(value + 0.0, 2)


def weighted_mean(scores: Mapping[str, int], rubric: Rubric) -> float | None:
    """Weighted mean of ``scores`` with weights re-normalised over present keys."""
    if not scores:
        return None
    total_weight = sum(rubric.weight(k) for k in scores)
    return _round(sum(rubric.weight(k) * v for k, v in scores.items()) / total_weight)


def compute_overall(criteria: Mapping[str, CriterionResult], rubric: Rubric) -> OverallScore:
    finals: dict[str, CriterionScore] = {
        k: r.final for k, r in criteria.items() if r.final is not None
    }
    sources = {s.value: 0 for s in ScoreSource}
    for score in finals.values():
        sources[score.source.value] += 1
    return OverallScore(
        score=weighted_mean({k: s.score for k, s in finals.items()}, rubric),
        criteria_used=tuple(k for k in rubric.keys if k in finals),
        missing=tuple(k for k in rubric.keys if k not in finals),
        sources=sources,
    )


def _source_overall(
    criteria: Mapping[str, CriterionResult], rubric: Rubric, source: ScoreSource
) -> float | None:
    """Overall from a single source, only if it covered every criterion."""
    picked: dict[str, int] = {}
    for key in rubric.keys:
        result = criteria[key]
        score = result.human if source is ScoreSource.HUMAN else result.automated
        if score is None:
            return None
        picked[key] = score.score
    return weighted_mean(picked, rubric)


def evaluate_item(
    item: EvaluationItem,
    rubric: Rubric | None = None,
    judgment: HumanJudgment | None = None,
    scorer: AutomatedScorer | None = None,
) -> ItemResult:
    """Evaluate a single item.

    Args:
        item: The prompt/response pair.
        rubric: Rubric to use (default rubric if omitted).
        judgment: Optional human judgment for this item.
        scorer: Automated scorer; pass ``None`` to disable automated scoring.
    """
    rubric = rubric or Rubric()
    if judgment is not None and judgment.item_id != item.id:
        raise ValueError(f"judgment is for item {judgment.item_id!r}, not {item.id!r}")
    automated = scorer.score(item) if scorer is not None else {}
    human = dict(judgment.scores) if judgment is not None else {}

    criteria = {
        key: CriterionResult(criterion=key, human=human.get(key), automated=automated.get(key))
        for key in rubric.keys
    }
    return ItemResult(
        item=item,
        criteria=criteria,
        overall=compute_overall(criteria, rubric),
        human_overall=_source_overall(criteria, rubric, ScoreSource.HUMAN),
        automated_overall=_source_overall(criteria, rubric, ScoreSource.AUTOMATED),
        annotator=judgment.annotator if judgment else None,
        comment=judgment.comment if judgment else None,
    )


def agreement_statistics(results: Sequence[ItemResult], rubric: Rubric) -> dict[str, Any]:
    """Compare human and automated scores where both exist.

    Reports, per criterion and pooled: number of paired scores, exact-agreement
    rate, within-one-point agreement rate, mean absolute difference, and mean
    signed difference (human minus automated; positive = heuristics too harsh).
    """

    def stats(diffs: list[int]) -> dict[str, Any]:
        if not diffs:
            return {
                "pairs": 0,
                "exact_agreement": None,
                "within_one": None,
                "mean_absolute_difference": None,
                "mean_signed_difference": None,
            }
        return {
            "pairs": len(diffs),
            "exact_agreement": _round(sum(d == 0 for d in diffs) / len(diffs)),
            "within_one": _round(sum(abs(d) <= 1 for d in diffs) / len(diffs)),
            "mean_absolute_difference": _round(mean(abs(d) for d in diffs)),
            "mean_signed_difference": _round(mean(diffs)),
        }

    per_criterion: dict[str, list[int]] = {k: [] for k in rubric.keys}
    for result in results:
        for key, crit in result.criteria.items():
            if crit.difference is not None:
                per_criterion[key].append(crit.difference)
    pooled = [d for diffs in per_criterion.values() for d in diffs]
    return {
        "per_criterion": {k: stats(v) for k, v in per_criterion.items()},
        "pooled": stats(pooled),
    }


def summarize(results: Sequence[ItemResult], rubric: Rubric) -> dict[str, Any]:
    """Dataset-level summary statistics."""
    criterion_means: dict[str, dict[str, Any]] = {}
    for key in rubric.keys:
        finals = [r.criteria[key].final for r in results if r.criteria[key].final is not None]
        humans = [r.criteria[key].human.score for r in results if r.criteria[key].human]
        autos = [r.criteria[key].automated.score for r in results if r.criteria[key].automated]
        criterion_means[key] = {
            "mean_final": _round(mean(s.score for s in finals)) if finals else None,
            "mean_human": _round(mean(humans)) if humans else None,
            "mean_automated": _round(mean(autos)) if autos else None,
            "scored_items": len(finals),
        }

    by_model: dict[str, list[float]] = defaultdict(list)
    for r in results:
        if r.overall.score is not None:
            by_model[r.item.model or "unspecified"].append(r.overall.score)

    overall_scores = [r.overall.score for r in results if r.overall.score is not None]
    return {
        "items_evaluated": len(results),
        "items_with_human_judgment": sum(r.annotator is not None for r in results),
        "items_complete": sum(r.overall.complete for r in results),
        "mean_overall": _round(mean(overall_scores)) if overall_scores else None,
        "criteria": criterion_means,
        "by_model": {
            m: {"items": len(v), "mean_overall": _round(mean(v))}
            for m, v in sorted(by_model.items())
        },
        "human_vs_automated_agreement": agreement_statistics(results, rubric),
        "items_needing_review": [r.item.id for r in results if r.needs_review],
    }


def evaluate_dataset(
    dataset: Dataset,
    rubric: Rubric | None = None,
    judgments: JudgmentSet | None = None,
    *,
    use_automated: bool = True,
    scorer: AutomatedScorer | None = None,
) -> EvaluationRun:
    """Evaluate every item in ``dataset``.

    Args:
        dataset: Items to evaluate.
        rubric: Rubric (default rubric if omitted).
        judgments: Optional human judgments.
        use_automated: Whether to compute automated heuristic scores.
        scorer: Custom automated scorer (ignored when ``use_automated`` is False).
    """
    rubric = rubric or Rubric()
    active_scorer = (scorer or AutomatedScorer(rubric)) if use_automated else None
    results = tuple(
        evaluate_item(
            item,
            rubric,
            judgments.get(item.id) if judgments is not None else None,
            active_scorer,
        )
        for item in dataset.items
    )
    return EvaluationRun(
        dataset=dataset,
        rubric=rubric,
        results=results,
        automated_enabled=use_automated,
        summary=summarize(results, rubric),
    )
