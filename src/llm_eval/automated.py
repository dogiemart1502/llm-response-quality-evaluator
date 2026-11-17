"""Transparent, deterministic automated scoring heuristics.

These heuristics produce *suggested* scores to help a human reviewer triage a
batch of responses. Every automated score records the raw measurements it was
derived from (``signals``) and a written explanation (``feedback``).

Important limitations (also surfaced in output and the README):

* **Accuracy** is only estimated when the dataset supplies reference key points
  written by a human. The score reflects lexical agreement with those points;
  it cannot verify facts, and it is fooled by negation ("is *not* caused by
  distance" matches "is caused by distance").
* **Relevance** uses word overlap with the prompt, so paraphrases are missed.
* **Clarity** uses readability statistics, which say nothing about logic.
* **Instruction following** only checks constraints declared in the dataset.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable

from llm_eval.models import Constraints, CriterionScore, EvaluationItem, ScoreSource
from llm_eval.rubric import SCORE_MAX, SCORE_MIN, Rubric
from llm_eval.text_utils import (
    INSTRUCTION_WORDS,
    bullet_items,
    contains_phrase,
    content_terms,
    flesch_reading_ease,
    numbered_items,
    split_sentences,
    tokenize,
    unique_terms,
)

#: Fraction of a key point's terms that must appear for it to count as covered.
KEY_POINT_MATCH_THRESHOLD = 0.6

AUTOMATED_DISCLAIMER = (
    "Automated scores are heuristic signals, not verified judgments. The accuracy signal "
    "measures lexical agreement with human-supplied reference key points and does not verify "
    "facts independently."
)


def fraction_to_score(fraction: float) -> int:
    """Map a 0-1 fraction linearly onto the 1-5 scale (round half up).

    0 -> 1, 0.25 -> 2, 0.5 -> 3, 0.75 -> 4, 1 -> 5.
    """
    fraction = min(max(fraction, 0.0), 1.0)
    return int(SCORE_MIN + (SCORE_MAX - SCORE_MIN) * fraction + 0.5)


def _clamp(score: int) -> int:
    return max(SCORE_MIN, min(SCORE_MAX, score))


def _auto(criterion: str, score: int, feedback: str, **signals: object) -> CriterionScore:
    return CriterionScore(
        criterion=criterion,
        score=score,
        source=ScoreSource.AUTOMATED,
        feedback=feedback,
        signals=signals,
    )


# --------------------------------------------------------------------------- accuracy


def score_accuracy(item: EvaluationItem) -> CriterionScore | None:
    """Estimate agreement with the item's reference key points.

    Returns ``None`` when no reference key points are available, because without
    supporting evidence there is no basis for an accuracy estimate.
    """
    if not item.reference_key_points:
        return None

    response_terms = set(content_terms(item.response))
    matched: list[str] = []
    missing: list[str] = []
    per_point: list[dict[str, object]] = []
    for point in item.reference_key_points:
        terms = unique_terms(point)
        if not terms:
            continue
        found = [t for t in terms if t in response_terms]
        coverage = len(found) / len(terms)
        is_match = coverage >= KEY_POINT_MATCH_THRESHOLD
        (matched if is_match else missing).append(point)
        per_point.append({"key_point": point, "term_coverage": round(coverage, 3)})

    total = len(matched) + len(missing)
    if total == 0:
        return None
    fraction = len(matched) / total
    score = fraction_to_score(fraction)

    feedback = f"Matched {len(matched)} of {total} reference key points."
    if missing:
        feedback += " Not found: " + "; ".join(f'"{m}"' for m in missing) + "."
    feedback += (
        " This is lexical agreement with the reference only; it does not detect contradictions "
        "or errors outside the reference, so a human should confirm accuracy."
    )
    return _auto(
        "accuracy",
        score,
        feedback,
        method="reference_key_point_coverage",
        key_points_matched=len(matched),
        key_points_total=total,
        coverage=round(fraction, 3),
        match_threshold=KEY_POINT_MATCH_THRESHOLD,
        per_key_point=per_point,
    )


# -------------------------------------------------------------------------- relevance


def score_relevance(item: EvaluationItem) -> CriterionScore:
    """Score the share of the prompt's topical terms that the response uses."""
    prompt_terms = unique_terms(item.prompt, INSTRUCTION_WORDS, skip_numbers=True)
    response_terms = set(content_terms(item.response))

    if not tokenize(item.response):
        return _auto("relevance", 1, "Response is empty.", method="prompt_term_coverage")
    if not prompt_terms:
        return _auto(
            "relevance",
            3,
            "Prompt has no topical terms to compare against; neutral score assigned.",
            method="prompt_term_coverage",
            prompt_terms=[],
        )

    found = [t for t in prompt_terms if t in response_terms]
    missing = [t for t in prompt_terms if t not in response_terms]
    fraction = len(found) / len(prompt_terms)
    score = fraction_to_score(fraction)
    feedback = f"Response uses {len(found)} of {len(prompt_terms)} topical prompt terms."
    if missing:
        feedback += f" Absent: {', '.join(missing)}."
    feedback += " Word overlap is a rough proxy and does not recognise paraphrases."
    return _auto(
        "relevance",
        score,
        feedback,
        method="prompt_term_coverage",
        coverage=round(fraction, 3),
        prompt_terms=prompt_terms,
        missing_terms=missing,
    )


# ---------------------------------------------------------------------------- clarity

LONG_SENTENCE_WORDS = 40


def score_clarity(item: EvaluationItem) -> CriterionScore:
    """Score readability: start at 5 and deduct for specific, named issues."""
    words = tokenize(item.response)
    if not words:
        return _auto("clarity", 1, "Response is empty.", method="readability_deductions")

    sentences = split_sentences(item.response) or [item.response]
    lengths = [len(tokenize(s)) for s in sentences]
    avg_len = len(words) / len(sentences)
    long_ratio = sum(n > LONG_SENTENCE_WORDS for n in lengths) / len(sentences)
    fre = flesch_reading_ease(item.response)

    normalised = [" ".join(tokenize(s)) for s in sentences]
    duplicate_sentences = len(normalised) - len(set(normalised))

    # Repeating the prompt's own topic words ("list" in an answer about lists) is
    # natural, so only non-topic terms count towards the repetition check.
    terms = content_terms(item.response)
    topic = set(content_terms(item.prompt))
    off_topic_counts = Counter(t for t in terms if t not in topic)
    top_term, top_count = (off_topic_counts.most_common(1) or [("", 0)])[0]
    repetition = top_count / len(terms) if terms else 0.0

    issues: list[tuple[int, str]] = []
    if avg_len > 35:
        issues.append((2, f"very long sentences (average {avg_len:.0f} words)"))
    elif avg_len > 25:
        issues.append((1, f"long sentences (average {avg_len:.0f} words)"))
    if long_ratio > 0.25:
        issues.append((1, f"{long_ratio:.0%} of sentences exceed {LONG_SENTENCE_WORDS} words"))
    if fre is not None and fre < 30:
        issues.append((1, f"low reading ease (Flesch {fre:.0f})"))
    if duplicate_sentences:
        issues.append((1, f"{duplicate_sentences} repeated sentence(s)"))
    if len(terms) >= 20 and repetition > 0.15:
        issues.append((1, f"heavy repetition of '{top_term}' ({repetition:.0%} of terms)"))

    score = _clamp(SCORE_MAX - sum(points for points, _ in issues))
    if issues:
        feedback = "Readability issues: " + "; ".join(text for _, text in issues) + "."
    else:
        feedback = "No readability issues detected by the heuristics."
    feedback += " Heuristics do not assess logical coherence or organisation."
    return _auto(
        "clarity",
        score,
        feedback,
        method="readability_deductions",
        word_count=len(words),
        sentence_count=len(sentences),
        average_sentence_length=round(avg_len, 2),
        long_sentence_ratio=round(long_ratio, 3),
        flesch_reading_ease=round(fre, 2) if fre is not None else None,
        duplicate_sentences=duplicate_sentences,
        top_term_share=round(repetition, 3),
    )


# -------------------------------------------------------------- instruction following


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False
    return True


def _check_format(fmt: str, text: str) -> tuple[bool, str]:
    if fmt == "bullet_list":
        n = len(bullet_items(text))
        return n >= 2, f"bulleted list ({n} bullet lines found)"
    if fmt == "numbered_list":
        n = len(numbered_items(text))
        return n >= 2, f"numbered list ({n} numbered lines found)"
    if fmt == "json":
        valid = _is_json(text.strip())
        return valid, "valid JSON" if valid else "response is not valid JSON on its own"
    if fmt == "paragraph":
        n = len(bullet_items(text)) + len(numbered_items(text))
        return n == 0, "prose without list formatting"
    raise ValueError(f"unsupported format {fmt!r}")  # pragma: no cover - guarded by Constraints


def check_constraints(constraints: Constraints, text: str) -> list[dict[str, object]]:
    """Run every declared constraint and return one record per check."""
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": passed, "detail": detail})

    n_words = len(tokenize(text))
    if constraints.max_words is not None:
        add(
            "max_words",
            n_words <= constraints.max_words,
            f"{n_words} words (limit {constraints.max_words})",
        )
    if constraints.min_words is not None:
        add(
            "min_words",
            n_words >= constraints.min_words,
            f"{n_words} words (minimum {constraints.min_words})",
        )
    if constraints.max_sentences is not None:
        n_sent = len(split_sentences(text))
        add(
            "max_sentences",
            n_sent <= constraints.max_sentences,
            f"{n_sent} sentences (limit {constraints.max_sentences})",
        )
    for kw in constraints.required_keywords:
        add(f"requires '{kw}'", contains_phrase(text, kw), f"required phrase '{kw}'")
    for phrase in constraints.forbidden_phrases:
        add(
            f"forbids '{phrase}'", not contains_phrase(text, phrase), f"forbidden phrase '{phrase}'"
        )
    if constraints.response_format is not None:
        passed, detail = _check_format(constraints.response_format, text)
        add(f"format={constraints.response_format}", passed, detail)
    if constraints.item_count is not None:
        items = (
            bullet_items(text)
            if constraints.response_format == "bullet_list"
            else numbered_items(text)
        )
        add(
            "item_count",
            len(items) == constraints.item_count,
            f"{len(items)} list items (expected {constraints.item_count})",
        )
    return checks


def score_instruction_following(item: EvaluationItem) -> CriterionScore | None:
    """Score the share of declared constraints satisfied. ``None`` if none declared."""
    if item.constraints.is_empty():
        return None
    checks = check_constraints(item.constraints, item.response)
    passed = [c for c in checks if c["passed"]]
    failed = [c for c in checks if not c["passed"]]
    fraction = len(passed) / len(checks)
    score = fraction_to_score(fraction)
    feedback = f"Satisfied {len(passed)} of {len(checks)} declared constraints."
    if failed:
        feedback += " Failed: " + "; ".join(f"{c['check']} - {c['detail']}" for c in failed) + "."
    return _auto(
        "instruction_following",
        score,
        feedback,
        method="declared_constraint_checks",
        checks_passed=len(passed),
        checks_total=len(checks),
        checks=checks,
    )


# ----------------------------------------------------------------------------- scorer

ScoringFunction = Callable[[EvaluationItem], "CriterionScore | None"]

DEFAULT_SCORERS: dict[str, ScoringFunction] = {
    "accuracy": score_accuracy,
    "relevance": score_relevance,
    "clarity": score_clarity,
    "instruction_following": score_instruction_following,
}


class AutomatedScorer:
    """Apply automated heuristics for each rubric criterion that has one.

    Args:
        rubric: Rubric whose criteria should be scored.
        scorers: Optional mapping of criterion key to scoring function, allowing
            custom heuristics to be plugged in or default ones replaced.
    """

    def __init__(
        self,
        rubric: Rubric | None = None,
        scorers: dict[str, ScoringFunction] | None = None,
    ) -> None:
        self.rubric = rubric or Rubric()
        self.scorers = dict(DEFAULT_SCORERS)
        if scorers:
            self.scorers.update(scorers)

    def score(self, item: EvaluationItem) -> dict[str, CriterionScore]:
        """Return automated scores for every criterion that could be scored."""
        results: dict[str, CriterionScore] = {}
        for key in self.rubric.keys:
            fn = self.scorers.get(key)
            if fn is None:
                continue
            result = fn(item)
            if result is not None:
                results[key] = result
        return results
