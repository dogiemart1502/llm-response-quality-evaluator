"""Plain-text reports for the terminal."""

from __future__ import annotations

import textwrap

from llm_eval.models import CriterionResult, EvaluationRun, ItemResult
from llm_eval.rubric import Rubric

_SHORT = {"accuracy": "ACC", "relevance": "REL", "clarity": "CLR", "instruction_following": "INS"}


def _short(key: str) -> str:
    return _SHORT.get(key, key[:3].upper())


def _cell(result: CriterionResult) -> str:
    """Render e.g. ``4h`` (human), ``3a`` (automated), ``4h/2a!`` (disagreement)."""
    final = result.final
    if final is None:
        return "-"
    text = f"{final.score}{final.source.value[0]}"
    if result.human is not None and result.automated is not None:
        text += f"/{result.automated.score}a"
        if result.needs_review:
            text += "!"
    return text


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def format_table(run: EvaluationRun) -> str:
    keys = run.rubric.keys
    headers = ["ITEM", "MODEL", *(_short(k) for k in keys), "OVERALL", "HUMAN", "AUTO"]
    rows = []
    for r in run.results:
        overall = _fmt(r.overall.score) + ("" if r.overall.complete else "*")
        rows.append(
            [
                r.item.id,
                r.item.model or "-",
                *(_cell(r.criteria[k]) for k in keys),
                overall,
                _fmt(r.human_overall),
                _fmt(r.automated_overall),
            ]
        )
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows, strict=True)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))]
    lines.append("  ".join("-" * w for w in widths))
    lines += ["  ".join(str(c).ljust(w) for c, w in zip(row, widths, strict=True)) for row in rows]
    return "\n".join(lines)


def format_summary(run: EvaluationRun) -> str:
    s = run.summary
    pooled = s["human_vs_automated_agreement"]["pooled"]
    lines = [
        f"Dataset: {run.dataset.name} ({s['items_evaluated']} items, "
        f"{s['items_with_human_judgment']} with human judgments)",
        "",
        format_table(run),
        "",
        "Legend: h = human score, a = automated heuristic score, 4h/2a = human 4 vs automated 2,",
        "        ! = human and automated differ by >= 2 (needs review),",
        "        * = overall computed without every criterion, - = not scored.",
        "",
        f"Mean overall score: {_fmt(s['mean_overall'])}",
    ]
    if pooled["pairs"]:
        lines.append(
            f"Human vs automated agreement ({pooled['pairs']} pairs): "
            f"exact {pooled['exact_agreement']:.0%}, within one point {pooled['within_one']:.0%}, "
            f"mean absolute difference {pooled['mean_absolute_difference']:.2f}"
        )
    if s["items_needing_review"]:
        lines.append("Items needing review: " + ", ".join(s["items_needing_review"]))
    if run.automated_enabled:
        lines.append("")
        lines.append(
            textwrap.fill(
                "Note: automated scores are heuristic signals. Automated accuracy only measures "
                "overlap with human-written reference key points and cannot verify facts.",
                width=90,
            )
        )
    return "\n".join(lines)


def format_item_detail(result: ItemResult, rubric: Rubric) -> str:
    lines = [f"== {result.item.id} ({result.item.model or 'unspecified model'}) =="]
    for key in rubric.keys:
        crit = result.criteria[key]
        lines.append(f"[{rubric.get(key).name}]")
        for score in (crit.human, crit.automated):
            if score is None:
                continue
            label = "Human" if score.source.value == "human" else "Automated"
            who = f" ({score.annotator})" if score.annotator else ""
            lines.append(f"  {label}{who}: {score.score}/5")
            lines.append(textwrap.indent(textwrap.fill(score.feedback, 84), "    "))
            for ev in score.evidence:
                lines.append(f"    evidence: {ev}")
        if crit.final is None:
            lines.append("  Not scored (no human judgment and no automated basis).")
    lines.append(
        f"Overall: {_fmt(result.overall.score)}"
        + ("" if result.overall.complete else f" (missing: {', '.join(result.overall.missing)})")
    )
    if result.comment:
        lines.append(f"Reviewer comment: {result.comment}")
    return "\n".join(lines)


def format_rubric(rubric: Rubric) -> str:
    lines = ["Scoring scale: 1 (worst) to 5 (best)", ""]
    for c in rubric.criteria:
        lines.append(f"{c.name} [{c.key}] - weight {rubric.weights[c.key]:.2f}")
        lines.append(textwrap.indent(textwrap.fill(c.description, 86), "  "))
        for score in sorted(c.levels, reverse=True):
            lines.append(f"    {score}: {c.levels[score]}")
        lines.append(
            textwrap.indent(textwrap.fill("Automated proxy: " + c.automated_proxy, 86), "  ")
        )
        lines.append("")
    return "\n".join(lines).rstrip()
