"""Reading input files and exporting results to JSON."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_eval.automated import AUTOMATED_DISCLAIMER
from llm_eval.errors import InputFileError
from llm_eval.models import Dataset, EvaluationRun, JudgmentSet
from llm_eval.rubric import Rubric
from llm_eval.scoring import SCORING_POLICY
from llm_eval.validation import parse_dataset, parse_judgments

SCHEMA_VERSION = "1.0"
MAX_FILE_BYTES = 20 * 1024 * 1024


def load_json(path: str | os.PathLike[str]) -> Any:
    """Load a JSON file, converting low-level failures into :class:`InputFileError`."""
    p = Path(path)
    if not p.exists():
        raise InputFileError(f"file not found: {p}")
    if not p.is_file():
        raise InputFileError(f"not a regular file: {p}")
    size = p.stat().st_size
    if size > MAX_FILE_BYTES:
        raise InputFileError(f"file is too large ({size} bytes; limit {MAX_FILE_BYTES}): {p}")
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise InputFileError(f"file is not valid UTF-8: {p} ({exc.reason})") from None
    except OSError as exc:
        raise InputFileError(f"could not read {p}: {exc.strerror or exc}") from None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputFileError(
            f"invalid JSON in {p} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from None


def load_dataset(path: str | os.PathLike[str]) -> Dataset:
    return parse_dataset(load_json(path))


def load_judgments(
    path: str | os.PathLike[str], rubric: Rubric | None = None, dataset: Dataset | None = None
) -> JudgmentSet:
    return parse_judgments(load_json(path), rubric=rubric, dataset=dataset)


def run_to_dict(run: EvaluationRun, generated_at: datetime | None = None) -> dict[str, Any]:
    """Serialise an :class:`EvaluationRun` into a JSON-compatible dict."""
    from llm_eval import __version__

    timestamp = (generated_at or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": timestamp,
        "tool": {"name": "llm-response-quality-evaluator", "version": __version__},
        "dataset": {
            "name": run.dataset.name,
            "description": run.dataset.description,
            "items": len(run.dataset.items),
        },
        "methodology": {
            "scoring_policy": SCORING_POLICY,
            "automated_scoring_enabled": run.automated_enabled,
            "automated_scoring_disclaimer": AUTOMATED_DISCLAIMER,
            "score_sources": {
                "human": "Score and feedback provided by a human reviewer (authoritative).",
                "automated": "Score computed by deterministic heuristics (a signal, not a "
                "verified judgment).",
            },
        },
        "rubric": run.rubric.to_dict(),
        "summary": dict(run.summary),
        "results": [r.to_dict() for r in run.results],
    }


def write_json(data: Any, path: str | os.PathLike[str]) -> Path:
    """Write JSON atomically (temp file + rename), creating parent directories."""
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
            os.replace(tmp, p)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    except OSError as exc:
        raise InputFileError(f"could not write {p}: {exc.strerror or exc}") from None
    return p


def export_results(
    run: EvaluationRun, path: str | os.PathLike[str], generated_at: datetime | None = None
) -> Path:
    """Export an evaluation run to a JSON file and return its path."""
    return write_json(run_to_dict(run, generated_at), path)


def judgment_template(dataset: Dataset, rubric: Rubric | None = None) -> dict[str, Any]:
    """Build a blank human-judgment file for annotators to fill in."""
    rubric = rubric or Rubric()
    return {
        "annotator": "REPLACE-WITH-YOUR-ID",
        "description": f"Human judgments for dataset '{dataset.name}'.",
        "judgments": [
            {
                "item_id": item.id,
                "scores": {
                    key: {"score": None, "feedback": "", "evidence": []} for key in rubric.keys
                },
                "comment": None,
            }
            for item in dataset.items
        ],
    }
