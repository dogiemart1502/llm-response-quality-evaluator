"""Shared fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llm_eval.models import Constraints, EvaluationItem

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


@pytest.fixture
def sample_dataset_path() -> Path:
    return DATA_DIR / "sample_responses.json"


@pytest.fixture
def sample_judgments_path() -> Path:
    return DATA_DIR / "human_judgments.json"


@pytest.fixture
def write_json(tmp_path: Path):
    """Write ``data`` to a temp file and return its path."""

    def _write(data: Any, name: str = "data.json") -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def minimal_dataset() -> dict[str, Any]:
    return {
        "name": "mini",
        "items": [
            {
                "id": "q1",
                "model": "m",
                "prompt": "Explain what photosynthesis is in no more than 50 words.",
                "response": "Photosynthesis converts light energy into chemical energy in plants.",
                "reference_key_points": ["converts light energy into chemical energy"],
                "constraints": {"max_words": 50},
            }
        ],
    }


@pytest.fixture
def minimal_judgments() -> dict[str, Any]:
    return {
        "annotator": "tester",
        "judgments": [
            {
                "item_id": "q1",
                "scores": {
                    "accuracy": {"score": 5, "feedback": "Correct.", "evidence": ["textbook"]},
                    "relevance": {"score": 5, "feedback": "On topic."},
                    "clarity": {"score": 4, "feedback": "Clear but terse."},
                    "instruction_following": {"score": 5, "feedback": "Under the limit."},
                },
            }
        ],
    }


def make_item(
    response: str,
    prompt: str = "Explain photosynthesis.",
    key_points: tuple[str, ...] = (),
    constraints: Constraints | None = None,
    item_id: str = "item",
) -> EvaluationItem:
    return EvaluationItem(
        id=item_id,
        prompt=prompt,
        response=response,
        reference_key_points=key_points,
        constraints=constraints or Constraints(),
    )
