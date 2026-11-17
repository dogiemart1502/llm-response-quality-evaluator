import json
from datetime import datetime, timezone

import pytest

from llm_eval.errors import InputFileError, ValidationError
from llm_eval.io import (
    SCHEMA_VERSION,
    export_results,
    judgment_template,
    load_dataset,
    load_json,
    load_judgments,
    run_to_dict,
)
from llm_eval.scoring import evaluate_dataset


def test_load_json_missing_file(tmp_path):
    with pytest.raises(InputFileError, match="not found"):
        load_json(tmp_path / "nope.json")


def test_load_json_directory(tmp_path):
    with pytest.raises(InputFileError, match="not a regular file"):
        load_json(tmp_path)


def test_load_json_reports_position_of_syntax_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{\n  "a": 1\n  "b": 2\n}', encoding="utf-8")  # missing comma
    with pytest.raises(InputFileError, match=r"line 3, column \d+"):
        load_json(path)


def test_load_json_rejects_non_utf8(tmp_path):
    path = tmp_path / "latin1.json"
    path.write_bytes(b'{"a": "\xe9"}')
    with pytest.raises(InputFileError, match="UTF-8"):
        load_json(path)


def test_load_dataset_validates(write_json):
    with pytest.raises(ValidationError):
        load_dataset(write_json({"items": []}))


def test_export_round_trip(tmp_path, write_json, minimal_dataset, minimal_judgments):
    dataset = load_dataset(write_json(minimal_dataset, "d.json"))
    judgments = load_judgments(write_json(minimal_judgments, "j.json"), dataset=dataset)
    run = evaluate_dataset(dataset, judgments=judgments)

    fixed = datetime(2026, 1, 1, tzinfo=timezone.utc)
    out = export_results(run, tmp_path / "nested" / "dir" / "results.json", generated_at=fixed)
    data = json.loads(out.read_text(encoding="utf-8"))

    assert data["schema_version"] == SCHEMA_VERSION
    assert data["generated_at"] == "2026-01-01T00:00:00+00:00"
    assert "Human scores take precedence" in data["methodology"]["scoring_policy"]
    assert "does not verify facts" in data["methodology"]["automated_scoring_disclaimer"]
    result = data["results"][0]
    acc = result["criteria"]["accuracy"]
    assert acc["final_source"] == "human"
    assert acc["human"]["source"] == "human"
    assert acc["human"]["evidence"] == ["textbook"]
    assert acc["automated"]["source"] == "automated"
    assert "signals" in acc["automated"]
    assert result["overall"]["complete"] is True
    assert result["human_overall"] is not None
    # No temp files left behind by the atomic write.
    assert [p.name for p in out.parent.iterdir()] == ["results.json"]


def test_run_to_dict_is_json_serialisable(minimal_dataset):
    from llm_eval.validation import parse_dataset

    run = evaluate_dataset(parse_dataset(minimal_dataset))
    json.dumps(run_to_dict(run))


def test_export_to_unwritable_location_raises(tmp_path, minimal_dataset):
    from llm_eval.validation import parse_dataset

    blocker = tmp_path / "file"
    blocker.write_text("x")
    run = evaluate_dataset(parse_dataset(minimal_dataset))
    with pytest.raises(InputFileError, match="could not write"):
        export_results(run, blocker / "results.json")


def test_judgment_template_covers_every_item_and_criterion(minimal_dataset):
    from llm_eval.validation import parse_dataset

    template = judgment_template(parse_dataset(minimal_dataset))
    assert [j["item_id"] for j in template["judgments"]] == ["q1"]
    assert set(template["judgments"][0]["scores"]) == {
        "accuracy",
        "relevance",
        "clarity",
        "instruction_following",
    }
