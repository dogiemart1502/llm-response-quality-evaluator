import json
import subprocess
import sys

from llm_eval.cli import EXIT_ERROR, EXIT_OK, EXIT_REVIEW_NEEDED, main


def test_evaluate_sample_data_exports_json(
    tmp_path, capsys, sample_dataset_path, sample_judgments_path
):
    out = tmp_path / "results.json"
    code = main(
        ["evaluate", str(sample_dataset_path), "-j", str(sample_judgments_path), "-o", str(out)]
    )
    assert code == EXIT_OK
    stdout = capsys.readouterr().out
    assert "photosynthesis-a" in stdout
    assert "Items needing review" in stdout
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["items_evaluated"] == 10


def test_evaluate_details_prints_feedback(capsys, sample_dataset_path, sample_judgments_path):
    main(["evaluate", str(sample_dataset_path), "-j", str(sample_judgments_path), "--details"])
    out = capsys.readouterr().out
    assert "Human (reviewer-1): 2/5" in out
    assert "Automated:" in out
    assert "evidence:" in out


def test_fail_on_review_exit_code(sample_dataset_path, sample_judgments_path):
    code = main(
        [
            "evaluate",
            str(sample_dataset_path),
            "-j",
            str(sample_judgments_path),
            "--fail-on-review",
            "-q",
        ]
    )
    assert code == EXIT_REVIEW_NEEDED


def test_custom_weights(capsys, sample_dataset_path):
    assert main(["evaluate", str(sample_dataset_path), "-w", "accuracy=1,clarity=1"]) == EXIT_OK


def test_invalid_weights_give_clean_error(capsys, sample_dataset_path):
    assert main(["evaluate", str(sample_dataset_path), "-w", "speed=1"]) == EXIT_ERROR
    assert "unknown criteria" in capsys.readouterr().err


def test_missing_file_gives_clean_error(capsys, tmp_path):
    assert main(["evaluate", str(tmp_path / "missing.json")]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Traceback" not in err


def test_invalid_dataset_reports_location(capsys, write_json):
    path = write_json({"items": [{"id": "a", "prompt": "p", "response": 5}]})
    assert main(["validate", str(path)]) == EXIT_ERROR
    assert "items[0].response" in capsys.readouterr().err


def test_no_automated_requires_judgments(capsys, sample_dataset_path):
    assert main(["evaluate", str(sample_dataset_path), "--no-automated"]) == EXIT_ERROR


def test_no_automated_with_judgments(tmp_path, sample_dataset_path, sample_judgments_path):
    out = tmp_path / "human_only.json"
    args = ["evaluate", str(sample_dataset_path), "-j", str(sample_judgments_path)]
    assert main([*args, "--no-automated", "-q", "-o", str(out)]) == EXIT_OK
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["methodology"]["automated_scoring_enabled"] is False
    for result in data["results"]:
        for crit in result["criteria"].values():
            assert crit["automated"] is None
            assert crit["final_source"] in ("human", None)


def test_rubric_command(capsys):
    assert main(["rubric"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "Instruction Following" in out
    assert "Automated proxy" in out


def test_validate_command(capsys, sample_dataset_path, sample_judgments_path):
    assert main(["validate", str(sample_dataset_path), "-j", str(sample_judgments_path)]) == 0
    assert "judgments OK: 9" in capsys.readouterr().out


def test_template_command(tmp_path, sample_dataset_path):
    out = tmp_path / "template.json"
    assert main(["template", str(sample_dataset_path), "-o", str(out)]) == EXIT_OK
    assert len(json.loads(out.read_text())["judgments"]) == 10


def test_module_entry_point(sample_dataset_path):
    proc = subprocess.run(
        [sys.executable, "-m", "llm_eval", "validate", str(sample_dataset_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Dataset OK" in proc.stdout
