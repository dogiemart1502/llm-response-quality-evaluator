"""Command-line interface: ``llm-eval`` (or ``python -m llm_eval``).

Sub-commands:

* ``evaluate`` - score a dataset (optionally with human judgments) and export JSON.
* ``rubric``   - print the rubric, level descriptors, and automated proxies.
* ``template`` - generate a blank human-judgment file for a dataset.
* ``validate`` - check a dataset (and judgments) without evaluating.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from llm_eval import __version__
from llm_eval.errors import EvaluatorError
from llm_eval.io import export_results, judgment_template, load_dataset, load_judgments, write_json
from llm_eval.report import format_item_detail, format_rubric, format_summary
from llm_eval.rubric import Rubric, parse_weights
from llm_eval.scoring import evaluate_dataset

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REVIEW_NEEDED = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-eval",
        description="Rubric-based quality evaluation of AI-generated responses.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    weights_help = "override criterion weights, e.g. accuracy=0.4,clarity=0.1"

    ev = sub.add_parser("evaluate", help="evaluate a dataset and export results to JSON")
    ev.add_argument("dataset", help="path to the dataset JSON file")
    ev.add_argument("-j", "--judgments", help="path to a human-judgments JSON file")
    ev.add_argument("-o", "--output", help="write full results to this JSON file")
    ev.add_argument("-w", "--weights", help=weights_help)
    ev.add_argument(
        "--no-automated",
        action="store_true",
        help="disable automated heuristic scoring (human scores only)",
    )
    ev.add_argument(
        "--details", action="store_true", help="print per-item feedback for every criterion"
    )
    ev.add_argument(
        "--fail-on-review",
        action="store_true",
        help=f"exit with code {EXIT_REVIEW_NEEDED} if any item needs review",
    )
    ev.add_argument("-q", "--quiet", action="store_true", help="suppress the summary table")

    rb = sub.add_parser("rubric", help="print the scoring rubric")
    rb.add_argument("-w", "--weights", help=weights_help)

    tp = sub.add_parser("template", help="create a blank human-judgment file for a dataset")
    tp.add_argument("dataset", help="path to the dataset JSON file")
    tp.add_argument("-o", "--output", required=True, help="where to write the template")

    va = sub.add_parser("validate", help="validate input files without evaluating")
    va.add_argument("dataset", help="path to the dataset JSON file")
    va.add_argument("-j", "--judgments", help="path to a human-judgments JSON file")
    return parser


def _rubric(weights: str | None) -> Rubric:
    rubric = Rubric()
    return rubric.with_weights(parse_weights(weights)) if weights else rubric


def _cmd_evaluate(args: argparse.Namespace) -> int:
    rubric = _rubric(args.weights)
    dataset = load_dataset(args.dataset)
    judgments = load_judgments(args.judgments, rubric, dataset) if args.judgments else None
    if args.no_automated and judgments is None:
        raise EvaluatorError("--no-automated requires --judgments; there would be nothing to score")

    run = evaluate_dataset(dataset, rubric, judgments, use_automated=not args.no_automated)
    if not args.quiet:
        print(format_summary(run))
        if args.details:
            for result in run.results:
                print()
                print(format_item_detail(result, rubric))
    if args.output:
        path = export_results(run, args.output)
        print(f"\nResults written to {path}", file=sys.stderr if args.quiet else sys.stdout)
    if args.fail_on_review and run.summary["items_needing_review"]:
        return EXIT_REVIEW_NEEDED
    return EXIT_OK


def _cmd_rubric(args: argparse.Namespace) -> int:
    print(format_rubric(_rubric(args.weights)))
    return EXIT_OK


def _cmd_template(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    path = write_json(judgment_template(dataset), args.output)
    print(f"Template for {len(dataset.items)} items written to {path}")
    return EXIT_OK


def _cmd_validate(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    message = f"Dataset OK: {len(dataset.items)} items"
    if args.judgments:
        judgments = load_judgments(args.judgments, Rubric(), dataset)
        message += f"; judgments OK: {len(judgments)} items judged"
    print(message)
    return EXIT_OK


COMMANDS = {
    "evaluate": _cmd_evaluate,
    "rubric": _cmd_rubric,
    "template": _cmd_template,
    "validate": _cmd_validate,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code instead of raising."""
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except EvaluatorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
