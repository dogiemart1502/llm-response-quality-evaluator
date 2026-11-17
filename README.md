# LLM Response Quality Evaluator

[![Tests](https://github.com/dogiemart1502/llm-response-quality-evaluator/actions/workflows/tests.yml/badge.svg)](https://github.com/dogiemart1502/llm-response-quality-evaluator/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
![Dependencies](https://img.shields.io/badge/runtime%20dependencies-none-brightgreen)

A Python tool for evaluating AI-generated responses with a 1–5 rubric for
**accuracy**, **relevance**, **clarity**, and **instruction following**. It is
built around the way human evaluation of model output usually works: reviewers
score each criterion and justify every score in writing, and lightweight
automated checks help them triage and spot disagreements.

It needs no paid APIs, no API keys, and no third-party runtime dependencies.

> **Human judgments and automated scores are kept separate.** Every score in the
> output is labeled `"source": "human"` or `"source": "automated"`. Human scores
> always take precedence. The automated scores are transparent heuristics, and
> the tool **does not claim to verify facts**. Its accuracy signal only measures
> agreement with reference key points that a person wrote.

---

## Features

| Area | What it does |
|---|---|
| **Rubric** | Four criteria, each with a written descriptor for every level from 1 to 5 and configurable weights (`llm-eval rubric` prints it). |
| **Human judgments** | Load reviewer scores from JSON. **Written feedback is required for every score**, and optional `evidence` (citations, sources) can back up accuracy calls. |
| **Automated signals** | Deterministic heuristics that are easy to explain: reference key-point coverage, prompt-term coverage, readability statistics, and checks against declared constraints (word/sentence limits, required or forbidden phrases, bullet/numbered/JSON/paragraph format, item count). Each automated score includes its raw `signals` and a written explanation. |
| **Scoring** | Weighted overall score on the 1–5 scale. Human and automated overalls are also reported separately. Criteria with no score are reported as missing, never filled in with a guess. |
| **Agreement analysis** | Exact agreement, agreement within one point, and mean absolute/signed difference between human and automated scores. Items where the two differ by ≥ 2 points are flagged for review. |
| **JSON export** | A versioned schema with the methodology, rubric, per-criterion scores from each source, feedback, evidence and summary statistics. Writes are atomic. |
| **Validation** | Schema validation with exact error locations (`items[3].constraints.max_words: must be a positive integer`), clean CLI error messages, and no tracebacks for expected errors. |
| **Quality** | 160 pytest tests (~96% coverage), ruff lint and format checks, and GitHub Actions on Python 3.10–3.13. |

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/dogiemart1502/llm-response-quality-evaluator.git
```

```bash
cd llm-response-quality-evaluator
```

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

```bash
pip install -e ".[dev]"
```

This installs the `llm-eval` command. You can also run the tool with
`python -m llm_eval`.

## Quick start

Evaluate the bundled sample responses against the bundled human judgments and
export the results:

```bash
llm-eval evaluate data/sample_responses.json -j data/human_judgments.json -o results.json
```

```text
ITEM              MODEL           ACC     REL     CLR    INS    OVERALL  HUMAN  AUTO
----------------  --------------  ------  ------  -----  -----  -------  -----  ----
photosynthesis-a  sample-model-a  5h/4a   5h/5a   5h/5a  5h/5a  5.00     5.00   4.65
photosynthesis-b  sample-model-b  2h/5a!  5h/5a   4h/5a  5h/5a  3.75     3.75   5.00
exercise-a        sample-model-a  5h/4a   5h/1a!  5h/5a  5h/5a  5.00     5.00   3.85
exercise-b        sample-model-b  4h/4a   4h/4a   1h/1a  1h/1a  2.65     2.65   2.65
seasons-a         sample-model-a  5h/5a   5h/4a   5h/5a  5h/5a  5.00     5.00   4.80
seasons-b         sample-model-b  1h/4a!  5h/4a   5h/5a  5h/5a  3.60     3.60   4.45
json-capital-a    sample-model-a  5h/5a   5h/5a   5h/5a  5h/5a  5.00     5.00   5.00
json-capital-b    sample-model-b  5h/5a   4h/5a   4h/5a  2h/3a  3.85     3.85   4.50
python-reverse-b  sample-model-b  2h/2a   1h/3a!  4h/5a  1h     1.95     1.95   -
python-reverse-a  sample-model-a  -       3a      5a     -      4.00*    -      -

Legend: h = human score, a = automated heuristic score, 4h/2a = human 4 vs automated 2,
        ! = human and automated differ by >= 2 (needs review),
        * = overall computed without every criterion, - = not scored.

Mean overall score: 3.98
Human vs automated agreement (35 pairs): exact 63%, within one point 89%, mean absolute difference 0.60
Items needing review: photosynthesis-b, exercise-a, seasons-b, python-reverse-b
```

The flagged rows show where the heuristics fall short and why human review is
part of the design:

- **`photosynthesis-b`** says plants *take in oxygen and release carbon dioxide*,
  which is backwards. Every reference term is present, so automated accuracy
  gives 5. The human reviewer gives 2 and cites the reaction equation.
- **`seasons-b`** states the distance-from-the-Sun misconception. The reference
  point "*not* caused by distance from the Sun" still matches lexically because
  the heuristic does not detect negation.
- **`exercise-a`** is a good answer. Automated relevance gives it 1 because the
  bullets never repeat the words "benefit" or "exercise".
- **`python-reverse-a`** has no human judgment and no reference key points, so
  accuracy is reported as **not scored** instead of being estimated, and the
  overall score is marked incomplete (`*`).

The full exported output for this run is in
[`examples/sample_results.json`](examples/sample_results.json).

## Usage

### Commands

| Command | Purpose |
|---|---|
| `llm-eval evaluate DATASET [-j JUDGMENTS] [-o OUT.json]` | Score a dataset, print a summary, and optionally export JSON. |
| `llm-eval validate DATASET [-j JUDGMENTS]` | Check input files without evaluating. |
| `llm-eval template DATASET -o judgments.json` | Generate a blank judgment file for reviewers to fill in. |
| `llm-eval rubric` | Print the rubric, level descriptors, weights, and what each automated proxy measures. |

Useful `evaluate` options:

| Option | Effect |
|---|---|
| `--details` | Print every score with its written feedback and evidence. |
| `-w accuracy=0.5,clarity=0.1` | Override criterion weights (they are re-normalised to sum to 1). |
| `--no-automated` | Human scores only. Requires `-j`. |
| `--fail-on-review` | Exit with code `3` if any item needs review, which is useful in CI pipelines. |
| `-q` | Suppress the table, for example when only exporting JSON. |

Exit codes: `0` success, `1` input or validation error (the message goes to
stderr), `2` invalid command-line usage, `3` review needed (only with
`--fail-on-review`).

### Typical workflow

1. Put prompts and model responses in a dataset file. Add `constraints` where
   the prompt contains instructions that a machine can check, and add
   `reference_key_points` where you have a trusted reference.
2. Create a judgment file with `llm-eval template` and hand it to reviewers.
3. Run `llm-eval evaluate`, then look at the flagged items first.
4. Use the exported JSON for further analysis, for example to compare models
   with `summary.by_model`.

### Python API

```python
from llm_eval.io import export_results, load_dataset, load_judgments
from llm_eval.rubric import Rubric
from llm_eval.scoring import evaluate_dataset

rubric = Rubric().with_weights({"accuracy": 0.5})
dataset = load_dataset("data/sample_responses.json")
judgments = load_judgments("data/human_judgments.json", rubric, dataset)

run = evaluate_dataset(dataset, rubric, judgments)
for result in run.results:
    acc = result.criteria["accuracy"]
    print(result.item.id, acc.final.score if acc.final else None, acc.needs_review)

export_results(run, "results.json")
```

You can plug in custom heuristics, for example a domain-specific checker, through
`AutomatedScorer(rubric, scorers={"clarity": my_function})`.

## Input formats

### Dataset (`data/sample_responses.json`)

```json
{
  "name": "my-dataset",
  "description": "optional",
  "items": [
    {
      "id": "seasons-a",
      "model": "model-x",
      "prompt": "Summarize the main cause of the seasons on Earth in two sentences.",
      "response": "Seasons happen because Earth's axis is tilted ...",
      "reference_key_points": [
        "Earth's axis is tilted about 23.5 degrees",
        "Seasons are not caused by distance from the Sun"
      ],
      "constraints": {"max_sentences": 2}
    }
  ]
}
```

`id`, `prompt` and `response` are required. The response may be empty, in which
case it scores poorly. Supported `constraints` are `max_words`, `min_words`,
`max_sentences`, `required_keywords`, `forbidden_phrases`, `format`
(`bullet_list` | `numbered_list` | `json` | `paragraph`), and `item_count` (list
formats only). Unknown fields are rejected so that typos are not silently
ignored.

### Human judgments (`data/human_judgments.json`)

```json
{
  "annotator": "reviewer-1",
  "judgments": [
    {
      "item_id": "seasons-b",
      "scores": {
        "accuracy": {
          "score": 1,
          "feedback": "States the distance-from-the-Sun misconception.",
          "evidence": ["Perihelion occurs in early January (northern-hemisphere winter)."]
        },
        "clarity": {"score": 5, "feedback": "Clearly written."}
      },
      "comment": "optional free-text note"
    }
  ]
}
```

Scores must be integers from 1 to 5, `feedback` is required, and a reviewer may
score only some of the criteria. Each `item_id` must exist in the dataset.

## Output format

Each result in the exported JSON looks like this (shortened):

```json
{
  "item_id": "photosynthesis-b",
  "criteria": {
    "accuracy": {
      "final_score": 2,
      "final_source": "human",
      "human":     {"score": 2, "source": "human", "feedback": "Reverses the gas exchange ...",
                    "annotator": "reviewer-1", "evidence": ["6CO2 + 6H2O + light -> ..."]},
      "automated": {"score": 5, "source": "automated", "feedback": "Matched 4 of 4 ...",
                    "signals": {"method": "reference_key_point_coverage", "coverage": 1.0}},
      "human_minus_automated": -3,
      "needs_review": true
    }
  },
  "overall": {"score": 3.75, "complete": true, "missing_criteria": [],
              "score_sources": {"human": 4, "automated": 0}},
  "human_overall": 3.75,
  "automated_overall": 5.0
}
```

The top level of the file also contains `schema_version`, `methodology`
(scoring policy and disclaimer), `rubric`, and `summary` (per-criterion means,
per-model means, agreement statistics, and items needing review).

## How scores are calculated

**Final score for each criterion.** The human score is used if there is one.
Otherwise the automated score is used. If neither exists, the criterion is
reported as missing.

**Overall score.** The overall score is the weighted mean of the final
criterion scores, rounded to 2 decimal places. Default weights are accuracy
0.35, instruction following 0.25, relevance 0.20 and clarity 0.20. When a
criterion is missing, the weights of the remaining criteria are re-normalised
and the result is marked `"complete": false`.

**Source-specific overalls.** `human_overall` and `automated_overall` are
computed only when that source scored every criterion, so they can always be
compared directly.

**Automated heuristics** (see `src/llm_eval/automated.py`):

| Criterion | Signal | Mapping to 1–5 |
|---|---|---|
| Accuracy | Share of reference key points whose content terms appear in the response (≥ 60 % term overlap per point). **Not scored without reference key points.** | `1 + 4 × fraction` |
| Relevance | Share of the prompt's topical terms (instruction words and numbers removed) that appear in the response. | `1 + 4 × fraction` |
| Clarity | Starts at 5 and deducts points for long sentences, low Flesch reading ease, duplicated sentences, and heavy repetition of non-topic terms. | `5 − deductions` (minimum 1) |
| Instruction following | Share of declared constraints that pass. **Not scored without constraints.** | `1 + 4 × fraction` |

## Limitations

This tool supports human evaluation. It does not replace it.

- **No independent fact-checking.** The automated accuracy score only measures
  word overlap with reference key points that a person supplied. It cannot
  detect negation, contradictions, wrong relationships between correct terms,
  or errors that the reference does not cover. Accuracy should be judged by
  people, using evidence.
- **Lexical heuristics only.** Relevance and accuracy rely on word overlap with
  crude stemming, so paraphrases and synonyms are missed. Clarity uses
  readability statistics and does not assess reasoning or structure.
- **English only.** Tokenisation, stopwords and syllable counting assume English.
- **Instruction checks are limited to what you declare.** Instructions written
  only in the prompt text (for example "use a friendly tone") are not checked
  automatically.
- **One annotator per item per file.** There is no built-in adjudication or
  inter-annotator agreement (such as Cohen's kappa) across multiple reviewers yet.
- **Illustrative sample data.** The bundled responses and judgments were
  written by hand to demonstrate the workflow. `sample-model-a` and
  `sample-model-b` are placeholder labels, not real models, and the judgments
  are examples rather than results from a study.
- **The mean overall score in the summary includes incomplete items.** Check
  `items_complete` and each item's `overall.complete` flag before comparing
  scores.

## Project structure

```text
src/llm_eval/
  rubric.py        Criteria, 1-5 level descriptors, weights
  models.py        Dataclasses: items, constraints, scores, results
  validation.py    JSON schema validation with path-aware errors
  automated.py     Transparent heuristic scorers
  text_utils.py    Tokenisation, sentence splitting, readability
  scoring.py       Score combination, overall scores, agreement statistics
  io.py            File loading, atomic JSON export, judgment templates
  report.py        Terminal tables and detail views
  cli.py           Command-line interface
data/              Sample dataset and illustrative human judgments
examples/          Exported results for the sample data
tests/             pytest suite (unit, CLI and end-to-end tests)
```

## Development

Run the tests:

```bash
python -m pytest
```

Run the tests with a coverage report:

```bash
python -m pytest --cov=llm_eval --cov-report=term-missing
```

Lint and check formatting:

```bash
ruff check .
```

```bash
ruff format --check .
```

If you change the sample data or the heuristics, regenerate the committed
example output (a test checks that it stays in sync):

```bash
llm-eval evaluate data/sample_responses.json -j data/human_judgments.json -o examples/sample_results.json -q
```

GitHub Actions runs lint, the test suite on Python 3.10–3.13, and a CLI smoke
test on every push and pull request.
