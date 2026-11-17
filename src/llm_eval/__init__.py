"""LLM Response Quality Evaluator.

A small, dependency-free toolkit for scoring AI-generated responses against a
1-5 rubric (accuracy, relevance, clarity, instruction following).

The package keeps two kinds of scores strictly separate:

* **Human judgments** - scores and written feedback supplied by a reviewer.
  These are treated as authoritative.
* **Automated scores** - deterministic, transparent heuristics (lexical overlap,
  readability, constraint checks). These are *signals* to help reviewers
  triage, not ground truth. In particular, the automated accuracy signal only
  measures agreement with reference key points that a human supplied; it does
  not and cannot verify facts on its own.
"""

from llm_eval.automated import AutomatedScorer
from llm_eval.errors import EvaluatorError, InputFileError, ValidationError
from llm_eval.models import (
    Constraints,
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
from llm_eval.rubric import DEFAULT_CRITERIA, Criterion, Rubric
from llm_eval.scoring import evaluate_dataset, evaluate_item

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_CRITERIA",
    "AutomatedScorer",
    "Constraints",
    "Criterion",
    "CriterionResult",
    "CriterionScore",
    "Dataset",
    "EvaluationItem",
    "EvaluationRun",
    "EvaluatorError",
    "HumanJudgment",
    "InputFileError",
    "ItemResult",
    "JudgmentSet",
    "OverallScore",
    "Rubric",
    "ScoreSource",
    "ValidationError",
    "__version__",
    "evaluate_dataset",
    "evaluate_item",
]
