from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH
from config import LLM_PROVIDER, OPENAI_API_KEY


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    if not _valid_openai_key():
        print("  Warning: OPENAI_API_KEY missing/placeholder; using lexical evaluation fallback.")
        return _fallback_evaluate(questions, answers, contexts, ground_truths)

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        df = result.to_pandas()
        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=list(row["contexts"]),
                ground_truth=row["ground_truth"],
                faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
                answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
                context_precision=float(row.get("context_precision", 0.0) or 0.0),
                context_recall=float(row.get("context_recall", 0.0) or 0.0),
            )
            for _, row in df.iterrows()
        ]
        return {
            "faithfulness": _mean([r.faithfulness for r in per_question]),
            "answer_relevancy": _mean([r.answer_relevancy for r in per_question]),
            "context_precision": _mean([r.context_precision for r in per_question]),
            "context_recall": _mean([r.context_recall for r in per_question]),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  Warning: RAGAS evaluation failed: {e}")
        return _fallback_evaluate(questions, answers, contexts, ground_truths)


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature, quote source spans"),
        "context_recall": ("Missing relevant chunks", "Improve chunking, add BM25 terms, or enrich context"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking, metadata filters, or lower top_k"),
        "answer_relevancy": ("Answer does not match question", "Improve prompt template and question understanding"),
    }

    analyzed = []
    for result in eval_results:
        scores = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }
        avg_score = _mean(scores.values())
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": result.question,
            "answer": result.answer,
            "ground_truth": result.ground_truth,
            "worst_metric": worst_metric,
            "score": float(scores[worst_metric]),
            "avg_score": float(avg_score),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    return sorted(analyzed, key=lambda item: item["avg_score"])[:bottom_n]


def _valid_openai_key() -> bool:
    return bool(
        LLM_PROVIDER == "openai"
        and OPENAI_API_KEY
        and OPENAI_API_KEY != "sk-..."
        and len(OPENAI_API_KEY) > 20
    )


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE))


def _overlap_score(source: str, target: str) -> float:
    source_tokens = _tokens(source)
    target_tokens = _tokens(target)
    if not source_tokens or not target_tokens:
        return 0.0
    return len(source_tokens & target_tokens) / max(len(target_tokens), 1)


def _mean(values) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _fallback_evaluate(questions: list[str], answers: list[str],
                       contexts: list[list[str]], ground_truths: list[str]) -> dict:
    per_question = []
    for question, answer, ctxs, ground_truth in zip(questions, answers, contexts, ground_truths):
        joined_context = "\n\n".join(ctxs)
        faithfulness_score = _overlap_score(answer, joined_context)
        relevancy_score = _mean([
            _overlap_score(answer, question),
            _overlap_score(answer, ground_truth),
        ])
        precision_scores = [
            _mean([_overlap_score(ctx, question), _overlap_score(ctx, ground_truth)])
            for ctx in ctxs
        ]
        context_precision_score = _mean(precision_scores)
        context_recall_score = _overlap_score(joined_context, ground_truth)
        per_question.append(EvalResult(
            question=question,
            answer=answer,
            contexts=ctxs,
            ground_truth=ground_truth,
            faithfulness=faithfulness_score,
            answer_relevancy=relevancy_score,
            context_precision=context_precision_score,
            context_recall=context_recall_score,
        ))

    return {
        "faithfulness": _mean([r.faithfulness for r in per_question]),
        "answer_relevancy": _mean([r.answer_relevancy for r in per_question]),
        "context_precision": _mean([r.context_precision for r in per_question]),
        "context_recall": _mean([r.context_recall for r in per_question]),
        "per_question": per_question,
    }


def save_report(results: dict, failures: list[dict], path: str = os.path.join("reports", "ragas_report.json")):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    report_dir = os.path.dirname(path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Report saved to {path}")
    except PermissionError as e:
        fallback_path = os.path.basename(path)
        with open(fallback_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  Warning: Cannot write {path}: {e}. Saved to {fallback_path} instead.")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
