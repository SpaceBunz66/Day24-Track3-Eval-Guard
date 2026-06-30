from __future__ import annotations

"""Phase A: RAGAS production evaluation for the 50-question test set."""

import json
import os
import sys
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ANSWERS_PATH, TEST_SET_PATH

Distribution = str

DIAGNOSTIC_TREE = {
    "faithfulness": ("LLM hallucinating", "Tighten system prompt, lower temperature"),
    "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    "answer_relevancy": ("Answer does not match question", "Improve prompt template"),
}


@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return (
            self.faithfulness
            + self.answer_relevancy
            + self.context_precision
            + self.context_recall
        ) / 4

    @property
    def worst_metric(self) -> str:
        scores = {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
        }
        return min(scores, key=scores.get)


def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    """Load the 50-question test set."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    """Load pre-generated answers from setup_answers.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"answers_50q.json not found at {path}\n"
            "Run first: python setup_answers.py"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    """Group the test set by distribution."""
    groups = {"factual": [], "multi_hop": [], "adversarial": []}
    for item in test_set:
        distribution = item.get("distribution", "unknown")
        groups.setdefault(distribution, []).append(item)
    return groups


def run_ragas_50q(answers: list[dict]) -> list[RagasResult]:
    """Run the Day 18 RAGAS evaluator over the generated 50 answers."""
    try:
        from src.m4_eval import evaluate_ragas
    except ImportError:
        print("Cannot import src.m4_eval. Copy the Day 18 files into src/ first.")
        return []

    questions = [item.get("question", "") for item in answers]
    answer_texts = [item.get("answer", "") for item in answers]
    contexts = [
        item.get("contexts", [])
        if isinstance(item.get("contexts", []), list)
        else [str(item.get("contexts", ""))]
        for item in answers
    ]
    ground_truths = [item.get("ground_truth", "") for item in answers]

    raw = evaluate_ragas(questions, answer_texts, contexts, ground_truths)
    per_question = raw.get("per_question", [])

    results: list[RagasResult] = []
    for index, (answer_item, eval_item) in enumerate(zip(answers, per_question), start=1):
        def metric(name: str) -> float:
            if isinstance(eval_item, dict):
                return float(eval_item.get(name, 0.0) or 0.0)
            return float(getattr(eval_item, name, 0.0) or 0.0)

        results.append(RagasResult(
            question_id=int(answer_item.get("id", answer_item.get("question_id", index))),
            distribution=answer_item.get("distribution", "unknown"),
            question=answer_item.get("question", ""),
            answer=answer_item.get("answer", ""),
            contexts=contexts[index - 1],
            ground_truth=answer_item.get("ground_truth", ""),
            faithfulness=metric("faithfulness"),
            answer_relevancy=metric("answer_relevancy"),
            context_precision=metric("context_precision"),
            context_recall=metric("context_recall"),
        ))
    return results


def bottom_10(results: list[RagasResult]) -> list[dict]:
    """Return the lowest-scoring questions with diagnosis and suggested fix."""
    output = []
    for rank, result in enumerate(sorted(results, key=lambda r: r.avg_score)[:10], start=1):
        diagnosis, suggested_fix = DIAGNOSTIC_TREE[result.worst_metric]
        output.append({
            "rank": rank,
            "question_id": result.question_id,
            "distribution": result.distribution,
            "question": result.question,
            "avg_score": round(result.avg_score, 4),
            "worst_metric": result.worst_metric,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    return output


def cluster_analysis(results: list[RagasResult]) -> dict:
    """Build a worst_metric x distribution failure matrix."""
    distributions = ["factual", "multi_hop", "adversarial"]
    matrix = {metric: {dist: 0 for dist in distributions} for metric in DIAGNOSTIC_TREE}

    for result in results:
        for metric_counts in matrix.values():
            metric_counts.setdefault(result.distribution, 0)
        matrix[result.worst_metric][result.distribution] += 1

    all_distributions = sorted({r.distribution for r in results} | set(distributions))
    if not results:
        return {
            "matrix": matrix,
            "dominant_failure_distribution": "",
            "dominant_failure_metric": "",
            "insight": "Khong co ket qua RAGAS de phan tich.",
        }

    dominant_distribution = max(
        all_distributions,
        key=lambda dist: sum(metric_counts.get(dist, 0) for metric_counts in matrix.values()),
    )
    dominant_metric = max(matrix, key=lambda metric: sum(matrix[metric].values()))
    insight = (
        f"Distribution '{dominant_distribution}' co nhieu failure nhat; "
        f"metric '{dominant_metric}' la diem yeu chinh. "
        f"Suggested fix: {DIAGNOSTIC_TREE[dominant_metric][1]}."
    )
    return {
        "matrix": matrix,
        "dominant_failure_distribution": dominant_distribution,
        "dominant_failure_metric": dominant_metric,
        "insight": insight,
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def save_phase_a_report(
    results: list[RagasResult],
    clusters: dict,
    path: str = "reports/ragas_50q.json",
) -> None:
    """Save Phase A report to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    per_distribution: dict[str, dict] = {}
    for distribution in ["factual", "multi_hop", "adversarial"]:
        subset = [r for r in results if r.distribution == distribution]
        if not subset:
            continue
        per_distribution[distribution] = {
            "count": len(subset),
            "faithfulness": _mean([r.faithfulness for r in subset]),
            "answer_relevancy": _mean([r.answer_relevancy for r in subset]),
            "context_precision": _mean([r.context_precision for r in subset]),
            "context_recall": _mean([r.context_recall for r in subset]),
            "avg_score": _mean([r.avg_score for r in subset]),
        }

    report = {
        "total_questions": len(results),
        "per_distribution": per_distribution,
        "failure_clusters": clusters,
        "bottom_10": bottom_10(results),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase A report saved -> {path}")


if __name__ == "__main__":
    test_set = load_test_set_50q()
    print(f"Loaded {len(test_set)} questions")

    groups = group_by_distribution(test_set)
    for distribution, questions in groups.items():
        print(f"  {distribution}: {len(questions)} questions")

    answers = load_answers()
    results = run_ragas_50q(answers)
    if not results:
        print("No RAGAS results generated.")
        sys.exit(1)

    clusters = cluster_analysis(results)
    save_phase_a_report(results, clusters)

    print("\nBottom 10 worst questions:")
    for item in bottom_10(results):
        print(
            f"  #{item['rank']} [{item['distribution']}] "
            f"avg={item['avg_score']:.3f} worst={item['worst_metric']} "
            f"- {item['question'][:60]}..."
        )
    print(
        "\nDominant failure: "
        f"{clusters.get('dominant_failure_distribution')} / "
        f"{clusters.get('dominant_failure_metric')}"
    )
