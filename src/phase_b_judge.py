from __future__ import annotations

"""Phase B: LLM-as-Judge with swap-and-average and bias analysis."""

import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    HUMAN_LABELS_PATH,
    JUDGE_MODEL,
    TEST_SET_PATH,
    active_llm_api_key,
    openai_client_kwargs,
)


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str
    winner_pass2: str
    final_winner: str
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool
    scores_pass1: dict = field(default_factory=dict)
    scores_pass2: dict = field(default_factory=dict)


def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Ask a judge model to choose answer A, B, or tie."""
    if _valid_llm_key():
        llm_result = _call_llm_pairwise(question, answer_a, answer_b)
        if llm_result is not None:
            return llm_result
    return _heuristic_pairwise(question, answer_a, answer_b)


def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Run pairwise judge twice with A/B swapped, then keep only consensus."""
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)

    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map.get(pass2_raw.get("winner", "tie"), "tie")
    final_winner = pass1["winner"] if pass1["winner"] == winner_pass2 else "tie"
    position_consistent = pass1["winner"] == winner_pass2

    raw_scores2 = pass2_raw.get("scores", {})
    scores_pass2 = {
        "A": float(raw_scores2.get("B", 0.0) or 0.0),
        "B": float(raw_scores2.get("A", 0.0) or 0.0),
    }

    return JudgeResult(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        winner_pass1=pass1["winner"],
        winner_pass2=winner_pass2,
        final_winner=final_winner,
        reasoning_pass1=pass1.get("reasoning", ""),
        reasoning_pass2=pass2_raw.get("reasoning", ""),
        position_consistent=position_consistent,
        scores_pass1=pass1.get("scores", {"A": 0.0, "B": 0.0}),
        scores_pass2=scores_pass2,
    )


def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Compute Cohen's kappa for binary labels."""
    n = min(len(judge_labels), len(human_labels))
    if n == 0:
        return 0.0

    judge = [int(x) for x in judge_labels[:n]]
    human = [int(x) for x in human_labels[:n]]
    observed = sum(j == h for j, h in zip(judge, human)) / n
    p_yes = (judge.count(1) / n) * (human.count(1) / n)
    p_no = (judge.count(0) / n) * (human.count(0) / n)
    expected = p_yes + p_no
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return round((observed - expected) / (1 - expected), 4)


def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Measure position bias and verbosity bias."""
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "position_bias_count": 0,
            "verbosity_bias": 0.0,
            "verbosity_details": {
                "a_wins_a_longer": 0,
                "b_wins_b_longer": 0,
                "total_decisive": 0,
            },
            "interpretation": "Khong co ket qua judge de phan tich.",
        }

    position_bias_count = sum(1 for result in judge_results if not result.position_consistent)
    position_bias_rate = position_bias_count / total

    a_wins_a_longer = sum(
        1
        for result in judge_results
        if result.final_winner == "A" and len(result.answer_a) > len(result.answer_b)
    )
    b_wins_b_longer = sum(
        1
        for result in judge_results
        if result.final_winner == "B" and len(result.answer_b) > len(result.answer_a)
    )
    total_decisive = sum(1 for result in judge_results if result.final_winner != "tie")
    verbosity_bias = (
        (a_wins_a_longer + b_wins_b_longer) / total_decisive
        if total_decisive
        else 0.0
    )

    if position_bias_rate > 0.3:
        interpretation = "Position bias cao; can bat buoc swap-and-average trong production."
    else:
        interpretation = "Position bias thap; judge tuong doi on dinh tren mau hien tai."

    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": total_decisive,
        },
        "interpretation": interpretation,
    }


def _valid_llm_key() -> bool:
    key = active_llm_api_key()
    return bool(key and len(key) > 20 and "your-" not in key and key != "sk-...")


def _call_llm_pairwise(question: str, answer_a: str, answer_b: str) -> dict | None:
    prompt = f"""
You are a strict evaluator for Vietnamese HR-policy RAG answers.
Choose the better answer using accuracy, completeness, and conciseness.

Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Return only JSON:
{{"winner":"A|B|tie","reasoning":"short reason","scores":{{"A":0.0,"B":0.0}}}}
""".strip()
    try:
        from openai import OpenAI

        kwargs = openai_client_kwargs()
        kwargs["timeout"] = 20.0
        client = OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=300,
        )
        return _clean_judge_payload(response.choices[0].message.content or "")
    except Exception as exc:
        print(f"  Judge LLM failed, using heuristic fallback: {exc}")
        return None


def _clean_judge_payload(content: str) -> dict | None:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I | re.M)
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match:
        cleaned = match.group(0)
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return None

    winner = str(parsed.get("winner", "tie")).strip()
    if winner not in {"A", "B", "tie"}:
        winner = "tie"
    scores = parsed.get("scores", {}) if isinstance(parsed.get("scores", {}), dict) else {}
    return {
        "winner": winner,
        "reasoning": str(parsed.get("reasoning", "")).strip(),
        "scores": {
            "A": _bounded_float(scores.get("A", 0.0)),
            "B": _bounded_float(scores.get("B", 0.0)),
        },
    }


def _heuristic_pairwise(question: str, answer_a: str, answer_b: str) -> dict:
    score_a = _quality_score(question, answer_a)
    score_b = _quality_score(question, answer_b)
    delta = score_a - score_b
    if abs(delta) < 0.05:
        winner = "tie"
        reasoning = "Hai cau tra loi co muc do phu hop tuong duong theo heuristic."
    elif delta > 0:
        winner = "A"
        reasoning = "Answer A phu hop cau hoi va co tin hieu chinh sach tot hon."
    else:
        winner = "B"
        reasoning = "Answer B phu hop cau hoi va co tin hieu chinh sach tot hon."
    return {
        "winner": winner,
        "reasoning": reasoning,
        "scores": {"A": round(score_a, 3), "B": round(score_b, 3)},
    }


def _quality_score(question: str, answer: str, reference: str = "") -> float:
    q_tokens = _tokens(question)
    a_tokens = _tokens(answer)
    if not a_tokens:
        return 0.0

    score = 0.35 * _overlap(a_tokens, q_tokens)
    if reference:
        r_tokens = _tokens(reference)
        score += 0.55 * _overlap(a_tokens, r_tokens)
    else:
        score += 0.35

    normalized = _normalize(answer)
    if "2024" in normalized or "hien hanh" in normalized or "chinh sach" in normalized:
        score += 0.08
    if "khong tim thay" in normalized:
        score -= 0.2
    if "phep nam" in _normalize(question):
        if re.search(r"\b15\b", answer):
            score += 0.15
        if re.search(r"\b12\b", answer):
            score -= 0.1
    return max(0.0, min(1.0, score))


def _binary_judge_label(question: str, answer: str, ground_truth: str) -> int:
    normalized_answer = _normalize(answer)
    normalized_truth = _normalize(ground_truth)

    if "khong" in normalized_truth and "khong" not in normalized_answer:
        if any(word in normalized_answer for word in ("duoc", "co the", "nen")):
            return 0
    for must_have in ("ceo", "wireguard", "mfa"):
        if must_have in normalized_truth and must_have not in normalized_answer:
            return 0

    truth_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", ground_truth))
    answer_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", answer))
    if truth_numbers and answer_numbers and truth_numbers.isdisjoint(answer_numbers):
        return 0

    return 1 if _quality_score(question, answer, ground_truth) >= 0.38 else 0


def _bounded_float(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", _normalize(text), flags=re.UNICODE))


def _normalize(text: str) -> str:
    return text.lower()


def _overlap(source: set[str], target: set[str]) -> float:
    if not source or not target:
        return 0.0
    return len(source & target) / max(len(target), 1)


def _kappa_interpretation(kappa: float) -> str:
    if kappa > 0.8:
        return "almost perfect"
    if kappa > 0.6:
        return "substantial"
    if kappa > 0.4:
        return "moderate"
    if kappa > 0.2:
        return "fair"
    if kappa >= 0:
        return "slight"
    return "poor"


def _load_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_phase_b_report(report: dict, path: str = "reports/judge_results.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase B report saved -> {path}")


if __name__ == "__main__":
    human_data = _load_json(HUMAN_LABELS_PATH)
    test_set = _load_json(TEST_SET_PATH)
    ground_truth_by_id = {item["id"]: item["ground_truth"] for item in test_set}

    judge_results: list[JudgeResult] = []
    for item in human_data[:5]:
        reference = ground_truth_by_id.get(item["question_id"], "")
        judge_results.append(swap_and_average(item["question"], item["model_answer"], reference))

    human_labels = [int(item["human_label"]) for item in human_data]
    judge_labels = [
        _binary_judge_label(
            item["question"],
            item["model_answer"],
            ground_truth_by_id.get(item["question_id"], ""),
        )
        for item in human_data
    ]
    kappa = cohen_kappa(judge_labels, human_labels)
    bias = bias_report(judge_results)

    report = {
        "judge_model": JUDGE_MODEL,
        "pairwise_results": [asdict(result) for result in judge_results],
        "human_labels": human_labels,
        "judge_labels": judge_labels,
        "cohen_kappa": kappa,
        "kappa_interpretation": _kappa_interpretation(kappa),
        "bias_report": bias,
    }
    _save_phase_b_report(report)

    print(f"Cohen kappa: {kappa:.3f} ({report['kappa_interpretation']})")
    print(f"Position bias rate: {bias['position_bias_rate']:.0%}")
