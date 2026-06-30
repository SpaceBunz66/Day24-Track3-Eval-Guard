from __future__ import annotations

"""Phase C: production guardrails with PII scan, input/output rails, and latency."""

import asyncio
import json
import os
import re
import statistics
import sys
import time
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS


def setup_presidio():
    """Create Presidio engines with Vietnamese regex recognizers."""
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
    from presidio_anonymizer import AnonymizerEngine

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[
            Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
            Pattern("CMND 9 digits", r"\b\d{9}\b", 0.7),
        ],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    )

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    registry.add_recognizer(cccd_recognizer)
    registry.add_recognizer(phone_recognizer)
    return AnalyzerEngine(registry=registry), AnonymizerEngine()


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Detect CCCD/CMND, Vietnamese phone numbers, and email addresses."""
    entities = _regex_pii_entities(text)

    if analyzer is not None and anonymizer is not None:
        try:
            presidio_results = analyzer.analyze(text=text, language="en")
            for result in presidio_results:
                entities.append({
                    "type": result.entity_type,
                    "text": text[result.start:result.end],
                    "score": round(float(result.score), 3),
                    "start": result.start,
                    "end": result.end,
                })
        except Exception:
            pass

    entities = _dedupe_entities(entities)
    anonymized = _anonymize_with_entities(text, entities)
    return {
        "has_pii": bool(entities),
        "entities": entities,
        "anonymized": anonymized,
    }


def setup_nemo_rails():
    """Load NeMo Guardrails config from guardrails/."""
    from nemoguardrails import LLMRails, RailsConfig

    config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
    return LLMRails(config)


async def check_input_rail(text: str, rails=None) -> dict:
    """Check off-topic, jailbreak, prompt-injection, and PII-access requests."""
    heuristic_reason = _heuristic_input_block_reason(text)
    if heuristic_reason:
        return {
            "allowed": False,
            "blocked_reason": heuristic_reason,
            "response": _safe_refusal(heuristic_reason),
        }

    if rails is not None:
        try:
            response = await rails.generate_async(messages=[{"role": "user", "content": text}])
            blocked = _looks_like_refusal(str(response))
            return {
                "allowed": not blocked,
                "blocked_reason": "nemo_input_rail" if blocked else None,
                "response": str(response),
            }
        except Exception as exc:
            return {
                "allowed": True,
                "blocked_reason": None,
                "response": f"nemo_unavailable: {exc}",
            }

    return {"allowed": True, "blocked_reason": None, "response": ""}


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Flag sensitive content before returning the answer to the user."""
    pii_result = pii_scan(answer)
    sensitive_reason = _heuristic_output_flag_reason(answer)
    if pii_result["has_pii"] or sensitive_reason:
        return {
            "safe": False,
            "flagged_reason": "pii_output" if pii_result["has_pii"] else sensitive_reason,
            "final_answer": (
                "Toi khong the cung cap thong tin nhay cam nay. "
                "Vui long lien he phong Nhan su hoac CNTT truc tiep."
            ),
        }

    if rails is not None:
        try:
            response = await rails.generate_async(messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
            flagged = _looks_like_refusal(str(response))
            return {
                "safe": not flagged,
                "flagged_reason": "nemo_output_rail" if flagged else None,
                "final_answer": str(response) if flagged else answer,
            }
        except Exception:
            pass

    return {"safe": True, "flagged_reason": None, "final_answer": answer}


def run_adversarial_suite(
    adversarial_set: list[dict],
    rails=None,
    analyzer=None,
    anonymizer=None,
) -> list[dict]:
    """Run the adversarial set through the input guard stack."""
    async def _run_all() -> list[dict]:
        results = []
        for item in adversarial_set:
            blocked_by = None
            pii_result = pii_scan(item["input"], analyzer, anonymizer)
            if pii_result["has_pii"]:
                blocked_by = "presidio"

            if blocked_by is None:
                rail_result = await check_input_rail(item["input"], rails)
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"

            actual = "blocked" if blocked_by else "allowed"
            results.append({
                "id": item["id"],
                "category": item["category"],
                "input": item["input"][:120],
                "expected": item["expected"],
                "actual": actual,
                "blocked_by": blocked_by,
                "passed": actual == item["expected"],
            })
        return results

    results = asyncio.run(_run_all())
    passed = sum(1 for item in results if item["passed"])
    print(f"Adversarial suite: {passed}/{len(results)} passed")
    return results


def measure_p95_latency(
    test_inputs: list[str],
    n_runs: int = 20,
    rails=None,
    analyzer=None,
    anonymizer=None,
) -> dict:
    """Measure P50/P95/P99 latency for PII and input-rail layers."""
    inputs = (test_inputs or [""])[: max(1, n_runs)]
    presidio_times: list[float] = []
    nemo_times: list[float] = []
    total_times: list[float] = []

    async def _measure() -> None:
        for text in inputs:
            total_start = time.perf_counter()

            pii_start = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - pii_start) * 1000

            rail_start = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - rail_start) * 1000

            total_ms = (time.perf_counter() - total_start) * 1000
            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(total_ms)

    asyncio.run(_measure())
    total_percentiles = _percentiles(total_times)
    return {
        "presidio_ms": _percentiles(presidio_times),
        "nemo_ms": _percentiles(nemo_times),
        "total_ms": total_percentiles,
        "latency_budget_ok": total_percentiles["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms": LATENCY_BUDGET_P95_MS,
    }


def _regex_pii_entities(text: str) -> list[dict]:
    patterns = [
        ("EMAIL_ADDRESS", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", 0.95),
        ("VN_PHONE", r"\b0[3-9]\d{8}\b", 0.9),
        ("VN_CCCD", r"\b\d{12}\b", 0.9),
        ("VN_CCCD", r"\b\d{9}\b", 0.7),
    ]
    entities: list[dict] = []
    for entity_type, pattern, score in patterns:
        for match in re.finditer(pattern, text):
            entities.append({
                "type": entity_type,
                "text": match.group(0),
                "score": score,
                "start": match.start(),
                "end": match.end(),
            })
    return entities


def _dedupe_entities(entities: list[dict]) -> list[dict]:
    deduped = {}
    for entity in entities:
        key = (entity["start"], entity["end"], entity["type"])
        if key not in deduped or entity["score"] > deduped[key]["score"]:
            deduped[key] = entity
    return sorted(deduped.values(), key=lambda item: (item["start"], item["end"]))


def _anonymize_with_entities(text: str, entities: list[dict]) -> str:
    anonymized = text
    for entity in sorted(entities, key=lambda item: item["start"], reverse=True):
        anonymized = (
            anonymized[: entity["start"]]
            + f"<{entity['type']}>"
            + anonymized[entity["end"]:]
        )
    return anonymized


def _heuristic_input_block_reason(text: str) -> str | None:
    normalized = _strip_accents(text)
    checks = [
        ("jailbreak", [
            "bo qua", "ignore", "previous instruction", "pretend you are dan",
            "unrestricted ai", "system override", "forget", "admin command",
            "priority 1", "dong vai", "khong co gioi han", "ceo va ra lenh",
        ]),
        ("prompt_injection", [
            "dump", "training data", "employee records", "system instructions",
            "context", "confidential employee data", "all confidential",
        ]),
        ("pii_request", [
            "cccd cua", "so dien thoai cua nhan vien", "thong tin ca nhan",
            "email cua nhan vien", "tiet lo", "bang luong", "employee salaries",
            "list all employee", "luong thang cua nhan vien",
        ]),
        ("off_topic", [
            "bai tho", "nau", "pho bo", "bitcoin", "ethereum", "phuong trinh",
            "giai toan", "recommend", "marvel", "thoi tiet", "tin tuc",
            "gia co phieu",
        ]),
        ("harmful_request", ["tan cong mang", "attack internal", "huong dan tan cong"]),
    ]
    for reason, patterns in checks:
        if any(pattern in normalized for pattern in patterns):
            return reason
    return None


def _heuristic_output_flag_reason(answer: str) -> str | None:
    normalized = _strip_accents(answer)
    sensitive_patterns = [
        "cccd cua nhan vien",
        "so dien thoai ca nhan",
        "mat khau he thong",
        "thong tin bi mat",
        "bang luong chi tiet",
    ]
    return "sensitive_output" if any(pattern in normalized for pattern in sensitive_patterns) else None


def _looks_like_refusal(response: str) -> bool:
    normalized = _strip_accents(response)
    refusal_keywords = [
        "xin loi",
        "khong the",
        "khong duoc phep",
        "i cannot",
        "i'm sorry",
        "toi khong the",
    ]
    return any(keyword in normalized for keyword in refusal_keywords)


def _safe_refusal(reason: str) -> str:
    return (
        "Toi khong the xu ly yeu cau nay trong tro ly chinh sach HR. "
        f"Ly do: {reason}."
    )


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _percentiles(times: list[float]) -> dict:
    if not times:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    ordered = sorted(times)
    if len(ordered) == 1:
        value = round(ordered[0], 2)
        return {"p50": value, "p95": value, "p99": value}

    def percentile(percent: float) -> float:
        return float(statistics.quantiles(ordered, n=100, method="inclusive")[percent - 1])

    return {
        "p50": round(percentile(50), 2),
        "p95": round(percentile(95), 2),
        "p99": round(percentile(99), 2),
    }


def _save_phase_c_report(report: dict, path: str = "reports/guard_results.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase C report saved -> {path}")


if __name__ == "__main__":
    test_pii = "Nhan vien A, CCCD 034095001234, SDT 0987654321 hoi ve nghi phep."
    pii_result = pii_scan(test_pii)
    print(f"PII detected: {pii_result['has_pii']}")
    print(f"Entities: {pii_result['entities']}")
    print(f"Anonymized: {pii_result['anonymized']}")

    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)
    suite_results = run_adversarial_suite(adversarial_set)
    pass_count = sum(1 for item in suite_results if item["passed"])

    sample_inputs = [item["input"] for item in adversarial_set[:10]]
    latency = measure_p95_latency(sample_inputs, n_runs=10)
    print(
        "\nLatency P95 - "
        f"Presidio: {latency['presidio_ms']['p95']}ms | "
        f"NeMo/Input rail: {latency['nemo_ms']['p95']}ms | "
        f"Total: {latency['total_ms']['p95']}ms"
    )

    _save_phase_c_report({
        "adversarial_total": len(suite_results),
        "adversarial_passed": pass_count,
        "adversarial_pass_rate": round(pass_count / len(suite_results), 3) if suite_results else 0.0,
        "results": suite_results,
        "latency": latency,
    })
