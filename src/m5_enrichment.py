from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, json, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEFAULT_CHAT_MODEL, active_llm_api_key, openai_client_kwargs


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if _valid_openai_key():
        try:
            from openai import OpenAI
            client = OpenAI(**openai_client_kwargs())
            resp = client.chat.completions.create(
                model=DEFAULT_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠️  OpenAI summarize failed: {e}")

    return _extractive_summary(text)


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if _valid_openai_key():
        try:
            from openai import OpenAI
            client = OpenAI(**openai_client_kwargs())
            resp = client.chat.completions.create(
                model=DEFAULT_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng."},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            questions = resp.choices[0].message.content.strip().split("\n")
            cleaned = [_clean_question(q) for q in questions if q.strip()]
            return [q for q in cleaned if q][:n_questions]
        except Exception as e:
            print(f"  ⚠️  OpenAI HyQA failed: {e}")

    return _fallback_questions(text, n_questions)


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if _valid_openai_key():
        try:
            from openai import OpenAI
            client = OpenAI(**openai_client_kwargs())
            resp = client.chat.completions.create(
                model=DEFAULT_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu."},
                    {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=80,
            )
            context = resp.choices[0].message.content.strip()
            return f"{context}\n\n{text}"
        except Exception as e:
            print(f"  ⚠️  OpenAI contextual failed: {e}")

    context = _fallback_context(text, document_title)
    return f"{context}\n\n{text}" if context else text


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if _valid_openai_key():
        try:
            from openai import OpenAI
            client = OpenAI(**openai_client_kwargs())
            resp = client.chat.completions.create(
                model=DEFAULT_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            return _parse_json(resp.choices[0].message.content)
        except Exception as e:
            print(f"  ⚠️  OpenAI metadata failed: {e}")

    return _fallback_metadata(text)


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    if _valid_openai_key():
        try:
            from openai import OpenAI
            client = OpenAI(**openai_client_kwargs())
            resp = client.chat.completions.create(
                model=DEFAULT_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": """Phân tích đoạn văn và trả về JSON:
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}"""},
                    {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=400,
            )
            parsed = _parse_json(resp.choices[0].message.content)
            if parsed:
                return parsed
        except Exception as e:
            print(f"  ⚠️  Enrichment API failed: {e}")

    return {
        "summary": _extractive_summary(text),
        "questions": _fallback_questions(text, 3),
        "context": _fallback_context(text, source),
        "metadata": _fallback_metadata(text),
    }


def _valid_openai_key() -> bool:
    key = active_llm_api_key()
    return bool(key and key != "sk-..." and "your-" not in key and len(key) > 20)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 0]


def _extractive_summary(text: str) -> str:
    sentences = _sentences(text)
    if not sentences:
        return text.strip()
    summary = ". ".join(s.rstrip(".") for s in sentences[:2]).strip()
    return f"{summary}." if summary and not summary.endswith((".", "!", "?")) else summary


def _clean_question(question: str) -> str:
    cleaned = question.strip().lstrip("0123456789.-)• ").strip()
    return cleaned if cleaned.endswith("?") else f"{cleaned}?"


def _fallback_questions(text: str, n_questions: int) -> list[str]:
    lowered = text.lower()
    questions: list[str] = []

    if "nghỉ phép" in lowered or "nghỉ" in lowered:
        questions.append("Quy định nghỉ phép trong tài liệu là gì?")
    if "mật khẩu" in lowered or "password" in lowered:
        questions.append("Chính sách mật khẩu yêu cầu những gì?")
    if "lương" in lowered or "phụ cấp" in lowered:
        questions.append("Quy định về lương hoặc phụ cấp là gì?")
    if "vpn" in lowered or "mfa" in lowered or "bảo mật" in lowered:
        questions.append("Quy định bảo mật hoặc truy cập hệ thống là gì?")

    for sentence in _sentences(text):
        if len(questions) >= n_questions:
            break
        normalized = sentence.rstrip(".:; ")
        if len(normalized) > 10:
            questions.append(f"{normalized}?")

    return questions[:n_questions]


def _fallback_context(text: str, document_title: str = "") -> str:
    topic = _fallback_metadata(text).get("topic", "chính sách nội bộ")
    if document_title:
        return f"Trích từ {document_title}, đoạn này nói về {topic}."
    return f"Đoạn này nói về {topic}."


def _fallback_metadata(text: str) -> dict:
    lowered = text.lower()
    category = "policy"
    topic = "chính sách nội bộ"

    rules = [
        (("nghỉ phép", "nghỉ ốm", "nghỉ không lương"), "hr", "nghỉ phép"),
        (("mật khẩu", "vpn", "mfa", "bảo mật"), "it", "bảo mật và truy cập hệ thống"),
        (("lương", "phụ cấp", "công tác phí", "chi phí"), "finance", "tài chính và phúc lợi"),
        (("đào tạo", "hiệu suất", "onboarding"), "hr", "đào tạo và nhân sự"),
        (("an toàn", "pccc", "sơ cứu"), "policy", "an toàn lao động"),
    ]
    for keywords, cat, detected_topic in rules:
        if any(keyword in lowered for keyword in keywords):
            category = cat
            topic = detected_topic
            break

    entities = sorted(set(re.findall(r"\b[A-ZĐ][\wÀ-ỹ.-]*\b|\b\d{2,4}\b", text, flags=re.UNICODE)))[:8]
    return {
        "topic": topic,
        "entities": entities,
        "category": category,
        "language": "vi",
    }


def _parse_json(content: str) -> dict:
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
