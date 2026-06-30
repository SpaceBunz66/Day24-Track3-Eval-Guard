from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re, math
from collections import Counter
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    sentences = _split_sentences(text)
    if not sentences:
        return []

    embeddings = _semantic_embeddings(sentences)

    groups: list[list[str]] = [[sentences[0]]]
    for i, sentence in enumerate(sentences[1:], start=1):
        current_text = " ".join(groups[-1])
        sim = _cosine(embeddings[i - 1], embeddings[i])

        # Avoid header-only/very tiny chunks while still splitting topic shifts.
        should_split = sim < threshold and len(current_text) >= 120
        if should_split:
            groups.append([sentence])
        else:
            groups[-1].append(sentence)

    return [
        Chunk(
            text=" ".join(group).strip(),
            metadata={**metadata, "strategy": "semantic", "chunk_index": i},
        )
        for i, group in enumerate(groups)
        if " ".join(group).strip()
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    parent_texts = _pack_paragraphs(text, parent_size)

    parents: list[Chunk] = []
    children: list[Chunk] = []
    source = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(metadata.get("source", "doc"))).strip("_") or "doc"

    for parent_index, parent_text in enumerate(parent_texts):
        pid = f"{source}_parent_{parent_index}"
        parent_meta = {
            **metadata,
            "strategy": "hierarchical",
            "chunk_type": "parent",
            "chunk_index": parent_index,
            "parent_id": pid,
        }
        parents.append(Chunk(text=parent_text, metadata=parent_meta))

        child_texts = _pack_paragraphs(parent_text, child_size)
        for child_index, child_text in enumerate(child_texts):
            child_meta = {
                **metadata,
                "strategy": "hierarchical",
                "chunk_type": "child",
                "chunk_index": len(children),
                "child_index": child_index,
                "parent_id": pid,
            }
            children.append(Chunk(text=child_text, metadata=child_meta, parent_id=pid))

    return (parents, children)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    chunks: list[Chunk] = []
    current_header = ""
    current_lines: list[str] = []

    def flush() -> None:
        section_text = "\n".join(current_lines).strip()
        if not section_text:
            return
        section_name = re.sub(r"^#{1,6}\s*", "", current_header).strip() or "preamble"
        chunks.append(Chunk(
            text=section_text,
            metadata={
                **metadata,
                "strategy": "structure",
                "section": section_name,
                "chunk_index": len(chunks),
            },
        ))

    for line in text.splitlines():
        if re.match(r"^#{1,6}\s+.+", line):
            flush()
            current_header = line.strip()
            current_lines = [current_header]
        else:
            if not current_lines and line.strip():
                current_header = "preamble"
            current_lines.append(line)

    flush()

    if not chunks and text.strip():
        return [Chunk(
            text=text.strip(),
            metadata={**metadata, "strategy": "structure", "section": "document", "chunk_index": 0},
        )]
    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split Vietnamese/Markdown text into sentence-like units."""
    parts = re.split(r"(?<=[.!?。])\s+|\n{2,}", text)
    sentences = []
    for part in parts:
        cleaned = re.sub(r"\s+", " ", part).strip()
        if cleaned:
            sentences.append(cleaned)
    return sentences


def _token_counter(text: str) -> Counter:
    tokens = re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)
    return Counter(tokens)


def _counter_cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    return dot / (norm_a * norm_b + 1e-9)


def _cosine(a, b) -> float:
    if isinstance(a, Counter) and isinstance(b, Counter):
        return _counter_cosine(a, b)

    try:
        from numpy import dot
        from numpy.linalg import norm
        return float(dot(a, b) / (norm(a) * norm(b) + 1e-9))
    except Exception:
        return 0.0


def _semantic_embeddings(sentences: list[str]):
    model_name = "all-MiniLM-L6-v2"
    if _model_available_locally(model_name):
        old_hf_offline = os.environ.get("HF_HUB_OFFLINE")
        old_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name, local_files_only=True)
            return model.encode(sentences)
        except Exception:
            pass
        finally:
            _restore_env("HF_HUB_OFFLINE", old_hf_offline)
            _restore_env("TRANSFORMERS_OFFLINE", old_transformers_offline)
    return [_token_counter(sentence) for sentence in sentences]


def _split_long_text(text: str, max_size: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_size and current:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _pack_paragraphs(text: str, max_size: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs and text.strip():
        paragraphs = [text.strip()]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        paragraph_parts = _split_long_text(paragraph, max_size) if len(paragraph) > max_size else [paragraph]
        for part in paragraph_parts:
            candidate = f"{current}\n\n{part}".strip() if current else part
            if len(candidate) > max_size and current:
                chunks.append(current)
                current = part
            else:
                current = candidate

    if current:
        chunks.append(current)
    return chunks


def _model_available_locally(model_name: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.file_download import _CACHED_NO_EXIST
        cached = try_to_load_from_cache(model_name, "config.json")
        return cached not in (None, _CACHED_NO_EXIST)
    except Exception:
        safe_name = model_name.replace("/", "--")
        cache_root = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        return os.path.isdir(os.path.join(cache_root, f"models--{safe_name}"))


def _restore_env(name: str, old_value: str | None) -> None:
    if old_value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = old_value


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
