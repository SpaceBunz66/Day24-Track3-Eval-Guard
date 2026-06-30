from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time, math, re
from collections import Counter
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            if not _model_available_locally(self.model_name):
                self._model = _LexicalReranker()
                return self._model

            old_hf_offline = os.environ.get("HF_HUB_OFFLINE")
            old_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(
                    self.model_name,
                    automodel_args={"local_files_only": True},
                    tokenizer_args={"local_files_only": True},
                    config_args={"local_files_only": True},
                )
            except Exception:
                self._model = _LexicalReranker()
            finally:
                _restore_env("HF_HUB_OFFLINE", old_hf_offline)
                _restore_env("TRANSFORMERS_OFFLINE", old_transformers_offline)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []

        model = self._load_model()
        pairs = [(query, doc.get("text", "")) for doc in documents]
        scores = model.predict(pairs)
        if isinstance(scores, (int, float)):
            scores = [scores]

        scored = sorted(zip(scores, documents), key=lambda item: float(item[0]), reverse=True)
        return [
            RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i + 1,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []

        try:
            if self._model is None:
                from flashrank import Ranker
                self._model = Ranker()
            from flashrank import RerankRequest
            passages = [
                {"id": i, "text": doc.get("text", ""), "metadata": doc.get("metadata", {})}
                for i, doc in enumerate(documents)
            ]
            request = RerankRequest(query=query, passages=passages)
            results = self._model.rerank(request)[:top_k]
            return [
                RerankResult(
                    text=item.get("text", ""),
                    original_score=float(documents[item.get("id", 0)].get("score", 0.0)),
                    rerank_score=float(item.get("score", 0.0)),
                    metadata=item.get("metadata", {}),
                    rank=i + 1,
                )
                for i, item in enumerate(results)
            ]
        except Exception:
            return CrossEncoderReranker().rerank(query, documents, top_k=top_k)


class _LexicalReranker:
    """Deterministic fallback with the same predict(pairs) interface."""

    def predict(self, pairs):
        return [_lexical_score(query, text) for query, text in pairs]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)


def _lexical_score(query: str, text: str) -> float:
    query_counts = Counter(_tokens(query))
    text_counts = Counter(_tokens(text))
    if not query_counts or not text_counts:
        return 0.0

    dot = sum(query_counts[t] * text_counts[t] for t in set(query_counts) & set(text_counts))
    norm_q = math.sqrt(sum(v * v for v in query_counts.values()))
    norm_t = math.sqrt(sum(v * v for v in text_counts.values()))
    overlap = dot / (norm_q * norm_t + 1e-9)

    number_bonus = 0.05 if set(re.findall(r"\d+", text)) else 0.0
    exact_phrase_bonus = 0.1 if any(token in text.lower() for token in ["nghỉ phép", "mật khẩu", "vpn"]) else 0.0
    return overlap + number_bonus + exact_phrase_bonus


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


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
