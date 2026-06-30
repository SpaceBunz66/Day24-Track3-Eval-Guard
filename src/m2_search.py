from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys, math, re
from collections import Counter
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except Exception:
        return text.replace("_", " ")


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = [_tokenize(chunk.get("text", "")) for chunk in chunks]
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus_tokens)
        except Exception:
            self.bm25 = _SimpleBM25(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for i in top_indices:
            score = float(scores[i])
            if score <= 0:
                continue
            doc = self.documents[i]
            results.append(SearchResult(
                text=doc.get("text", ""),
                score=score,
                metadata=doc.get("metadata", {}),
                method="bm25",
            ))
        return results


class DenseSearch:
    def __init__(self):
        try:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        except Exception:
            self.client = None
        self._encoder = None
        self._memory_chunks: list[dict] = []
        self._memory_vectors = None

    def _get_encoder(self):
        if self._encoder is None:
            if not _model_available_locally(EMBEDDING_MODEL):
                self._encoder = False
                return self._encoder

            old_hf_offline = os.environ.get("HF_HUB_OFFLINE")
            old_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
            except Exception:
                self._encoder = False
            finally:
                _restore_env("HF_HUB_OFFLINE", old_hf_offline)
                _restore_env("TRANSFORMERS_OFFLINE", old_transformers_offline)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        self._memory_chunks = chunks
        texts = [c.get("text", "") for c in chunks]
        encoder = self._get_encoder()

        if encoder:
            vectors = encoder.encode(texts, show_progress_bar=False)
            self._memory_vectors = vectors
        else:
            self._memory_vectors = None

        if not self.client or not encoder:
            return

        try:
            from qdrant_client.models import Distance, VectorParams, PointStruct
            self.client.recreate_collection(
                collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            points = [
                PointStruct(
                    id=i,
                    vector=v.tolist(),
                    payload={**chunks[i].get("metadata", {}), "text": chunks[i].get("text", "")},
                )
                for i, v in enumerate(vectors)
            ]
            self.client.upsert(collection, points)
        except Exception as e:
            print(f"  ⚠️  Qdrant indexing unavailable, using in-memory dense fallback: {e}")

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        encoder = self._get_encoder()
        if encoder and self.client:
            try:
                query_vector = encoder.encode(query).tolist()
                response = self.client.query_points(collection, query=query_vector, limit=top_k)
                return [
                    SearchResult(
                        text=pt.payload.get("text", ""),
                        score=float(pt.score),
                        metadata={k: v for k, v in pt.payload.items() if k != "text"},
                        method="dense",
                    )
                    for pt in response.points
                ]
            except Exception as e:
                print(f"  ⚠️  Qdrant search unavailable, using in-memory dense fallback: {e}")

        return self._memory_search(query, top_k, encoder)

    def _memory_search(self, query: str, top_k: int, encoder) -> list[SearchResult]:
        if not self._memory_chunks:
            return []

        if encoder and self._memory_vectors is not None:
            query_vector = encoder.encode(query)
            scores = [_vector_cosine(query_vector, vector) for vector in self._memory_vectors]
        else:
            query_tokens = Counter(_tokenize(query))
            scores = [_counter_cosine(query_tokens, Counter(_tokenize(c.get("text", ""))))
                      for c in self._memory_chunks]

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(
                text=self._memory_chunks[i].get("text", ""),
                score=float(scores[i]),
                metadata=self._memory_chunks[i].get("metadata", {}),
                method="dense",
            )
            for i in top_indices
            if scores[i] > 0
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores: dict[str, dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {"score": 0.0, "result": result}
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)

    fused = sorted(rrf_scores.values(), key=lambda item: item["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=item["result"].metadata,
            method="hybrid",
        )
        for item in fused
    ]


def _tokenize(text: str) -> list[str]:
    segmented = segment_vietnamese(text).lower()
    return re.findall(r"[\wÀ-ỹ]+", segmented, flags=re.UNICODE)


def _counter_cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b[t] for t in set(a) & set(b))
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    return dot / (norm_a * norm_b + 1e-9)


def _vector_cosine(a, b) -> float:
    try:
        import numpy as np
        a_arr = np.asarray(a)
        b_arr = np.asarray(b)
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-9))
    except Exception:
        return 0.0


class _SimpleBM25:
    """Tiny BM25 fallback used when rank-bm25 is unavailable."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus_tokens = corpus_tokens
        self.k1 = k1
        self.b = b
        self.doc_freq: Counter = Counter()
        self.doc_len = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        for doc in corpus_tokens:
            self.doc_freq.update(set(doc))
        self.n_docs = len(corpus_tokens)

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = []
        for doc, dl in zip(self.corpus_tokens, self.doc_len):
            freqs = Counter(doc)
            score = 0.0
            for token in query_tokens:
                if token not in freqs:
                    continue
                df = self.doc_freq.get(token, 0)
                idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
                tf = freqs[token]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl + 1e-9))
                score += idf * (tf * (self.k1 + 1)) / (denom + 1e-9)
            scores.append(score)
        return scores


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


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
