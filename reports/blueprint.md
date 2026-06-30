# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Nguyễn Thái Hoàng  
**Ngày:** 30/06/2026

---

## Guard Stack Architecture

```
User Input
    |
    v (~0.04ms P95)
[Presidio/Regex PII Scan]
    | block if: VN_CCCD / VN_PHONE / EMAIL detected
    | action:   reject + log sanitized query
    v (~0.06ms P95)
[Input Rail]
    | block if: off-topic / jailbreak / prompt injection / PII request
    | action:   refuse with safe reason
    v
[RAG Pipeline (Day 18)]
    | M1 Chunk -> M2 Search -> M3 Rerank -> OpenRouter free model
    v
[Output Rail]
    | flag if: PII in response / sensitive content
    | action:   replace with safe response + log incident
    v
User Response
```

---

## Guard Stack Pipeline

| Layer | Tool | Latency P95 | Failure Action |
|---|---|---:|---|
| PII Detection | Presidio-compatible regex + optional Presidio | 0.04ms | Reject + log |
| Topic/Jailbreak | Input rail heuristic + optional NeMo | 0.06ms | Refuse + reason |
| RAG Pipeline | Day 18 RAG | <2000ms target | Fallback to retrieved context |
| Output Check | Output rail heuristic + optional NeMo | <1ms local target | Block + safe response |

---

## Latency Budget

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---:|---:|---:|---|
| Presidio PII | 0.02 | 0.04 | 0.05 | <10ms |
| NeMo/Input Rail | 0.03 | 0.06 | 0.06 | <300ms |
| RAG Pipeline | Not measured in Phase C | <2000 target | Not measured | <2000ms |
| NeMo Output Rail | <1 local target | <1 local target | <1 local target | <300ms |
| **Total Guard** | 0.05 | **0.10** | 0.11 | **<500ms** |

**Budget OK?** Yes  
**Comment:** Guard stack local hiện rất nhanh vì đang dùng regex/heuristic fallback. Nếu bật NeMo/LLM rail thật, bottleneck sẽ chuyển sang network/API latency; khi đó cần cache, timeout ngắn, batch eval offline và fallback rule-based khi provider lỗi.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # yêu cầu >= 15/20 trong lab, production target >= 18/20

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; print(measure_p95_latency(['test']))"
  # P95 total guard < 500ms
```

**CI gate status hiện tại:**

- RAGAS faithfulness overall khoảng 0.505: chưa đạt gate 0.75.
- RAGAS avg_score overall khoảng 0.544: chưa đạt gate 0.65.
- Guardrail adversarial suite: đạt 20/20.
- P95 total guard latency: đạt 0.10ms < 500ms.

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness daily sample | <0.70 | Review retrieval + prompt, open quality incident |
| RAGAS context_precision | <0.60 | Tune reranker, metadata filter, top-k |
| Adversarial pass rate | <90% | Add new rail patterns and regression cases |
| Guard P95 latency | >600ms | Enable cache, reduce LLM rail calls, fallback rules |
| PII detected count | spike >10/hour | Security alert and review abuse pattern |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | 0.544 |
| Worst metric | context_precision |
| Dominant failure distribution | factual |
| Cohen's kappa | 0.444 (moderate) |
| Adversarial pass rate | 20 / 20 |
| Guard P95 latency | 0.10 ms |

---

## Nhận xét & Cải tiến

Pipeline guardrail hoạt động tốt trên adversarial suite nhờ lớp PII local và input rail chặn jailbreak/off-topic/prompt injection. Điểm yếu lớn nhất của RAG nằm ở retrieval precision: nhiều câu retrieve đúng một phần nhưng kèm nhiều context nhiễu, làm answer lệch hoặc thiếu trọng tâm. Nếu deploy production, cần ưu tiên metadata versioning cho policy hiện hành, reranker tốt hơn, và CI gate cho các câu version conflict/negation trap. LLM judge hiện mới đạt moderate agreement, nên chưa nên dùng làm nguồn chấm duy nhất mà nên kết hợp human audit định kỳ.
