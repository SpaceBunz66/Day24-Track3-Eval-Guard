# LLM Judge Bias Report - Phase B

**Sinh viên:** Nguyễn Thái Hoàng  
**Ngày:** 30/06/2026  
**Judge model:** openrouter/free

---

## 1. Pairwise Judge Results

| # | Question (tóm tắt) | Winner | Reasoning tóm tắt |
|---:|---|---|---|
| 1 | Nghỉ khi kết hôn | tie | Hai câu tương đương, reference chỉ đầy đủ hơn một chi tiết nhỏ |
| 2 | Mua thiết bị 55 triệu | tie | Heuristic chưa phân biệt tốt Director và CEO trong pairwise |
| 3 | Thưởng Tết tối thiểu | B | Reference đầy đủ hơn vì nêu điều kiện 6 tháng và pro-rata |
| 4 | Senior 9 năm: phép năm và lương | B | Reference nêu rõ v2024, công thức thâm niên và band lương |
| 5 | Hoàn trả khóa học 25 triệu | B | Reference đầy đủ hơn vì nêu 100% chi phí và thời hạn cam kết |

---

## 2. Swap-and-Average Results

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---:|---|---|---|---|
| 1 | tie | tie | tie | Yes |
| 2 | tie | tie | tie | Yes |
| 3 | B | B | B | Yes |
| 4 | B | B | B | Yes |
| 5 | B | B | B | Yes |

**Position bias rate:** 0% (= 0/5 case không nhất quán)

---

## 3. Cohen's Kappa Analysis

**Human labels:** `human_labels_10q.json`  
**Judge labels:** `[1, 0, 0, 1, 0, 0, 0, 0, 1, 0]`

| Question ID | Human Label | Judge Label | Agree? |
|---:|---:|---:|---|
| 1 | 1 | 1 | Yes |
| 5 | 0 | 0 | Yes |
| 12 | 1 | 0 | No |
| 21 | 1 | 1 | Yes |
| 23 | 1 | 0 | No |
| 29 | 0 | 0 | Yes |
| 33 | 1 | 0 | No |
| 41 | 0 | 0 | Yes |
| 46 | 1 | 1 | Yes |
| 50 | 0 | 0 | Yes |

**Cohen's kappa:** 0.444  
**Interpretation:** moderate

---

## 4. Verbosity Bias

Trong các case có winner rõ ràng:

- A thắng + A dài hơn B: 0 / 3 cases
- B thắng + B dài hơn A: 3 / 3 cases
- **Verbosity bias rate:** 100%

**Kết luận:** Judge đang có xu hướng chọn câu dài hơn trong các case quyết định, vì answer B thường là ground-truth/reference đầy đủ hơn. Đây chưa chắc là bias xấu trong lab này, nhưng trong production cần tách tiêu chí "đầy đủ" khỏi "dài" bằng rubric điểm rõ ràng: đúng số liệu, đúng policy version, đủ điều kiện áp dụng, rồi mới xét độ súc tích.

---

## 5. Nhận xét chung

Kappa 0.444 mới ở mức moderate, chưa đủ ngưỡng substantial > 0.6. Position bias thấp nhờ swap-and-average, nhưng verbosity bias cao do judge ưu tiên câu trả lời dài/reference. Với production, nên dùng judge model thật qua OpenRouter/OpenAI ở nhiệt độ 0, yêu cầu JSON schema, chạy swap bắt buộc và audit định kỳ trên human labels. Những câu tính toán hoặc version conflict cần rubric riêng thay vì chỉ dùng overlap/heuristic.
