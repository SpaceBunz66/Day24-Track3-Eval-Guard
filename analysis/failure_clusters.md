# Failure Cluster Analysis - Phase A

**Sinh viên:** Nguyễn Thái Hoàng  
**Ngày:** 30/06/2026

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 0.487 | 0.513 | 0.525 |
| answer_relevancy | 0.719 | 0.475 | 0.463 |
| context_precision | 0.510 | 0.369 | 0.399 |
| context_recall | 0.911 | 0.567 | 0.401 |
| **avg_score** | **0.657** | **0.481** | **0.447** |

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---:|---|---|---:|---|
| 1 | multi_hop | Nhân viên thử việc tháng thứ 3 phát hiện vi phạm bảo mật. Họ nên và không nên làm gì theo chính sách? | 0.226 | context_recall |
| 2 | adversarial | Khi phát hiện malware trên máy tính công ty, nhân viên có nên tự xử lý không? | 0.329 | answer_relevancy |
| 3 | adversarial | Nhân viên Manager có thể dùng VPN cá nhân khi WFH để tăng bảo mật thêm không? | 0.331 | answer_relevancy |
| 4 | multi_hop | Nếu cần mua laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT? | 0.337 | context_precision |
| 5 | adversarial | Bao lâu phải đổi mật khẩu một lần? | 0.371 | context_recall |
| 6 | multi_hop | Nhân viên tự ý xóa malware khỏi máy và chia sẻ thông tin sự cố trên Slack nội bộ vi phạm chính sách nào? | 0.379 | context_precision |
| 7 | multi_hop | Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu? | 0.381 | answer_relevancy |
| 8 | factual | Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt? | 0.409 | context_precision |
| 9 | multi_hop | Nhân viên Manager có thâm niên 12 năm: tổng phụ cấp hằng tháng và số ngày phép năm theo v2024 là bao nhiêu? | 0.410 | context_precision |
| 10 | multi_hop | Nhân viên tạm ứng 4 triệu và nhân viên khác tạm ứng 7 triệu: quy trình phê duyệt khác nhau thế nào? | 0.420 | answer_relevancy |

---

## 3. Failure Cluster Matrix

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 10 | 1 | 1 | 12 |
| answer_relevancy | 1 | 2 | 2 | 5 |
| context_precision | 9 | 16 | 2 | 27 |
| context_recall | 0 | 1 | 5 | 6 |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** factual  
**Dominant metric:** context_precision

**Lý do phân tích:**

Kết quả aggregate cho thấy `context_precision` là lỗi lớn nhất, xuất hiện 27/50 câu. Với factual, nhiều câu hỏi đơn giản vẫn retrieve kèm chunk không liên quan, ví dụ câu phê duyệt mua thiết bị bị kéo sang chính sách nghỉ không lương hoặc tạm ứng. Với multi-hop, lỗi này rõ hơn vì câu hỏi cần ghép nhiều tài liệu, nhưng top-k/rerank vẫn đưa nhiều ngữ cảnh nhiễu. Adversarial có avg_score thấp nhất, chủ yếu do version conflict và negation trap làm giảm `context_recall` hoặc khiến câu trả lời lệch ý.

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | Câu trả lời dùng thông tin từ context không đủ chắc hoặc trích sai policy | Bắt model trích dẫn source, giảm temperature, ưu tiên policy hiện hành |
| context_recall | Thiếu chunk quan trọng, đặc biệt ở câu bảo mật/mật khẩu/VPN | Tăng hybrid top-k, thêm BM25 keyword tiếng Việt, cải thiện chunk theo heading |
| context_precision | Quá nhiều chunk không liên quan lọt vào context | Siết metadata filter, rerank mạnh hơn, giảm `RERANK_TOP_K` khi confidence thấp |
| answer_relevancy | Câu trả lời không trả lời đúng trọng tâm hoặc thiếu tính toán | Prompt buộc trả lời theo từng bước và kiểm tra lại số liệu trước khi trả lời |

---

## 6. Nhận xét về Adversarial Distribution

Adversarial có avg_score thấp nhất (0.447), thấp hơn multi_hop (0.481) và factual (0.657). Ba câu adversarial rơi vào bottom 10: malware tự xử lý, VPN cá nhân khi WFH, và chu kỳ đổi mật khẩu. Đây là các câu bẫy vì câu hỏi nghe hợp lý nhưng chính sách đúng lại là phủ định hoặc phải ưu tiên version hiện hành. Pipeline cần metadata versioning rõ hơn cho v2024/v2.0 và prompt cần ép model nhận diện các cụm như "không được", "bị cấm", "hiện hành", "đã hết hiệu lực".
