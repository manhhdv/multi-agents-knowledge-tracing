HƯỚNG DẪN CHẤM FEEDBACK CỦA HỆ THỐNG GIA SƯ — 30 trường hợp (MathDial)

MỤC TIÊU
Mỗi dòng gồm một đoạn hội thoại gia sư – học sinh, lượt trả lời mới nhất của học sinh, đáp án đúng của bài,
và phản hồi (feedback) do hệ thống sinh ra cho lượt đó. Chấm chất lượng feedback theo 3 tiêu chí, thang 1–5.

QUY TẮC BẮT BUỘC
1. Chấm độc lập: không trao đổi với người chấm còn lại cho đến khi cả hai nộp file.
2. Không dùng ChatGPT/Claude/công cụ AI nào để hỗ trợ chấm.
3. Chỉ điền các cột chấm điểm và ghi chú. Không sửa, không xóa, không sắp xếp lại dòng.
4. Tự đánh giá câu trả lời của học sinh đúng hay sai dựa trên hội thoại và đáp án; hệ thống không cho bạn biết.
5. Chấm từng tiêu chí riêng: một feedback rõ ràng nhưng sai toán vẫn có thể được Clarity cao, Correctness thấp.

3 TIÊU CHÍ (giống định nghĩa đưa cho LLM chấm)
• correctness — Toán học trong feedback đúng, và nhận định về câu trả lời của học sinh (đúng/sai/chưa rõ) là đúng.
• relevance — Feedback nói đúng vào lượt này của học sinh, lỗi của học sinh và kỹ năng yếu nhất, không chung chung.
• clarity — Feedback rõ ràng, dễ hiểu và phù hợp với người học.

THANG ĐIỂM
  5 — Rất tốt: không có vấn đề nào.
  4 — Tốt: một vấn đề nhỏ, không ảnh hưởng đến việc học.
  3 — Tạm được: có vấn đề rõ ràng nhưng feedback vẫn dùng được.
  2 — Kém: vấn đề nghiêm trọng, dễ gây hiểu nhầm hoặc vô ích.
  1 — Rất kém: sai hoàn toàn, lạc đề hoặc gây hại cho việc học.

VÍ DỤ ĐỊNH MỨC CORRECTNESS (không lấy từ dữ liệu)
  Học sinh: "3/4 + 1/4 = 4/8". Đáp án: 1.
  5 — "Gần đúng rồi! Khi cộng hai phân số cùng mẫu, mẫu số có thay đổi không?"
  3 — "Chưa đúng. Hãy kiểm tra lại cách cộng phân số." (đúng nhưng không chỉ ra lỗi)
  1 — "Chính xác, 4/8 là kết quả đúng!" (nhận định sai)

CỘT lo_dap_an (có / khong)
Chọn "co" nếu feedback nói ra đáp án cuối cùng khi học sinh CHƯA trả lời đúng. Nếu học sinh đã đúng, chọn "khong".

CỘT ghi_chu (không bắt buộc)
Ghi ngắn lý do khi cho điểm 1–2 hoặc khi phân vân.

KHI XONG
Kiểm tra ô "Đã chấm" bên dưới báo 30/30, lưu file giữ nguyên tên, gửi lại cho người điều phối.
