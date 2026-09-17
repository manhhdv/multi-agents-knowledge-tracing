HƯỚNG DẪN GÁN NHÃN LOẠI LỖI — 46 lượt học sinh (CoMTA)

MỤC TIÊU
Mỗi dòng là một lượt trả lời của học sinh mà bộ dữ liệu đánh giá là SAI. Chọn MỘT loại lỗi chính cho lượt đó.

QUY TẮC BẮT BUỘC
1. Gán độc lập: không trao đổi với người gán còn lại cho đến khi cả hai nộp file.
2. Không dùng ChatGPT/Claude/công cụ AI nào để gợi ý nhãn.
3. Chỉ điền cột "gold", "gold_secondary", "ghi_chu". Không sửa, không xóa, không sắp xếp lại dòng.
4. Chỉ dựa vào ngữ cảnh và lượt học sinh. "[…]" ở đầu ngữ cảnh nghĩa là phần đầu hội thoại đã bị lược bớt.

5 LOẠI LỖI (định nghĩa giống hệt định nghĩa đưa cho mô hình)
• conceptual — hiểu sai một khái niệm hoặc quan hệ giữa các đại lượng.
  Ví dụ: "Hình chữ nhật 3×5 có diện tích 16" (nhầm diện tích với nửa chu vi);
         "1/2 + 1/3 = 2/5" (cộng tử với tử, mẫu với mẫu).
• procedural — một bước của phương pháp bị sai, bị thiếu hoặc sai thứ tự;
  học sinh biết cần làm gì nhưng thực hiện quy trình không đúng.
  Ví dụ: "2 + 3 × 4 = 20" (làm phép cộng trước); "x² − 5x + 6 = 0 ⇒ (x−2)(x−3) = 0 ⇒ x = 2" (bỏ sót nghiệm).
• calculation — phương pháp đúng nhưng tính toán số học sai.
  Ví dụ: "6 × 9 = 56"; "2x = 18 ⇒ x = 8".
• careless — sơ suất: chép sai số, sai dấu, đọc nhầm đề, gõ nhầm; học sinh gần như chắc chắn tự sửa được nếu nhìn lại.
  Ví dụ: đề cho 45 nhưng học sinh viết 54; "x − 5 = 3 ⇒ x = −8" khi các bước khác đều đúng.
• other — lượt không chứa lời giải hay câu trả lời toán học để phân loại: hỏi lại, xin giải thích, phàn nàn,
  trả lời "yes"/"không biết", lạc đề; hoặc có lỗi nhưng không thuộc 4 loại trên.
  Ví dụ: "Bạn giải thích lại được không?"; "tôi không hiểu".

THỨ TỰ QUYẾT ĐỊNH (xét lần lượt, dừng ở loại đầu tiên phù hợp)
  1) Lượt không có câu trả lời/lời giải toán học?  → other
  2) Chỉ là sơ suất chép/dấu/đọc đề, phần còn lại đúng?  → careless
  3) Phương pháp đúng, chỉ sai phép tính số học?  → calculation
  4) Đúng hướng nhưng sai/thiếu/sai thứ tự một bước?  → procedural
  5) Học sinh tin vào một quy tắc hoặc quan hệ sai?  → conceptual

PHÂN BIỆT KHI PHÂN VÂN
• conceptual vs procedural: nếu cho làm lại bài tương tự, học sinh có lặp đúng lỗi đó vì tin là đúng không?
  Có → conceptual. Không, chỉ thực hiện hỏng một bước → procedural.
• calculation vs careless: sai một phép tính số học → calculation; sai do chép, dấu, đọc đề → careless.
• Lượt có nhiều lỗi: chọn lỗi xảy ra SỚM NHẤT làm hỏng câu trả lời.

CỘT gold_secondary (không bắt buộc)
Chỉ điền khi một loại thứ hai cũng thực sự hợp lý. Để trống nếu bạn khá chắc chắn.

CỘT ghi_chu (không bắt buộc)
Ghi ngắn lý do nếu phân vân. Ghi chú giúp buổi thống nhất nhãn nhanh hơn.

KHI XONG
Kiểm tra ô "Đã gán" bên dưới báo 46/46, lưu file giữ nguyên tên, gửi lại cho người điều phối.
