HƯỚNG DẪN XÁC MINH ĐỘC LẬP TÍNH ĐÚNG/SAI CỦA LƯỢT HỌC SINH — 139 lượt (MathDial, tập test)

MỤC TIÊU
Mỗi dòng gồm đề bài, đoạn hội thoại gia sư - học sinh trước lượt này, lượt trả lời mới nhất của
học sinh, và đáp án đúng của bài. Hãy tự quyết định lượt đó của học sinh ĐÚNG hay SAI.
Kết quả dùng làm mốc tham chiếu do người xác minh, để đối chiếu với nhãn GPT-4o có sẵn trong bộ
dữ liệu và với các bộ chấm tự động (SAVA, LLM-judge, regex). Nhãn của các hệ đó KHÔNG hiển thị
cho bạn, và nhãn GPT-4o cũng không phải đáp án đúng của công việc này - chính nó đang được kiểm tra.

QUY TẮC BẮT BUỘC
1. Xác minh độc lập: không trao đổi với người xác minh còn lại cho đến khi cả hai nộp file.
2. Không dùng ChatGPT/Claude/công cụ AI nào để gợi ý nhãn.
3. Chỉ điền cột "dung_sai" và "ghi_chu". Không sửa, không xóa, không sắp xếp lại dòng.
4. Chỉ dựa vào đề bài, ngữ cảnh, lượt học sinh và đáp án. "[…]" ở đầu ngữ cảnh nghĩa là phần đầu
   hội thoại đã bị lược bớt.
5. Tự tính lại khi cần: đáp án ở cột "dap_an" là đáp án CUỐI của bài, không phải đáp án của bước
   mà học sinh đang làm.

3 NHÃN
• dung     - Lượt của học sinh đúng về mặt toán học TRONG NGỮ CẢNH của bước đang làm.
             Bao gồm cả khi học sinh nêu một giá trị trung gian đúng mà giá trị đó khác đáp án cuối.
             Ví dụ: bài có đáp án cuối 10080, học sinh đang tính giá hiện tại của một máy và nói
             "3000" - nếu 3000 đúng cho bước đó thì nhãn là dung.
• sai      - Lượt chứa một câu trả lời hoặc một bước toán học sai.
• chua_ro  - Lượt không chứa nội dung toán học để phán xét (hỏi lại, xin giải thích, "vâng",
             "tôi không hiểu", lạc đề), hoặc không đủ thông tin để kết luận đúng/sai.

THỨ TỰ QUYẾT ĐỊNH (xét lần lượt, dừng ở nhãn đầu tiên phù hợp)
  1) Lượt không có nội dung toán học, hoặc không thể kết luận?         -> chua_ro
  2) Có bước/câu trả lời toán học và nó sai?                            -> sai
  3) Còn lại                                                            -> dung

PHÂN BIỆT KHI PHÂN VÂN (điểm quan trọng nhất của công việc này)
• BƯỚC TRUNG GIAN vs ĐÁP ÁN CUỐI: đây là ca hay bị chấm sai nhất. Hãy xác định học sinh đang trả
  lời cho bước nào dựa vào câu hỏi gần nhất của gia sư trong ngữ cảnh, rồi mới so sánh.
  Một giá trị trung gian đúng KHÔNG phải là "sai" chỉ vì nó khác cột "dap_an".
• Học sinh vừa nêu đúng một phần vừa sai một phần: nếu phần sai làm hỏng kết luận của lượt -> sai.
• Học sinh chỉ nhắc lại/đồng ý với điều gia sư vừa nói, không tự đưa ra bước nào -> chua_ro.
• Đúng về số nhưng sai đơn vị hoặc sai cách diễn đạt đại lượng -> sai, và ghi lý do vào ghi_chu.
• Không chắc giữa hai nhãn: chọn nhãn bạn nghiêng về hơn và GHI RÕ vào ghi_chu. Đừng để trống.

CỘT ghi_chu (không bắt buộc, nhưng rất nên điền khi phân vân)
Ghi ngắn lý do. Ghi chú giúp buổi thống nhất nhãn nhanh hơn và giúp phân tích các ca mà bộ chấm
tự động và nhãn GPT-4o không thống nhất với nhau.
