# 🧠 Hệ thống phân loại cảm xúc văn bản tiếng Việt
 
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://phan-tich-cam-xuc-tieng-viet.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9%2B-8b5cf6)
![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-22d3ee)
 
Ứng dụng web phân loại cảm xúc **tích cực / tiêu cực** cho văn bản tiếng Việt, sử dụng kiến trúc **PhoBERT + BiLSTM + Attention**. Ứng dụng cho phép phân tích từng câu hoặc cả tệp dữ liệu, đồng thời hiển thị biểu đồ Attention để người dùng hiểu mô hình đang chú ý vào những từ nào.
 
🔗 **Dùng thử trực tuyến:** https://phan-tich-cam-xuc-tieng-viet.streamlit.app/
 
---
 
## ✨ Tính năng
 
| Tính năng | Mô tả |
|---|---|
| **Phân tích 1 câu trực tiếp** | Nhập một câu tiếng Việt và nhận kết quả cảm xúc cùng độ tin cậy ngay lập tức. |
| **Phân tích hàng loạt qua file** | Tải lên tệp `.csv` hoặc `.xlsx`, chọn cột chứa nội dung và hệ thống tự động gán nhãn cho từng dòng. |
| **Xác suất từng lớp** | Hiển thị xác suất Tích cực và Tiêu cực. |
| **Explainable AI** | Biểu đồ Attention cho thấy mức độ tập trung của mô hình vào từng token. |
| **Biểu đồ phân bố** | Biểu đồ tròn thể hiện tỷ lệ cảm xúc của toàn bộ tệp dữ liệu. |
| **Xuất kết quả** | Tải về tệp đã gán nhãn ở định dạng `.csv` hoặc `.xlsx` (cột mới: `Dự đoán cảm xúc`). |
| **Hiệu ứng cảm xúc** | Kết quả tích cực hoặc tiêu cực kích hoạt hiệu ứng toàn màn hình, có thể bấm vào từng emoji để làm tan biến. |
 
---
 
## 🏗️ Kiến trúc mô hình
 
```
Văn bản đầu vào
      │
      ▼
 Tokenizer PhoBERT (max_length = 64)
      │
      ▼
 PhoBERT (vinai/phobert-base)     → biểu diễn ngữ nghĩa theo ngữ cảnh
      │
      ▼
 BiLSTM                           → nắm bắt ngữ cảnh hai chiều của chuỗi
      │
      ▼
 Attention                        → xác định token quan trọng
      │
      ▼
 Lớp phân loại + Softmax          → Tiêu cực / Tích cực
```
 
- **PhoBERT**: mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt, biểu diễn ngữ nghĩa của từng token.
- **BiLSTM**: khai thác thông tin ngữ cảnh theo cả hai chiều của câu.
- **Attention**: gán trọng số cho từng token, vừa giúp mô hình tập trung vào từ khóa, vừa dùng để trực quan hóa quá trình dự đoán.
---
 
## 📁 Cấu trúc dự án
 
```
.
├── app.py             # Giao diện Streamlit (landing page + trang phân tích)
├── models.py          # Định nghĩa lớp PhoBERT_BILSTM_Attention
├── best_model.pt      # Trọng số mô hình đã huấn luyện
├── requirements.txt   # Các thư viện cần cài đặt
└── README.md
```
 
---
 
## 🚀 Cài đặt và chạy cục bộ
 
**1. Clone dự án**
 
```bash
git clone <đường-dẫn-repository>
cd <tên-thư-mục>
```
 
**2. Tạo môi trường ảo (khuyến nghị)**
 
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```
 
**3. Cài đặt thư viện**
 
```bash
pip install -r requirements.txt
```
 
**4. Đặt file trọng số**
 
Đảm bảo `best_model.pt` nằm cùng thư mục với `app.py`.
 
**5. Chạy ứng dụng**
 
```bash
streamlit run app.py
```
 
## 📖 Hướng dẫn sử dụng
 
### Phân tích 1 câu
 
1. Từ trang chủ, bấm **Bắt đầu phân tích cảm xúc**.
2. Chọn chế độ **Phân tích 1 câu trực tiếp**.
3. Nhập câu tiếng Việt và bấm **Bắt đầu phân tích cảm xúc**.
4. Xem kết quả, xác suất từng lớp và biểu đồ Attention.
### Phân tích qua file
 
1. Chọn chế độ **Phân tích qua file dữ liệu**.
2. Tải lên tệp `.csv` hoặc `.xlsx`.
3. Chọn cột chứa nội dung cần phân tích.
4. Bấm **Chạy phân tích hàng loạt** và chờ thanh tiến trình hoàn tất.
5. Xem biểu đồ phân bố, xem trước kết quả và tải tệp đã gán nhãn.
---
 
## 🛠️ Công nghệ sử dụng
 
- [Streamlit](https://streamlit.io/): xây dựng giao diện web
- [PyTorch](https://pytorch.org/): huấn luyện và suy luận mô hình
- [Hugging Face Transformers](https://huggingface.co/docs/transformers): tải PhoBERT và tokenizer
- [PhoBERT](https://github.com/VinAIResearch/PhoBERT): mô hình ngôn ngữ tiếng Việt
- Pandas, Matplotlib, OpenPyXL: xử lý dữ liệu, vẽ biểu đồ, xuất Excel
---
 
## ⚠️ Lưu ý
 
- Văn bản được cắt hoặc đệm về tối đa **64 token**, vì vậy câu quá dài sẽ bị cắt bớt.
- Mô hình hiện phân loại **hai nhãn** (tích cực và tiêu cực), chưa có nhãn trung tính.
- Lần mở đầu tiên ứng dụng cần tải PhoBERT nên có thể mất một lúc để khởi động.
- Kết quả mang tính tham khảo và có thể sai với câu mỉa mai, viết tắt hoặc tiếng lóng.
---
 
