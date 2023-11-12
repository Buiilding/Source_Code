Mô hình Phân loại Ngôn ngữ Ký hiệu
Đây là một Mô hình Phân loại Ngôn ngữ Ký hiệu có khả năng phân loại các cử chỉ hoặc biểu tượng ngôn ngữ ký hiệu vào các lớp khác nhau. Mô hình này được huấn luyện trên một tập dữ liệu các hình ảnh ngôn ngữ ký hiệu và có thể được sử dụng để phân loại thời gian thực các cử chỉ ngôn ngữ ký hiệu trong hình ảnh.

Yêu cầu
Đảm bảo bạn đã cài đặt các yêu cầu sau trước khi chạy mã:

- torch==1.9.0
- streamlit==0.86.0
- Pillow==8.3.1
- torchvision==0.10.0
- numpy==1.21.0
- scikit-learn==0.24.2
- pandas==1.3.0
- matplotlib==3.4.2
- opencv-python==4.5.3.56

Bạn có thể cài đặt các gói cần thiết bằng cách chạy lệnh:
    pip install -r requirements.txt

Tập dữ liệu
Tập dữ liệu được sử dụng để huấn luyện mô hình nằm trong thư mục Data_Sample. Nó chứa các hình ảnh của các cử chỉ ngôn ngữ ký hiệu khác nhau, được phân loại vào các lớp khác nhau.

Cách sử dụng
Để chạy Mô hình Phân loại Ngôn ngữ Ký hiệu, làm theo các bước sau:

1. Sao chép kho lưu trữ hoặc tải xuống mã nguồn.

2. Cài đặt các gói cần thiết được đề cập trong phần yêu cầu.

3. Mở terminal hoặc command prompt và điều hướng đến thư mục dự án.

4. Chạy lệnh sau để bắt đầu ứng dụng:
    streamlit run main.py
5. Khi ứng dụng bắt đầu, một trang web sẽ mở trong trình duyệt của bạn.

6. Bạn sẽ thấy một menu thanh bên với các tùy chọn khác nhau. Chọn "AI Computer Vision" để truy cập tính năng phân loại ngôn ngữ ký hiệu.

6. Nhấp vào nút "Chọn một hình ảnh" để tải lên một hình ảnh chứa cử chỉ ngôn ngữ ký hiệu.

7. Mô hình sẽ phân tích hình ảnh và hiển thị cử chỉ ngôn ngữ ký hiệu được dự đoán cùng với lớp tương ứng.

8. Bạn cũng có thể khám phá các tính năng khác như trang "Độ chính xác" để xem độ chính xác của mô hình và trang "Tập dữ liệu" để xem các hình ảnh mẫu được sử dụng để huấn luyện.


Cảm ơn
Mô hình Phân loại Ngôn ngữ Ký hiệu dựa trên kỹ thuật học sâu sử dụng PyTorch và streamlit.
Tập dữ liệu được sử dụng để huấn luyện mô hình được lấy từ Kaggle.
Cấu trúc dự án và tổ chức mã được lấy cảm hứng từ các phương pháp tốt nhất trong dự án học máy.
Tài liệu tham khảo
Tài liệu PyTorch
Tài liệu streamlit
Kaggle
Hãy tự do sửa đổi và cải tiến mã theo yêu cầu của bạn. Nếu bạn có bất kỳ câu hỏi hoặc cần trợ giúp thêm, xin vui lòng liên hệ.

SDT : 0965250428