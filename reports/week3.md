## 1. Chuẩn hóa quá trình huấn luyện:

- Trong quá trình huấn luyện, sẽ có nhiều tham số điều khiển quá trình huấn luyện như:

| Tham số       | Giá trị ban đầu | Ý nghĩa                               |
| ------------- | --------------- | ------------------------------------- |
| BATCH_SIZE    | 32              | Số mẫu trong một batch                |
| LEARNING_RATE | 0.001           | Tốc độ cập nhật trọng số              |
| NUM_EPOCHS    | 30              | Số vòng huấn luyện tối đa             |
| PATIENCE      | 7               | Số epoch chờ trước khi Early Stopping |

- Những tham số này hiện tại đang nằm rải rác trong nhiều file, vì vậy việc tái lập thực nghiệm cũng như là kiểm thử những cấu hình khác nhau sẽ trở nên khó khăn
- Vì vậy, ta sẽ gom những tham số này vào một file config.py giúp:
  - Có một nơi để quản lý tất cả tham số
  - Các file Python khác trong dự án có thể dùng chung cấu hình
  - Dễ dàng thay đổi tham số để thực nghiệm
  - Dễ dàng ghi lại tham số cấu hình tương ứng với kết quả huấn luyện

### 1.1 Traning configuration

- Đây là những tham số điều khiển quá trình huấn luyện.

| Tham số       | Giá trị ban đầu | Ý nghĩa                               |
| ------------- | --------------- | ------------------------------------- |
| BATCH_SIZE    | 32              | Số mẫu trong một batch                |
| LEARNING_RATE | 0.001           | Tốc độ cập nhật trọng số              |
| NUM_EPOCHS    | 30              | Số vòng huấn luyện tối đa             |
| PATIENCE      | 7               | Số epoch chờ trước khi Early Stopping |

### 1.2 Dataset Configuration

- Các thông số này liên quan đến dữ liệu đầu vào và tiền xử lý.

| Tham số     | Giá trị ban đầu | Ý nghĩa                         |
| ----------- | --------------- | ------------------------------- |
| NUM_CLASSES | 10              | Số lớp âm thanh                 |
| SAMPLE_RATE | 16000           | Tần số lấy mẫu                  |
| DURATION    | 4.0             | Thời lượng file âm thanh (giây) |
| N_MELS      | 64              | Số Mel frequency bins           |

### 1.3 Model Configuration

- Thông số này liên quan đến kiến trúc CNN.

| Tham số | Giá trị ban đầu | Ý nghĩa                                           |
| ------- | --------------- | ------------------------------------------------- |
| DROPOUT | 0.3             | Tỷ lệ neuron bị bỏ qua trong quá trình huấn luyện |

### 1.4 Reproducibility Configuration

| Tham số | Giá trị ban đầu | Ý nghĩa                     |
| ------- | --------------- | --------------------------- |
| SEED    | 42              | Giá trị khởi tạo ngẫu nhiên |

- Đây là tham số Random Seed
- Trong Deep Learning, có nhiều hoạt động sử dụng số ngẫu nhiên:
  - Khởi tạo trọng số ban đầu của CNN
  - Xáo trộn dữ liệu trước mỗi epoch
  - Dropout lựa chọn neuron để loại bỏ
  - Các phép biến đổi ngẫu nhiên nếu có
- Do đó, dẫn đến việc cùng một kiến trúc và cùng một dữ liệu huấn luyện nhưng kết quả huấn luyện có thể khác nhau
- Random Seed giúp kiểm soát những yếu tố ngẫu nhiên này