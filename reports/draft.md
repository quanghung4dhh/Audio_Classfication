Hướng thực hiện đã chọn: CNN làm mô hình baseline và BEATs làm hướng Self-Supervised Learning (SSL) chính

Đã tìm hiểu được về:
1. Self-supervised learning
2. Mel Spectrogram

=======================================================================================================

Thư viện Pytorch?
Mạng CNN?

***Pytorch: https://pytorch.org/get-started/locally/

1. Tensor? [Sample Code](../learning/tensor.py)
- Tensor là lớp cấu trúc đa chiều, được PyTorch thiết kế để hỗ trợ tính toán Deep Learning và tự động tính đạo hàm.
- VD các cấu trúc có số chiều khác nhau:
  + Số đơn: 5 - 0 chiều
  + Mảng hay vector: [1,2,3] - 1 chiều
  + Ma trận: [1,2,3] - 2 chiều
             [4,5,6]
  + Tensor: Nó gồm 3 chiều trở lên
- Các thuộc tính quan trọng:
  + shape: kích thước của từng chiều
  + ndim: số chiều của Tensor
  + dtype: kiểu dữ liệu (như float32 hoặc int64)
  + device: Tensor đang nằm trên CPU hay GPU
- Áp dụng trong bài toán phân loại âm thanh: 
  + Có thể dùng tensor để biểu diễn 1 đoạn âm thanh có n mẫu
  + Với mel-spectrogram, dữ liệu có thể là [batch, freq, time] (Có thể thấy trong biểu đồ spectrogram) 
  + Hình dạng thực tế đưa vào phụ thuộc vào cách tiền xử lý âm thanh
  + Cần biết trước dữ liệu đưa vào như thế nào để sử dụng Pytorch

2. Autograd - tính đạo hàm tự động [Sample Code](../learning/autograd.py)

3. Xây dựng mô hình Neutral Network? [Sample Code](../learning/NN.py)
- Để xây dựng mô hình thì Pytorch cung cấp torch.nn
- Có thể ứng dụng đến các lớp thương gặp trong CNN

4. Optimizer và quá trình huấn luyện [Sample Code](../learning/optimize.py)
- Loss function: đo mức độ sai của dự đoán
- Optimizer: Cập nhật trọng số của mô hình dựa trên gradient

5. DataLoader: Đưa dữ liệu vào mô hình [Sample Code](../learning/dataloader.py)
- Dataset: Định nghĩa cách lấy một mẫu dữ liệu và nhãn
- DataLoader: Chia batch, xáo trộn dữ liệu, nạp mẫu
- Training Loop: Dự đoán -> Loss -> Gradient -> Cập nhật

=======================================================================================================

*** Mạng neutral network, loss function, gradient?
