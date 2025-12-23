# Báo Cáo Dự Án: Giải Hệ Phương Trình Tuyến Tính

## 1. Giới Thiệu
Dự án này cài đặt và minh họa các thuật toán số để giải hệ phương trình tuyến tính $Ax = b$.  
Dự án bao gồm cả phương pháp trực tiếp (Direct Methods) và phương pháp lặp (Iterative Methods), kèm theo trực quan hóa kết quả.

## 2. Các Phương Pháp Đã Cài Đặt

### 2.1 Phương Pháp Trực Tiếp (Direct Methods)
Các phương pháp này tìm ra nghiệm chính xác (trong giới hạn sai số máy tính) sau một số bước hữu hạn.

*   **Gaussian Elimination (Khử Gauss)**
    *   **Mô tả**: Biến đổi ma trận mở rộng về dạng bậc thang dòng để giải.
    *   **File**: `solvers/gauss.py`
    *   **Hình ảnh**: `gauss_3d.png` - Biểu diễn 3 mặt phẳng trong không gian 3D cắt nhau tại nghiệm.

*   **LU Decomposition (Phân rã LU)**
    *   **Mô tả**: Phân rã ma trận $A$ thành tích của ma trận tam giác dưới $L$ và tam giác trên $U$ ($A = LU$), sau đó giải hệ phương trình thông qua các bước thay thế xuôi và ngược.
    *   **File**: `solvers/lu_decomposition.py`
    *   **Hình ảnh**: `lu_3d.png` - Biểu đồ nhiệt (Heatmap) hiển thị cấu trúc của hai ma trận $L$ và $U$.

### 2.2 Phương Pháp Lặp (Iterative Methods)
Các phương pháp này bắt đầu từ một nghiệm đoán ban đầu và cải thiện nó qua từng bước lặp cho đến khi sai số nhỏ hơn mức cho phép.

*   **Gauss-Seidel Method**
    *   **Mô tả**: Sử dụng các giá trị nghiệm mới nhất vừa tính được ngay trong vòng lặp hiện tại để tính các biến tiếp theo. Thường hội tụ nhanh hơn phương pháp Jacobi.
    *   **File**: `solvers/gauss_seidel.py`
    *   **Hình ảnh**: `gauss_seidel_convergence.png` - Đồ thị sai số (L2 Error) theo số lần lặp, cho thấy quá trình hội tụ về 0.

*   **SOR Method (Successive Over-Relaxation)**
    *   **Mô tả**: Một biến thể cải tiến của Gauss-Seidel, sử dụng hệ số nới lỏng $\omega$ (omega) để tăng tốc độ hội tụ.
    *   **File**: `solvers/sor.py`
    *   **Tham số**: $\omega = 1.1$ (trong demo).
    *   **Hình ảnh**: `sor_convergence.png` - Đồ thị sai số theo số lần lặp.

## 3. Hướng Dẫn Sử Dụng

1.  **Cài đặt thư viện**:
    ```bash
    pip install numpy matplotlib
    ```

2.  **Chạy chương trình demo**:
    ```bash
    python main.py
    ```

3.  **Kết quả**:
    *   Chương trình sẽ in ra nghiệm và sai số của từng phương pháp.
    *   Các cửa sổ đồ thị sẽ lần lượt hiện ra (bạn cần đóng cửa sổ hiện tại để xem cửa sổ tiếp theo).
    *   4 file ảnh kết quả sẽ được lưu vào thư mục hiện tại.

## 4. Kết Quả Mong Đợi (Demo)
Với hệ phương trình mẫu:
$$
\begin{cases}
4x + y + z = 12 \\
x + 5y + 2z = 13 \\
x + 2y + 6z = 22
\end{cases}
$$
Nghiệm kỳ vọng: $x=2, y=1, z=3$.

Tất cả các phương pháp đều phải hội tụ về nghiệm này với sai số rất nhỏ.
