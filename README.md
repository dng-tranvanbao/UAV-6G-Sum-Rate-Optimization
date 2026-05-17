# 🛩️ UAV Position Optimization for 6G Network Sum-Rate Maximization

Ứng dụng mô phỏng trực quan hóa bài toán tối ưu hóa vị trí UAV (Unmanned Aerial Vehicle) để tối đa hóa tổng tốc độ truyền tải dữ liệu (Sum-Rate) cho mạng di động thế hệ mới 6G sử dụng thư viện **Streamlit** và **Matplotlib**.

Dự án này được phát triển cho môn học **CS 723 - Mạng Không Dây Nâng Cao** tại trường Đại học Duy Tân (DTU).

---

## 🌟 Tính Năng Chính

*   **Tùy biến tham số mô phỏng linh hoạt:**
    *   Thay đổi số lượng người dùng dưới đất (Ground Users - GUs) từ $3$ đến $10$ người qua thanh Slider. Vị trí các user sẽ được tạo ngẫu nhiên trong vùng không gian 2D kích thước $100 \times 100$ m.
    *   Chỉnh độ cao bay của UAV ($10$ m đến $200$ m).
    *   Tùy chỉnh công suất phát của UAV ($10$ dBm đến $46$ dBm).
    *   Tùy chọn độ phân giải của lưới quét tọa độ tìm vị trí tối ưu (Grid Search N x N).
    *   Cài đặt Random Seed để tái lập chính xác các phân bố ngẫu nhiên của các user.
*   **Triển khai 3 thuật toán xác định vị trí UAV:**
    1.  **📍 Đứng yên tại tâm (Stationary):** UAV cố định ở tọa độ $(0, 0)$ trên bản đồ.
    2.  **🚀 Tham lam (Greedy):** UAV bay về hướng của người dùng ở xa tâm nhất (Heuristic) để kéo gần khoảng cách đối với các user chịu suy hao nặng nề nhất.
    3.  **🎯 Tối ưu hóa (Optimized Grid Search):** Quét toàn bộ không gian tọa độ 2D của lưới với độ phân giải tùy chọn để tìm ra tọa độ $(x, y)$ giúp cực đại hóa tổng dung lượng kênh truyền Shannon (Sum-Rate) của hệ thống.
*   **Giao diện Trực Quan hóa Hiện Đại & Đẹp Mắt:**
    *   Bản đồ 2D (Scatter Plot) hiển thị vị trí thời gian thực của các Ground Users và 3 vị trí UAV tương ứng với 3 thuật toán, kèm theo đường liên kết mờ nối giữa UAV và người dùng để mô tả độ xa/gần.
    *   Biểu đồ Bar Chart so sánh trực quan Tổng Sum-Rate (Mbps) đạt được của cả 3 thuật toán, có hiển thị tỷ lệ phần trăm cải thiện hiệu năng (%).
    *   Biểu đồ chi tiết (Per-User Capacity Bar Chart) cho thấy sự phân bổ băng thông/tốc độ cho từng người dùng cụ thể ứng với từng thuật toán.
    *   Bảng thống kê số liệu tốc độ chi tiết của mỗi người dùng (Mbps).
    *   Bảng thông số kỹ thuật chi tiết cùng tọa độ UAV chính xác tìm được của từng thuật toán.

---

## 📐 Cơ Sở Lý Thuyết & Tham Số Hệ Thống

Mô phỏng áp dụng mô hình kênh truyền truyền sóng mmWave (ở tần số $28\text{ GHz}$) thường được định hướng cho mạng 6G với công thức tính dung lượng Shannon:

$$C_i = B \cdot \log_2\left(1 + \text{SNR}_i\right)$$

Trong đó:
*   $B = 20\text{ MHz}$ (Băng thông kênh truyền).
*   $\text{SNR}_i = \frac{P_{rx, i}}{\sigma^2}$ (Tỷ số tín hiệu trên nhiễu tại User $i$).
*   Công suất nhiễu $\sigma^2 = -100\text{ dBm}$ (Thermal noise floor).
*   Công suất nhận được tại User $i$: $P_{rx, i} = \frac{P_{tx}}{\text{PL}(d_i)}$.
*   Mô hình suy hao đường truyền (Path Loss - LoS UAV): $\text{PL}(d_i) = \text{FSPL}_{1\text{m}} \cdot (d_i)^{\alpha}$ với hệ số suy hao đường truyền $\alpha = 2.2$.
*   Khoảng cách 3D từ UAV $(x_{uav}, y_{uav}, h)$ tới User $i$ $(x_i, y_i, 0)$ là: $d_i = \sqrt{(x_i - x_{uav})^2 + (y_i - y_{uav})^2 + h^2}$.

---

## 🛠️ Hướng Dẫn Cài Đặt & Chạy Dự Án

Thực hiện theo các bước dưới đây để chạy ứng dụng trên máy tính cá nhân của bạn:

### 1. Yêu cầu Hệ thống
Dự án yêu cầu cài đặt sẵn **Python** (phiên bản từ `3.8` trở lên). Bạn có thể kiểm tra phiên bản Python hiện tại bằng lệnh:
```bash
python --version
```

### 2. Tải Mã Nguồn & Truy Cập Thư Mục Dự Án
Di chuyển vào thư mục chứa dự án:
```bash
cd d:\Brady\DTU-Learning\CS_723_Mang_Khong_Day_Nang_Cao\UAV_Demo
```

### 3. Cài Đặt Các Thư Viện Cần Thiết
Cài đặt tất cả các thư viện phụ thuộc bằng công cụ quản lý gói `pip`:
```bash
pip install -r requirements.txt
```
*(Nếu chưa có file requirements.txt, bạn có thể cài trực tiếp bằng lệnh: `pip install streamlit matplotlib numpy pandas`)*

### 4. Chạy Ứng Dụng Streamlit
Khởi chạy máy chủ phát triển cục bộ của Streamlit bằng lệnh:
```bash
streamlit run app.py
```

Sau khi chạy lệnh trên, ứng dụng sẽ tự động mở một tab mới trên trình duyệt mặc định của bạn tại địa chỉ:
*   👉 **Local URL:** `http://localhost:8501`

Nếu ứng dụng không tự động mở, bạn có thể copy địa chỉ trên dán vào trình duyệt web.

---

## 🖥️ Giao Diện Ứng Dụng Khi Chạy

Khi khởi chạy ứng dụng thành công, giao diện sẽ xuất hiện bao gồm:
1.  **Sidebar bên trái:** Cho phép bạn toàn quyền kiểm soát các tham số như số lượng User, độ cao bay của UAV, công suất phát, Random Seed để thay đổi cách sinh vị trí ngẫu nhiên và mật độ lưới dò tìm vị trí tối ưu.
2.  **Khu vực trung tâm:**
    *   **3 Thẻ KPI Chỉ Số Nhanh:** Hiển thị trực quan tốc độ Sum-Rate của 3 phương án để bạn nhận ra ngay hiệu quả vượt trội của thuật toán Tối ưu hóa.
    *   **Bản đồ 2D Vị Trí:** Minh họa trực quan vị trí phân bố của người dùng và UAV.
    *   **Biểu đồ Bar so sánh hiệu năng:** Cho thấy mức độ tối ưu hóa vượt trội (bao nhiêu % cải tiến so với phương án đứng yên mặc định).
    *   **Chi tiết tốc độ từng User:** Dưới dạng biểu đồ cột nhóm và bảng số liệu phân tích rõ sự cân bằng tốc độ (Fairness) giữa các vị trí.
    *   **Mục Chi tiết kỹ thuật & Tọa độ:** Bảng thông tin mở rộng chứa thông số kênh truyền và tọa độ chính xác của UAV.

---
