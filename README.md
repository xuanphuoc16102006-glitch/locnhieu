# Web lọc nhiễu âm thanh

Ứng dụng hỗ trợ thử nghiệm và đánh giá các phương pháp lọc nhiễu âm thanh trên giao diện [Streamlit](https://streamlit.io/). Người dùng có thể tải tệp WAV hoặc dùng tín hiệu mẫu, chủ động tạo nhiễu, áp dụng bộ lọc và so sánh kết quả qua âm thanh, biểu đồ và SNR.

## Tính năng

- Tải tệp WAV hoặc dùng tín hiệu mẫu tích hợp.
- Tạo nhiễu trắng, tiếng ù 50 Hz hoặc cả hai; điều chỉnh SNR đầu vào từ -5 đến 20 dB.
- Lọc bằng Notch 50 Hz, Bandpass 300–3400 Hz hoặc Spectral Subtraction.
- Nghe tín hiệu sạch, tín hiệu có nhiễu và tín hiệu sau lọc.
- Phân tích dạng sóng, phổ công suất Welch, spectrogram và mức thay đổi SNR.

## Cài đặt và khởi chạy

Cần có Python 3.10 trở lên. Mở terminal tại thư mục dự án và tạo môi trường ảo.

**Windows PowerShell**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Sau khi kích hoạt môi trường ảo, cài đặt và chạy ứng dụng:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Mở địa chỉ được in trong terminal, thường là `http://localhost:8501`.

## Sử dụng ứng dụng

1. Tải tệp WAV ở thanh bên hoặc để trống để dùng tín hiệu mẫu 2 giây, 16 kHz.
2. Chọn loại nhiễu và SNR đầu vào. SNR càng thấp, nhiễu được thêm càng mạnh.
3. Chọn phương pháp lọc. Với Spectral Subtraction, điều chỉnh thêm hệ số Alpha.
4. Nhấn **Xử lý âm thanh**, sau đó nghe ba phiên bản và đối chiếu chỉ số, biểu đồ.

Tệp WAV cần có thời lượng tối thiểu 1 giây và tần số lấy mẫu từ 8 kHz trở lên. Âm thanh nhiều kênh được trộn về mono trước khi xử lý.

## Nguyên lý và phân tích kết quả

### Mô hình nhiễu

Tín hiệu đầu vào được mô hình hóa là tổng của tín hiệu sạch và nhiễu:

$$
x[n] = s[n] + v[n]
$$

Trong đó $s[n]$ là tín hiệu sạch, $v[n]$ là nhiễu và $x[n]$ là tín hiệu cần lọc. Nhiễu trắng có năng lượng trải rộng; tiếng ù tập trung quanh 50 Hz. SNR đầu vào biểu thị tỷ lệ công suất tín hiệu trên công suất nhiễu:

$$
\mathrm{SNR}_{\mathrm{dB}} = 10\log_{10}\left(\frac{P_s}{P_v}\right)
$$

Khi so sánh bộ lọc, nên giữ nguyên tín hiệu, loại nhiễu và SNR đầu vào rồi chỉ thay đổi phương pháp lọc hoặc Alpha.

### Phương pháp lọc

- **Notch 50 Hz** sử dụng bộ lọc IIR có vùng triệt hẹp. Với $Q = 35$, độ rộng vùng triệt xấp xỉ $50/35 \approx 1.43$ Hz. Bộ lọc phù hợp với tiếng ù ổn định, nhưng ít tác động đến nhiễu trắng.
- **Bandpass 300–3400 Hz** sử dụng bộ lọc Butterworth bậc 4 để giữ dải thoại phổ biến và suy giảm tần số ngoài dải. Vì vậy, thành phần hữu ích nằm ngoài dải này cũng có thể bị giảm.
- **Spectral Subtraction** dùng STFT để ước lượng phổ nhiễu $D[k]$ từ 0,5 giây đầu, sau đó trừ phổ nhiễu khỏi biên độ phổ đầu vào:

$$
|\hat{S}[m,k]| = \max\left(|X[m,k]| - \alpha D[k],\ 0.02|X[m,k]|\right)
$$

Trong đó $X[m,k]$ là phổ đầu vào, $m$ là chỉ số khung, $k$ là chỉ số tần số và $\alpha$ là hệ số khử. Pha đầu vào được giữ lại khi tái tạo tín hiệu. Alpha lớn tăng mức khử nhưng cũng có thể gây méo hoặc tạo âm thanh lạo xạo (*musical noise*).

### Đọc chỉ số và biểu đồ

- **SNR sau lọc** cao hơn thường cho thấy sai số so với tín hiệu sạch thấp hơn. Mức cải thiện là SNR sau lọc trừ SNR trước lọc: dương là tăng, gần 0 dB là ít thay đổi, âm là sai số tăng.
- **Dạng sóng** so sánh biên độ theo thời gian. Ứng dụng có đoạn đệm im lặng 0,5 giây ở đầu tín hiệu.
- **Phổ công suất Welch** cho thấy phân bố năng lượng theo tần số. Đỉnh gần 50 Hz thường gợi ý tiếng ù; nền phổ dâng trên dải rộng thường gợi ý nhiễu trắng.
- **Spectrogram** thể hiện năng lượng theo thời gian và tần số. Vùng sáng hơn biểu thị năng lượng cao hơn; nền nhiễu giảm thường làm các vùng phân tán tối đi.

Kết quả kỳ vọng: Notch làm giảm đỉnh 50 Hz; Bandpass suy giảm năng lượng ngoài dải 300–3400 Hz; Spectral Subtraction làm giảm nền nhiễu khi đoạn ước lượng đại diện tốt cho nhiễu. Notch có thể cải thiện ít với nhiễu trắng, còn Bandpass có thể làm mất nội dung ngoài dải. Nên giữ nguyên loại nhiễu và SNR khi so sánh các phương pháp.

