import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter
from scipy import signal

# Minimal Ricker CWT to avoid deprecated SciPy wavelet helpers
def ricker_wavelet(points, width):
    width = float(width)
    t = np.linspace(-(points - 1) / 2, (points - 1) / 2, points)
    norm = 2.0 / (np.sqrt(3 * width) * np.pi ** 0.25)
    return norm * (1 - (t ** 2) / (width ** 2)) * np.exp(-(t ** 2) / (2 * width ** 2))


def cwt_ricker(data, widths):
    data = np.asarray(data)
    output = np.zeros((len(widths), data.size))
    for i, width in enumerate(widths):
        points = int(max(1, np.ceil(width * 10)))  # match old SciPy cwt window length
        wavelet = ricker_wavelet(points, width)
        output[i, :] = signal.convolve(data, wavelet, mode="same")
    return output

# 1. Tạo tín hiệu giả lập (4 giây, 250Hz)
fs = 250
t = np.arange(0, 4, 1/fs)
# Tín hiệu gồm: Nhiễu chậm (2Hz) + Alpha liên tục (10Hz) + Beta đột ngột (25Hz tại giây 2-2.5)
sig_noise = 0.5 * np.sin(2 * np.pi * 2 * t) 
sig_alpha = 0.8 * np.sin(2 * np.pi * 10 * t)
sig_beta_burst = np.zeros_like(t)
sig_beta_burst[500:625] = 1.0 * np.sin(2 * np.pi * 25 * t[500:625]) # Burst ở giây 2.0 - 2.5

raw_signal = sig_noise + sig_alpha + sig_beta_burst + 0.2 * np.random.randn(len(t))

# 2. Lọc FIR (Bandpass 7-30 Hz) - Giống code của bạn
# Tạo bộ lọc FIR
numtaps = 101
fir_coeff = firwin(numtaps, [7, 30], fs=fs, pass_zero=False)
# Áp dụng bộ lọc
fir_filtered = lfilter(fir_coeff, 1.0, raw_signal)
# Dịch chuyển pha (để khớp hình ảnh) do độ trễ của FIR
shift = (numtaps - 1) // 2
fir_filtered_shifted = fir_filtered[shift:] 
t_fir = t[:-shift]

# 3. Phân tích Wavelet (Time-Frequency)
freqs = np.arange(5, 35, 1)
widths = 500 / (2 * np.pi * freqs) # Chuyển đổi tần số sang độ rộng wavelet
cwtmatr = cwt_ricker(raw_signal, widths)

# --- VẼ ĐỒ THỊ ---
plt.figure(figsize=(12, 10))

# Hình 1: Tín hiệu gốc
plt.subplot(3, 1, 1)
plt.plot(t, raw_signal, color='gray', alpha=0.7, label='Raw Signal')
plt.plot(t, sig_beta_burst, color='red', linestyle='--', linewidth=2, label='Beta Burst (Sự kiện cần tìm)')
plt.title("1. Tín hiệu gốc (Gồm nhiễu chậm + Alpha liên tục + Beta Burst ngắn)")
plt.legend()

# Hình 2: Kết quả lọc FIR (7-30Hz)
plt.subplot(3, 1, 2)
plt.plot(t_fir, fir_filtered_shifted, color='blue')
plt.title("2. Lọc FIR (7-30Hz): Giữ lại Alpha & Beta, loại bỏ nhiễu chậm")
plt.axvline(2.0, color='r', linestyle='--', alpha=0.5)
plt.axvline(2.5, color='r', linestyle='--', alpha=0.5)
plt.text(2.05, 1.5, "Beta Burst ở đây", color='red')

# Hình 3: Wavelet Transform (Scalogram)
plt.subplot(3, 1, 3)
plt.imshow(np.abs(cwtmatr), extent=[0, 4, 5, 35], cmap='jet', aspect='auto', origin='lower')
plt.title("3. Wavelet: Bản đồ Thời gian - Tần số")
plt.ylabel("Tần số (Hz)")
plt.xlabel("Thời gian (s)")
plt.colorbar(label="Năng lượng")
# Khoanh vùng sự kiện
plt.annotate('Alpha (10Hz) liên tục', xy=(0.5, 10), xytext=(0.5, 20),
             arrowprops=dict(facecolor='white', shrink=0.05), color='white')
plt.annotate('Beta Burst (25Hz)', xy=(2.25, 25), xytext=(2.8, 30),
             arrowprops=dict(facecolor='white', shrink=0.05), color='white')

plt.tight_layout()
plt.show()
