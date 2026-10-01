from __future__ import annotations

from io import BytesIO

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
from scipy import signal
from scipy.signal.windows import hamming

plt.style.use("dark_background")


st.set_page_config(
    page_title="Lọc nhiễu âm thanh",
    page_icon="♫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
        :root { --ink: #f8fafc; --muted: #94a3b8; --paper: #0f172a; --panel: #1e293b;
            --line: #334155; --green: #10b981; --coral: #fb7185; }
        html, body, [class*="css"] { font-family: 'Manrope', sans-serif; color: var(--ink); }
        .stApp { background: var(--paper); color: var(--ink); }
        [data-testid="stSidebar"] { background: #1e293b; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
    .eyebrow { font: 500 11px 'DM Mono', monospace; letter-spacing: 1px; text-transform: uppercase; color: var(--green); }
    .hero { padding: 1.35rem 0 1.1rem; border-bottom: 1px solid var(--line); margin-bottom: 1.2rem; }
    .hero h1 { font-size: 2rem; line-height: 1.12; margin: .35rem 0 .45rem; letter-spacing: 0; }
    .hero p { color: var(--muted); margin: 0; font-size: .95rem; }
    [data-testid="stMetric"] { background: var(--panel); padding: 14px 17px; border: 1px solid var(--line); border-radius: 6px; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { font-family: 'DM Mono', monospace; color: var(--ink); }
    .source-note { color: var(--muted); font-size: .85rem; padding: .25rem 0 .8rem; }
        div.stButton > button { border-radius: 8px; }
        div.stButton > button[kind="primary"] { background: var(--green); border: 0; color: #052e25; font-weight: 700; }
        div.stButton > button[kind="primary"]:hover { background: #34d399; border: 0; color: #052e25; }
    div[data-testid="stAudio"] { padding-top: .3rem; }
    .section-label { font: 500 11px 'DM Mono', monospace; text-transform: uppercase; color: var(--muted); }
        [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] { color: var(--ink); }
    </style>
    """,
    unsafe_allow_html=True,
)


def sample_signal(sample_rate: int) -> np.ndarray:
    time = np.arange(2 * sample_rate, dtype=np.float64) / sample_rate
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 3 * time))
    samples = envelope * (
        0.6 * np.sin(2 * np.pi * 440 * time)
        + 0.3 * np.sin(2 * np.pi * 1000 * time)
        + 0.2 * np.sin(2 * np.pi * 2000 * time)
    )
    return (samples / np.max(np.abs(samples))).astype(np.float32)


def read_audio(uploaded_file) -> tuple[np.ndarray, int, str]:
    try:
        data, sample_rate = sf.read(BytesIO(uploaded_file.getvalue()), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError(f"Không đọc được WAV: {exc}") from exc
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError("File không có mẫu âm thanh hợp lệ.")
    channel_count = data.shape[1]
    mono = data.mean(axis=1)
    if sample_rate < 8000:
        raise ValueError(f"Tần số lấy mẫu {sample_rate} Hz quá thấp; cần ít nhất 8000 Hz.")
    if mono.size < sample_rate:
        raise ValueError("Âm thanh quá ngắn; cần ít nhất 1 giây.")
    peak = float(np.max(np.abs(mono)))
    if peak == 0:
        raise ValueError("File chỉ chứa im lặng.")
    return mono / peak, int(sample_rate), f"{channel_count} kênh" if channel_count > 1 else "Mono"


def add_noise(clean: np.ndarray, sample_rate: int, kind: str, snr_db: float) -> np.ndarray:
    silence_count = round(0.5 * sample_rate)
    padded = np.concatenate((np.zeros(silence_count, dtype=np.float32), clean))
    noise_power = float(np.mean(padded**2)) / (10 ** (snr_db / 10))
    time = np.arange(padded.size, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(0)
    white = np.sqrt(noise_power) * rng.standard_normal(padded.size)
    hum = np.sqrt(2 * noise_power) * np.sin(2 * np.pi * 50 * time)
    if kind == "Nhiễu trắng":
        noise = white
    elif kind == "Tiếng ù 50 Hz":
        noise = hum
    else:
        noise = np.sqrt(0.5) * white + np.sqrt(0.5) * hum
    return padded, padded + noise.astype(np.float32)


def filter_audio(noisy: np.ndarray, sample_rate: int, method: str, alpha: float) -> np.ndarray:
    if method == "Notch 50 Hz":
        numerator, denominator = signal.iirnotch(50, 35, fs=sample_rate)
        return signal.filtfilt(numerator, denominator, noisy).astype(np.float32)
    if method == "Thông dải 300-3400 Hz":
        sections = signal.butter(
            4, [300, 3400], btype="bandpass", fs=sample_rate, output="sos"
        )
        return signal.sosfiltfilt(sections, noisy).astype(np.float32)

    n_fft, hop_length = 512, 128
    window = hamming(n_fft, sym=False).astype(np.float32)
    spectrum = librosa.stft(
        noisy, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        window=window, center=True,
    )
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    silence_count = round(0.5 * sample_rate)
    noise_frames = max(1, (silence_count - n_fft) // hop_length + 1)
    noise_frames = min(noise_frames, magnitude.shape[1])
    noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    clean_magnitude = np.maximum(magnitude - alpha * noise_profile, 0.02 * magnitude)
    reconstructed = librosa.istft(
        clean_magnitude * np.exp(1j * phase), n_fft=n_fft,
        hop_length=hop_length, win_length=n_fft, window=window,
        center=True, length=noisy.size,
    )
    return reconstructed.astype(np.float32)


def measure_snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    error_power = float(np.sum((reference - estimate) ** 2))
    reference_power = float(np.sum(reference**2))
    if error_power == 0:
        return float("inf")
    return 10 * np.log10(reference_power / error_power)


def make_plots(clean: np.ndarray, noisy: np.ndarray, filtered: np.ndarray, sample_rate: int):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    fig.patch.set_facecolor("#0f172a")
    for axis in axes:
        axis.set_facecolor("#111c31")
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#475569")
        axis.tick_params(colors="#cbd5e1", labelsize=8)
        axis.xaxis.label.set_color("#f8fafc")
        axis.yaxis.label.set_color("#f8fafc")
        axis.title.set_color("#f8fafc")
        axis.grid(True, color="#334155", linewidth=0.7)

    time = np.arange(clean.size) / sample_rate
    stride = max(1, clean.size // 30000)
    axes[0].plot(time[::stride], noisy[::stride], color="#fb7185", alpha=0.65, lw=0.75, label="Có nhiễu")
    axes[0].plot(time[::stride], filtered[::stride], color="#34d399", lw=0.8, label="Sau lọc")
    axes[0].plot(time[::stride], clean[::stride], color="#60a5fa", alpha=0.85, lw=0.75, label="Sạch")
    axes[0].set(title="Dạng sóng", xlabel="Thời gian (s)", ylabel="Biên độ")
    axes[0].legend(frameon=False, ncol=3, fontsize=8, loc="upper right", labelcolor="#f8fafc")

    segment_length = min(1024, clean.size)
    frequencies, clean_psd = signal.welch(clean, sample_rate, nperseg=segment_length)
    _, noisy_psd = signal.welch(noisy, sample_rate, nperseg=segment_length)
    _, filtered_psd = signal.welch(filtered, sample_rate, nperseg=segment_length)
    for spectrum, label, color in (
        (clean_psd, "Sạch", "#60a5fa"),
        (noisy_psd, "Có nhiễu", "#fb7185"),
        (filtered_psd, "Sau lọc", "#34d399"),
    ):
        axes[1].plot(frequencies, 10 * np.log10(spectrum + np.finfo(float).eps), label=label, color=color, lw=1)
    axes[1].set(title="Phổ công suất · Welch", xlabel="Tần số (Hz)", ylabel="dB/Hz", xlim=(0, sample_rate / 2))
    axes[1].legend(frameon=False, ncol=3, fontsize=8, loc="upper right", labelcolor="#f8fafc")

    spec = librosa.stft(filtered, n_fft=512, hop_length=128, window="hann")
    spec_db = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
    img = axes[2].imshow(
        spec_db, origin="lower", aspect="auto", interpolation="nearest",
        extent=(0, filtered.size / sample_rate, 0, sample_rate / 2),
        cmap="magma", vmin=-80, vmax=0,
    )
    axes[2].set(title="Spectrogram · tín hiệu sau lọc", xlabel="Thời gian (s)", ylabel="Tần số (Hz)", ylim=(0, sample_rate / 2))
    colorbar = fig.colorbar(img, ax=axes[2], label="dB", pad=0.01)
    colorbar.ax.yaxis.label.set_color("#f8fafc")
    colorbar.ax.tick_params(colors="#cbd5e1")
    return fig


with st.sidebar:
    st.markdown('<div class="eyebrow">Âm thanh / xử lý tín hiệu</div>', unsafe_allow_html=True)
    st.header("Thiết lập")
    uploaded_file = st.file_uploader("Tệp nguồn · WAV", type=["wav"])
    noise_kind = st.selectbox("Loại nhiễu", ["Nhiễu trắng", "Tiếng ù 50 Hz", "Cả hai"])
    input_snr = st.slider("SNR đầu vào (dB)", min_value=-5.0, max_value=20.0, value=5.0, step=0.5)
    method = st.selectbox("Phương pháp lọc", ["Notch 50 Hz", "Thông dải 300-3400 Hz", "Trừ phổ"])
    alpha = st.slider("Alpha · trừ phổ", min_value=1.0, max_value=4.0, value=2.0, step=0.1, disabled=method != "Trừ phổ")
    process = st.button("Xử lý âm thanh", type="primary", use_container_width=True)
    st.caption("Nếu chưa tải WAV, ứng dụng dùng tín hiệu mẫu 16 kHz / 2 giây.")


st.markdown(
    '<div class="hero"><div class="eyebrow">Signal lab · 01</div>'
    '<h1>Lọc nhiễu âm thanh</h1>'
    '<p>Thêm nhiễu có kiểm soát, so sánh bộ lọc và nghe kết quả ngay trên trình duyệt.</p></div>',
    unsafe_allow_html=True,
)

if process or "audio_result" not in st.session_state:
    try:
        if uploaded_file is None:
            source = sample_signal(16000)
            sample_rate = 16000
            source_name = "Tín hiệu mẫu · 3 thành phần sin"
            channel_note = "Mono"
        else:
            source, sample_rate, channel_note = read_audio(uploaded_file)
            source_name = uploaded_file.name

        if sample_rate < 8000:
            raise ValueError("Tần số lấy mẫu cần ít nhất 8000 Hz để bộ lọc thông dải hoạt động.")
        clean, noisy = add_noise(source, sample_rate, noise_kind, input_snr)
        filtered = filter_audio(noisy, sample_rate, method, alpha)
        length = min(clean.size, noisy.size, filtered.size)
        clean, noisy, filtered = clean[:length], noisy[:length], filtered[:length]
        silence_count = round(0.5 * sample_rate)
        reference = clean[silence_count:]
        result = {
            "clean": clean, "noisy": noisy, "filtered": filtered,
            "sample_rate": sample_rate, "source_name": source_name,
            "channel_note": channel_note, "snr_before": measure_snr(clean, noisy),
            "snr_after": measure_snr(clean, filtered),
            "method": method, "noise_kind": noise_kind,
            "input_snr": input_snr, "alpha": alpha,
        }
        st.session_state.audio_result = result
    except Exception as exc:
        st.error(str(exc))
        st.stop()

result = st.session_state.audio_result
st.markdown(
    f'<div class="source-note">Nguồn: <b>{result["source_name"]}</b> &nbsp;·&nbsp; '
    f'{result["sample_rate"]:,} Hz &nbsp;·&nbsp; {result["channel_note"]} &nbsp;·&nbsp; '
    f'{result["method"]} / {result["noise_kind"]}</div>',
    unsafe_allow_html=True,
)

snr_before = result["snr_before"]
snr_after = result["snr_after"]
columns = st.columns(3)
columns[0].metric("SNR trước lọc", f"{snr_before:.2f} dB")
columns[1].metric("SNR sau lọc", f"{snr_after:.2f} dB")
columns[2].metric("Cải thiện", f"{snr_after - snr_before:+.2f} dB", delta=f"{snr_after - snr_before:+.2f} dB")

st.markdown("<br>", unsafe_allow_html=True)
audio_columns = st.columns(3)
audio_items = [
    ("Tín hiệu sạch", result["clean"]),
    ("Tín hiệu có nhiễu", result["noisy"]),
    ("Sau khi lọc", result["filtered"]),
]
for column, (label, samples) in zip(audio_columns, audio_items):
    column.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)
    column.audio(samples, sample_rate=result["sample_rate"])

chart = make_plots(result["clean"], result["noisy"], result["filtered"], result["sample_rate"])
st.pyplot(chart, use_container_width=True)
plt.close(chart)

st.caption("SNR được tính trên phần tín hiệu nguồn, không tính 0,5 giây im lặng dùng để ước lượng phổ nhiễu.")
