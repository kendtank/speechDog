# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/22 下午3:39
@Author  : Kend
@FileName: log_mel_plot.py
@Software: PyCharm
@modifier:
"""



import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# ====== 参数 ======
SAMPLE_RATE = 16000
N_FFT = 400        # 25ms 窗口
HOP_LENGTH = 160   # 10ms 步长
N_MELS = 64        # Mel 频带数

def plot_log_mel(audio_path, title):
    # 加载音频
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    print(f"{title} - 时长: {len(y)/sr:.2f}s, 采样点数: {len(y)}")

    # 生成 Mel 频谱
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=300,      # 狗吠能量主要集中在 300Hz 以上，可以过滤低频噪声
        fmax=8000      # 上限频率（16kHz 的一半）
    )

    # ====== 取 log 变换得到 log-Mel ======
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # 可视化
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mel_spec, sr=sr, hop_length=HOP_LENGTH,
                             x_axis="time", y_axis="mel", fmax=8000, cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Log-Mel Spectrogram - {title}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_log_mel(
        r"D:\kend\myPython\speechDog-master\youtube_wav\bark_segmentation\processed_template_preserving\outdoor_braking_clean_preserving.wav",
        "Dog Bark"
    )
    plot_log_mel(
        r"D:\work\datasets\tinyML\no_bark\no_dog_bark_013.wav",
        "No Bark"
    )

