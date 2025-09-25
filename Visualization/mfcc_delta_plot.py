# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/22 下午3:43
@Author  : Kend
@FileName: mfcc_delta_plot.py
@Software: PyCharm
@modifier:
"""



# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/25
@Author  : Kend
@FileName: mfcc_delta_plot.py
@Software: PyCharm
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# ====== 参数 ======
SAMPLE_RATE = 16000
N_FFT = 400        # 25ms 窗口
HOP_LENGTH = 160   # 10ms 步长
N_MFCC = 13        # 基础 MFCC 维度

def plot_mfcc_delta(audio_path, title):
    # 加载音频
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    print(f"{title} - 时长: {len(y)/sr:.2f}s, 采样点数: {len(y)}")

    # 计算基础 MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmin=300,
        fmax=8000
    )

    # 计算 Δ 和 ΔΔ
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # 拼接 (13 + 13 + 13 = 39)
    mfcc_feat = np.vstack([mfcc, delta, delta2])

    # 可视化
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(mfcc_feat, sr=sr, hop_length=HOP_LENGTH,
                             x_axis="time", cmap="viridis")
    plt.colorbar(format="%+2.0f")
    plt.title(f"MFCC + Δ + ΔΔ (39 dims) - {title}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_mfcc_delta(
        r"D:\kend\myPython\speechDog-master\youtube_wav\bark_segmentation\processed_template_preserving\outdoor_braking_clean_preserving.wav",
        "Dog Bark"
    )
    plot_mfcc_delta(
        r"D:\work\datasets\tinyML\no_bark\no_dog_bark_013.wav",
        "No Bark"
    )
