# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/20 下午5:24
@Author  : Kend
@FileName: mel_plot.py
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

def plot_mel(audio_path, title):
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
        fmin=300,       # 下限频率（避免低频噪声） 狗吠是可以设置为300mhz的
        fmax=8000      # 上限频率（16kHz 的一半）
    )

    # 转换成 dB
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 可视化
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=HOP_LENGTH,
                             x_axis="time", y_axis="mel", fmax=8000)
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_mel(r"D:\kend\myPython\speechDog-master\youtube_wav\bark_segmentation\processed_template_preserving"
             r"\outdoor_braking_clean_preserving.wav", "braking")
    plot_mel(r"D:\work\datasets\tinyML\no_bark\no_dog_bark_013.wav", "no_braking")
