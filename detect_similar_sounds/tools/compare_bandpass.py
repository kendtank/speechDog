# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/17 下午1:58
@Author  : Kend
@FileName: compare_bandpass.py
@Software: PyCharm
@modifier:
"""

"""
巴特沃斯带通滤波器, 对比处理之后的音频之后的效果
"""

import os
# 把项目的路径设置为当前路径
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
print(ROOT)
os.chdir(ROOT)


import soundfile as sf
import librosa.display
import numpy as np
from matplotlib import pyplot as plt
from detect_similar_sounds.bark_segmentation import bandpass

# 1. 读取原始音频
y, sr = sf.read("youtube_wav/bark_segmentation/bark01.mp3")
print("音频时长: ", len(y) / sr, "秒")  # 音频时长:  5.76 秒
# 查看是不是双声道
print("音频声道数: ", y.shape[1])  # 音频声道数:  2

y = y.mean(axis=1)  # 降为单声道，左右声道取平均， 更稳健


# 2. 带通滤波
y_filt = bandpass(y, sr=sr, low=200, high=7500, order=4)


# 可视化
# 3. 可视化对比
plt.figure(figsize=(12, 8))

# 波形对比
plt.subplot(2, 1, 1)
librosa.display.waveshow(y, sr=sr, alpha=0.5, label="origin")
librosa.display.waveshow(y_filt, sr=sr, alpha=0.8, label="filtered")
plt.title("wave-show")
plt.legend()

# 频谱对比
plt.subplot(2, 1, 2)
fft_orig = np.abs(np.fft.rfft(y))
fft_filt = np.abs(np.fft.rfft(y_filt))
freqs = np.fft.rfftfreq(len(y), 1 / sr)

plt.semilogy(freqs, fft_orig, label="origin")
plt.semilogy(freqs, fft_filt, label="filtered")
plt.title("semi-logy")
plt.xlabel("Hz")
plt.ylabel("high")
plt.legend()

plt.tight_layout()
plt.show()


# 3. 保存结果，方便听对比
# sf.write(f"{ROOT}\\youtube_wav\\bark_segmentation\\bark01_filtered.wav", y_filt, sr)