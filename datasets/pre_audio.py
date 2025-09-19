# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/19 下午3:19
@Author  : Kend
@FileName: pre_audio.py
@Software: PyCharm
@modifier:
"""



import os
import soundfile as sf
import librosa

# 输入输出文件夹
in_dir = "D:/work/datasets/tinyML/bark_origion"
out_dir = "D:/work/datasets/tinyML/bark_16k"
os.makedirs(out_dir, exist_ok=True)

target_sr = 16000  # 统一采样率
min_dur, max_dur = 0.1, 1.0  # 时长范围

report = []

for fname in os.listdir(in_dir):
    if not fname.lower().endswith(".wav"):
        continue

    in_path = os.path.join(in_dir, fname)
    out_path = os.path.join(out_dir, fname)

    # 加载音频（自动转单声道）
    y, sr = librosa.load(in_path, sr=target_sr, mono=True)

    # 保存为 PCM16 WAV
    sf.write(out_path, y, target_sr, subtype="PCM_16")

    # 获取时长信息
    dur = len(y) / target_sr
    ok = (min_dur <= dur <= max_dur)

    report.append((fname, sr, len(y), dur, "OK" if ok else "BAD"))

# 打印报表
print(f"{'File':40s} {'SR':6s} {'Frames':10s} {'Dur(s)':10s} {'Check'}")
for r in report:
    print(f"{r[0]:40s} {r[1]:6d} {r[2]:10d} {r[3]:10.3f} {r[4]}")
