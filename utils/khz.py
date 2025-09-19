# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/19 上午11:17
@Author  : Kend
@FileName: khz.py
@Software: PyCharm
@modifier:
"""


import soundfile as sf

file_path = r"D:\work\datasets\braking_dog\#Labrador Dog Barking clip #Dog traing - PVNFOUJI.mp3"
f = sf.SoundFile(file_path)

print("采样率:", f.samplerate)      # 期望 >=16000
print("声道数:", f.channels)        # 期望 1（单声道）
print("样本数:", len(f))            # 音频总长度（samples）
print("时长(秒):", len(f)/f.samplerate)
