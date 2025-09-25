# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/20 下午6:36
@Author  : Kend
@FileName: val_embedding.py
@Software: PyCharm
@modifier:
"""


"""
TinyML 狗吠声纹验证推理脚本
功能：
1. 注册特征库：文件夹 -> embedding
2. 测试音频/特征 -> 输出与特征库的相似度
"""

import os
import numpy as np
import librosa
import tensorflow as tf
from scipy.spatial.distance import cosine

# ---------------------------
# 参数配置
# ---------------------------
TFLITE_MODEL_PATH = r"D:\kend\myPython\speechDog-master\tiny\tiny_bark_embedding.tflite"
SAMPLE_RATE = 16000
TARGET_LEN = 0.4
N_MELS = 40
HOP_LENGTH = 160
N_FFT = 400

# ---------------------------
# 工具函数
# ---------------------------
def load_and_pad(file, target_len=TARGET_LEN, sr=SAMPLE_RATE):
    y, orig_sr = librosa.load(file, sr=None, mono=True)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)  # ✅ 注意关键字参数
    target_samples = int(target_len * sr)
    if len(y) < target_samples:
        pad_len = target_samples - len(y)
        left = np.random.randint(0, pad_len)
        right = pad_len - left
        y = np.pad(y, (left, right), mode="constant")
    elif len(y) > target_samples:
        y = y[:target_samples]
    return y


def wav_to_mel(y, sr=SAMPLE_RATE, n_mels=N_MELS, time_frames=64):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 补零或裁剪到固定时间帧数
    if mel_db.shape[1] < time_frames:
        pad_width = time_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0,0),(0,pad_width)), mode='constant')
    elif mel_db.shape[1] > time_frames:
        mel_db = mel_db[:, :time_frames]

    return mel_db.astype(np.float32)


def preprocess_file(file):
    y = load_and_pad(file)
    mel = wav_to_mel(y)
    mel = np.expand_dims(mel, axis=-1)  # NHWC
    mel = np.expand_dims(mel, axis=0)   # batch
    return mel

# ---------------------------
# TFLite 推理
# ---------------------------
class TFLiteDogModel:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict_embedding(self, mel_input):
        self.interpreter.set_tensor(self.input_details[0]['index'], mel_input)
        self.interpreter.invoke()
        emb = self.interpreter.get_tensor(self.output_details[0]['index'])
        # L2 正则化
        emb = emb / np.linalg.norm(emb, ord=2, axis=1, keepdims=True)
        return emb[0]

# ---------------------------
# 特征库管理
# ---------------------------
def register_folder(model, folder):
    feature_db = {}
    for root, _, files in os.walk(folder):
        for f in files:
            if not f.lower().endswith(".wav"):
                continue
            path = os.path.join(root, f)
            mel = preprocess_file(path)
            emb = model.predict_embedding(mel)
            feature_db[f] = emb
    print(f"注册完毕，共 {len(feature_db)} 条特征")
    return feature_db

def test_file(model, feature_db, file):
    mel = preprocess_file(file)
    emb = model.predict_embedding(mel)
    print(f"测试文件: {file}")
    for name, db_emb in feature_db.items():
        sim = 1 - cosine(emb, db_emb)
        print(f"相似度: {sim:.4f} -> {name}")


def test_path(model, feature_db, path):
    if os.path.isfile(path):
        # 单文件测试
        mel = preprocess_file(path)
        emb = model.predict_embedding(mel)
        print(f"测试文件: {path}")
        for name, db_emb in feature_db.items():
            sim = 1 - cosine(emb, db_emb)
            print(f"相似度: {sim:.4f} -> {name}")
    elif os.path.isdir(path):
        # 文件夹测试
        for root, _, files in os.walk(path):
            for f in files:
                if not f.lower().endswith(".wav"):
                    continue
                file_path = os.path.join(root, f)
                mel = preprocess_file(file_path)
                emb = model.predict_embedding(mel)
                print(f"\n测试文件: {file_path}")
                for name, db_emb in feature_db.items():
                    sim = 1 - cosine(emb, db_emb)
                    print(f"相似度: {sim:.4f} -> {name}")
    else:
        raise ValueError("路径不是文件也不是文件夹")



# ---------------------------
# 主流程
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["register", "test"], help="register 文件夹 / test 文件")
    parser.add_argument("path", help="注册文件夹或测试文件路径")
    args = parser.parse_args()

    model = TFLiteDogModel(TFLITE_MODEL_PATH)

    if args.mode == "register":
        feature_db = register_folder(model, args.path)
        # 保存特征库
        np.save("dog_feature_db.npy", feature_db)
        print("特征库保存为 dog_feature_db.npy")
    elif args.mode == "test":
        if not os.path.exists("dog_feature_db.npy"):
            raise FileNotFoundError("请先执行注册 mode")
        feature_db = np.load("dog_feature_db.npy", allow_pickle=True).item()
        test_path(model, feature_db, args.path)

"""
注册特征库：
    python val_embedding.py register D:\kend\myPython\speechDog-master\datasets\compare_dog\dog01

# 测试单个文件
python val_embedding.py test D:\kend\myPython\speechDog-master\datasets\compare_dog\dog01\dog_bark_001.WAV

# 测试整个文件夹
python val_embedding.py test D:\work\datasets\tinyML\bark_origion

"""
