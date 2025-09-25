# -*- coding: utf-8 -*-
import os
import numpy as np
import librosa
import tensorflow as tf
from scipy.spatial.distance import cosine

# ---------------------------
# 参数配置
# ---------------------------
TFLITE_MODEL_PATH = r"D:\kend\myPython\speechDog-master\tiny\train_model_embading\checkpoints\mfcc_tdnn_triplet_savedmodel.tflite"
SAMPLE_RATE = 16000
TARGET_LEN = 0.5  # 增加到0.5秒以匹配预处理脚本
# N_MFCC = 39
HOP_LENGTH = 160
N_FFT = 512  # 与预处理脚本保持一致
TIME_FRAMES = 51  # 增加时间帧数以匹配预处理脚本
N_MFCC = 20       # 原始MFCC数量，与预处理脚本保持一致

# ---------------------------
# 工具函数
# ---------------------------
def load_and_pad(file, target_len=TARGET_LEN, sr=SAMPLE_RATE):
    y, orig_sr = librosa.load(file, sr=None, mono=True)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    target_samples = int(target_len * sr)
    if len(y) < target_samples:
        pad_len = target_samples - len(y)
        left = np.random.randint(0, pad_len)
        right = pad_len - left
        y = np.pad(y, (left, right), mode="constant")
    elif len(y) > target_samples:
        y = y[:target_samples]
    return y


def wav_to_mfcc(y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, time_frames=TIME_FRAMES):
    """转换为 MFCC 特征 (20+Δ+ΔΔ=60) 并固定时间帧"""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feat = np.vstack([mfcc, delta, delta2])
    feat = (feat - np.mean(feat)) / (np.std(feat) + 1e-6)

    # pad/crop 时间帧
    if feat.shape[1] < time_frames:
        pad_width = time_frames - feat.shape[1]
        feat = np.pad(feat, ((0,0),(0,pad_width)), mode='constant')
    elif feat.shape[1] > time_frames:
        feat = feat[:, :time_frames]

    return feat.astype(np.float32)

def preprocess_file(file):
    y = load_and_pad(file)
    mfcc = wav_to_mfcc(y)
    mfcc = np.expand_dims(mfcc, axis=0)   # batch
    mfcc = np.expand_dims(mfcc, axis=-1)  # channel
    return mfcc


# ---------------------------
# TFLite 推理
# ---------------------------
class TFLiteDogModel:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict_embedding(self, mfcc_input):
        self.interpreter.set_tensor(self.input_details[0]['index'], mfcc_input)
        self.interpreter.invoke()
        emb = self.interpreter.get_tensor(self.output_details[0]['index'])
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
            mfcc = preprocess_file(path)
            emb = model.predict_embedding(mfcc)
            feature_db[f] = emb
    print(f"注册完毕，共 {len(feature_db)} 条特征")
    return feature_db

def test_file(model, feature_db, file):
    mfcc = preprocess_file(file)
    emb = model.predict_embedding(mfcc)
    print(f"测试文件: {file}")
    
    # 存储相似度结果用于排序
    similarities = []
    for name, db_emb in feature_db.items():
        sim = 1 - cosine(emb, db_emb)
        similarities.append((name, sim))
    
    # 按相似度排序并显示前5个最相似的结果
    similarities.sort(key=lambda x: x[1], reverse=True)
    for i, (name, sim) in enumerate(similarities[:5]):
        print(f"相似度: {sim:.4f} -> {name}")

def test_path(model, feature_db, path):
    if os.path.isfile(path):
        test_file(model, feature_db, path)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in files:
                if not f.lower().endswith(".wav"):
                    continue
                file_path = os.path.join(root, f)
                test_file(model, feature_db, file_path)
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
        np.save("dog_feature_db.npy", feature_db)
        print("特征库保存为 dog_feature_db.npy")
    elif args.mode == "test":
        if not os.path.exists("dog_feature_db.npy"):
            raise FileNotFoundError("请先执行注册 mode")
        feature_db = np.load("dog_feature_db.npy", allow_pickle=True).item()
        test_path(model, feature_db, args.path)

"""
注册特征库：
python val_embedding_1dcnn.py register D:\kend\myPython\speechDog-master\datasets\compare_dog\dog01

# 测试单个文件
python val_embedding_1dcnn.py test D:\kend\myPython\speechDog-master\datasets\compare_dog\dog01\dog_bark_001.WAV

# 测试整个文件夹
python val_embedding_1dcnn.py test D:\work\datasets\tinyML\bark_origion
"""