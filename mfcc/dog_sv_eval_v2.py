# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/13 11:05
@Author  : Kend
@FileName: dog_sv_eval_v2
@Software: PyCharm
@modifier: CHATGPT
"""

"""
Dog Speaker Verification v2
改进点：
1. 特征 = MFCC + Δ + ΔΔ + log-Mel
2. CMVN 标准化
3. PCA 降维
4. 余弦相似度
5. 自动解析文件名里的 dog id
"""

import os
import argparse
import numpy as np
import librosa
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# ========== 特征提取 ==========
def extract_features(wav_path, sr=16000, n_mfcc=20, n_mels=40):
    y, sr = librosa.load(wav_path, sr=sr)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # log-Mel
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    logmel = librosa.power_to_db(mel)

    # 拼接
    feat = np.vstack([mfcc, delta1, delta2, logmel])

    # 时间平均池化
    feat = np.mean(feat, axis=1)

    return feat

# ========== 向量归一化 + PCA ==========
def build_embeddings(file_list, pca=None, scaler=None):
    feats = []
    ids = []
    for f in file_list:
        x = extract_features(f)
        feats.append(x)
        ids.append(parse_id_from_filename(os.path.basename(f)))
    feats = np.array(feats)

    # 标准化
    if scaler is None:
        scaler = StandardScaler().fit(feats)
    feats = scaler.transform(feats)

    # PCA
    if pca is None:
        pca = PCA(n_components=min(32, len(feats)))
        feats = pca.fit_transform(feats)
    else:
        feats = pca.transform(feats)

    return feats, ids, pca, scaler

# ========== 相似度 ==========
def cosine_score(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8)

# ========== 解析文件名 ==========
def parse_id_from_filename(fname):
    """
    假设文件名格式:
    - dog1_xxx.WAV --> true = dog1
    - bad.WAV / noise.WAV --> true = background
    """
    fname = fname.lower()
    if fname.startswith("dog"):
        return fname.split("_")[0]  # dog1, dog2, ...
    else:
        return "background"

# ========== 评估 ==========
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import numpy as np

def evaluate(enroll_feats, enroll_ids, test_feats, test_ids):
    scores = []
    labels = []
    for i, feat in enumerate(test_feats):
        sims = np.dot(enroll_feats, feat) / (np.linalg.norm(enroll_feats, axis=1) * np.linalg.norm(feat) + 1e-8)
        pred_id = enroll_ids[np.argmax(sims)]
        true_id = test_ids[i]
        scores.append(np.max(sims))
        labels.append(int(pred_id == true_id))

    # 保存分数
    import pandas as pd
    pd.DataFrame({"score": scores, "label": labels}).to_csv("scores_v2.csv", index=False)
    print("[INFO] scores saved to scores_v2.csv")

    # 如果没有负样本，就跳过 EER
    if len(set(labels)) < 2:
        print("[WARN] Only one class in y_true, cannot compute EER or ROC-AUC.")
        acc = accuracy_score([1] * len(labels), labels)  # 全部当正类
        print(f"[RESULT] Accuracy = {acc:.4f}")
        return

    # 计算 EER
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_threshold = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[eer_threshold]

    auc = roc_auc_score(labels, scores)
    acc = accuracy_score([1] * len(labels), labels)

    print(f"[RESULT] EER = {eer:.4f}, AUC = {auc:.4f}, Accuracy = {acc:.4f}")


# ========== 主程序 ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enroll_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    args = parser.parse_args()

    # 注册
    def collect_wavs(root_dir):
        wavs = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(".wav"):
                    wavs.append(os.path.join(root, f))
        return wavs


    # 注册
    enroll_files = collect_wavs(args.enroll_dir)
    if len(enroll_files) == 0:
        raise RuntimeError(f"[ERROR] No .wav files found in enroll_dir (including subfolders): {args.enroll_dir}")
    print(f"[INFO] Building enrollments from {len(enroll_files)} files...")
    enroll_feats, enroll_ids, pca, scaler = build_embeddings(enroll_files)

    # 测试
    test_files = collect_wavs(args.test_dir)
    if len(test_files) == 0:
        raise RuntimeError(f"[ERROR] No .wav files found in test_dir (including subfolders): {args.test_dir}")
    print(f"[INFO] Building tests from {len(test_files)} files...")
    test_feats, test_ids, _, _ = build_embeddings(test_files, pca=pca, scaler=scaler)

    # 测试
    test_files = [os.path.join(args.test_dir, f) for f in os.listdir(args.test_dir) if f.lower().endswith(".wav")]
    if len(test_files) == 0:
        raise RuntimeError(f"[ERROR] No .wav files found in test_dir: {args.test_dir}")
    print(f"[INFO] Building tests from {len(test_files)} files...")
    test_feats, test_ids, _, _ = build_embeddings(test_files, pca=pca, scaler=scaler)

    # 评估
    evaluate(enroll_feats, enroll_ids, test_feats, test_ids)


"""
python mfcc/dog_sv_eval_v2.py --enroll_dir ./youtube_wav/brakng_dog_datasets --test_dir ./youtube_wav/test

[INFO] Building enrollments from 25 files...
[INFO] Building tests from 9 files...
[INFO] Building tests from 9 files...
[INFO] scores saved to scores_v2.csv
[RESULT] EER = 0.6667, AUC = 0.5556, Accuracy = 0.6667
"""