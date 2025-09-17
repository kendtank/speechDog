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


def evaluate(enroll_feats, enroll_ids, test_feats, test_ids, test_files, threshold=0.0):
    # 打印注册信息
    print("[INFO] Building enrollments...")
    unique_enroll_ids = sorted(list(set(enroll_ids)))
    print(f"[INFO] enrolled ids: {unique_enroll_ids}")
    
    # 打印测试信息
    test_filenames = [os.path.basename(f) for f in test_files]
    print("[INFO] Building tests...")
    print(f"[INFO] test files: {test_filenames}")
    
    scores = []
    labels = []
    known_scores = []
    known_labels = []
    
    for i, feat in enumerate(test_feats):
        # 计算与所有注册样本的相似度
        sims = np.dot(enroll_feats, feat) / (np.linalg.norm(enroll_feats, axis=1) * np.linalg.norm(feat) + 1e-8)
        
        # 对每个狗ID，取其所有注册样本的最大相似度
        id_to_max_sim = {}
        for j, eid in enumerate(enroll_ids):
            if eid not in id_to_max_sim:
                id_to_max_sim[eid] = sims[j]
            else:
                id_to_max_sim[eid] = max(id_to_max_sim[eid], sims[j])
        
        # 获取预测ID和分数
        pred_id = max(id_to_max_sim, key=id_to_max_sim.get)
        score = id_to_max_sim[pred_id]
        
        # 如果使用阈值且最高分数低于阈值，则预测为未知
        if threshold > 0 and score < threshold:
            pred_id = "unknown"
        
        # 获取真实ID
        true_id = test_ids[i]
        
        # 保存分数和标签 (所有测试样本)
        scores.append(score)
        labels.append(1 if true_id != "background" else 0)  # 1表示正类(狗叫声)，0表示负类(非狗叫声)
        
        # 仅对已知狗ID计算识别准确率
        if true_id != "background":
            known_scores.append(score)
            known_labels.append(int(pred_id == true_id))
        
        # 打印当前测试文件的结果
        filename = test_filenames[i]
        print(f"{filename} --> pred={pred_id}, score={score:.4f}, true={true_id}")
        
        # 打印所有分数
        all_scores = {}
        for eid in unique_enroll_ids:
            all_scores[eid] = round(id_to_max_sim.get(eid, 0), 4)
        print(f"  All scores: {all_scores}")

    # 分类统计
    total = len(test_feats)
    background = sum(1 for id in test_ids if id == "background")
    known = total - background
    print(f"\n[INFO] Test samples: {total} total ({known} dogs, {background} background)")
    
    # 如果没有负样本，就跳过 EER
    if len(set(labels)) < 2:
        print("[WARN] Only one class in y_true, cannot compute EER or ROC-AUC.")
        acc = accuracy_score([1] * len(known_labels), known_labels)  # 全部当正类
        print(f"[RESULT] Accuracy = {acc:.4f}")
        return

    # 计算 EER
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_threshold = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[eer_threshold]

    auc = roc_auc_score(labels, scores)
    
    # 计算准确率（使用0.5作为阈值）
    pred_labels = [1 if s > 0.5 else 0 for s in scores]
    acc = accuracy_score(labels, pred_labels)
    
    # 计算已知狗ID的准确率
    known_acc = accuracy_score(known_labels, [1] * len(known_labels)) if known_labels else 0
    
    print(f"[RESULT] EER = {eer:.4f}, AUC = {auc:.4f}, Detection Accuracy = {acc:.4f}, Recognition Accuracy = {known_acc:.4f}")


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

    # # 测试
    # test_files = [os.path.join(args.test_dir, f) for f in os.listdir(args.test_dir) if f.lower().endswith(".wav")]
    # if len(test_files) == 0:
    #     raise RuntimeError(f"[ERROR] No .wav files found in test_dir: {args.test_dir}")
    # print(f"[INFO] Building tests from {len(test_files)} files...")
    # test_feats, test_ids, _, _ = build_embeddings(test_files, pca=pca, scaler=scaler)

    # 评估
    evaluate(enroll_feats, enroll_ids, test_feats, test_ids, test_files)


"""
python mfcc/dog_sv_eval_v2.py --enroll_dir ./youtube_wav/brakng_dog_datasets --test_dir ./youtube_wav/test

[INFO] Building enrollments from 25 files...
[INFO] Building tests from 9 files...
[INFO] Building enrollments...
[INFO] enrolled ids: ['dog1', 'dog2', 'dog3', 'dog4', 'dog5']
[INFO] Building tests...
[INFO] test files: ['bad.WAV', 'bad2.WAV', 'dog1_test_01.WAV', 'dog2_test_01.WAV', 'dog3_test_01.WAV', 'dog4_test_01.WAV', 'dog4_test_02.WAV', 'dog5_test_01.WAV', 'dog5_test_02.WAV']
bad.WAV --> pred=dog5, score=0.9425, true=background
  All scores: {'dog1': -1e-04, 'dog2': 0.3699, 'dog3': 0.1708, 'dog4': 0.3515, 'dog5': 0.9425}
bad2.WAV --> pred=dog5, score=0.9598, true=background
  All scores: {'dog1': 0.1324, 'dog2': 0.3043, 'dog3': 0.1545, 'dog4': 0.3183, 'dog5': 0.9598}
dog1_test_01.WAV --> pred=dog1, score=1.0000, true=dog1
  All scores: {'dog1': 1.0, 'dog2': 0.2088, 'dog3': -0.0142, 'dog4': -0.1635, 'dog5': -0.0018}
dog2_test_01.WAV --> pred=dog5, score=0.5658, true=dog2
  All scores: {'dog1': 0.0123, 'dog2': 0.4969, 'dog3': 0.1422, 'dog4': 0.1122, 'dog5': 0.5658}
dog3_test_01.WAV --> pred=dog3, score=0.7644, true=dog3
  All scores: {'dog1': 0.0915, 'dog2': -0.099, 'dog3': 0.7644, 'dog4': 0.2796, 'dog5': 0.1477}
dog4_test_01.WAV --> pred=dog4, score=0.8554, true=dog4
  All scores: {'dog1': -0.1024, 'dog2': -0.2609, 'dog3': 0.2613, 'dog4': 0.8554, 'dog5': 0.0877}
dog4_test_02.WAV --> pred=dog4, score=0.8464, true=dog4
  All scores: {'dog1': -0.125, 'dog2': 0.0178, 'dog3': 0.2073, 'dog4': 0.8464, 'dog5': 0.5945}
dog5_test_01.WAV --> pred=dog5, score=0.8906, true=dog5
  All scores: {'dog1': -0.2008, 'dog2': 0.051, 'dog3': -0.0605, 'dog4': 0.1209, 'dog5': 0.8906}
dog5_test_02.WAV --> pred=dog5, score=0.9690, true=dog5
  All scores: {'dog1': 0.0557, 'dog2': 0.3547, 'dog3': 0.1405, 'dog4': 0.3048, 'dog5': 0.969}

[INFO] Test samples: 9 total (7 dogs, 2 background)
[RESULT] EER = 1.0000, AUC = 0.2857, Detection Accuracy = 0.7778, Recognition Accuracy = 0.8571



"""