# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/16 16:00
@Author  : Kend & Qwen
@FileName: dog_gmm_ubm_enhanced.py
@Software: PyCharm
"""

"""
Enhanced Dog Bark Recognition using GMM-UBM with improved features
"""

import os
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import argparse
import csv

# ================== 配置 ==================
SR = 16000
N_MFCC = 20
N_MELS = 40
FRAME_LEN = 0.025
FRAME_STEP = 0.01
DOG_GMM_COMPONENTS = 3
UBM_COMPONENTS = 16
EPS = 1e-8
BACKGROUND_THRESH = -85

# ================== 音频加载与预处理 ==================
def load_audio(path):
    y, _ = librosa.load(path, sr=SR)
    # 去除静音段
    y, _ = librosa.effects.trim(y, top_db=20)
    
    # 如果音频太短，进行重复延长
    if len(y) < SR * 0.1:
        repeat_times = int(np.ceil((SR * 0.1) / len(y)))
        y = np.tile(y, repeat_times)
        y = y[:int(SR * 0.1)]
    
    return y

# ================== 增强特征提取 ==================
def extract_enhanced_features(y):
    # 计算帧数，确保所有特征使用相同的帧参数
    n_fft = int(SR * FRAME_LEN)
    hop_length = int(SR * FRAME_STEP)
    
    # MFCC特征
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SR,
        n_mfcc=N_MFCC,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=N_MELS,
        htk=True
    )
    
    # Delta和Delta-Delta特征
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # 谱质心
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=n_fft, hop_length=hop_length)
    
    # 谱带宽
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SR, n_fft=n_fft, hop_length=hop_length)
    
    # 谱滚降点
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR, n_fft=n_fft, hop_length=hop_length)
    
    # 零交叉率
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)
    
    # 谱平坦度
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)
    
    # 确保所有特征具有相同的帧数
    min_frames = min(
        mfcc.shape[1], 
        delta.shape[1], 
        delta2.shape[1],
        spectral_centroids.shape[1],
        spectral_bandwidth.shape[1],
        spectral_rolloff.shape[1],
        zcr.shape[1],
        flatness.shape[1]
    )
    
    # 截取所有特征到相同的帧数
    mfcc = mfcc[:, :min_frames]
    delta = delta[:, :min_frames]
    delta2 = delta2[:, :min_frames]
    spectral_centroids = spectral_centroids[:, :min_frames]
    spectral_bandwidth = spectral_bandwidth[:, :min_frames]
    spectral_rolloff = spectral_rolloff[:, :min_frames]
    zcr = zcr[:, :min_frames]
    flatness = flatness[:, :min_frames]
    
    # 拼接所有特征
    features = np.vstack([
        mfcc, delta, delta2,
        spectral_centroids,
        spectral_bandwidth,
        spectral_rolloff,
        zcr,
        flatness
    ])
    
    # 确保有足够的帧
    if features.shape[1] < 5:
        repeat_times = int(np.ceil(5 / features.shape[1]))
        features = np.tile(features, (1, repeat_times))
        features = features[:, :5]
    
    return features.T  # 转置为 (帧数, 特征维度)

# ================== 构建训练集 ==================
def build_enroll(enroll_dir):
    enroll_data = {}
    all_feats = []
    for pid in os.listdir(enroll_dir):
        pdir = os.path.join(enroll_dir, pid)
        if not os.path.isdir(pdir): 
            continue
        feats = []
        for f in os.listdir(pdir):
            if not f.lower().endswith(('.wav', '.flac')): 
                continue
            try:
                y = load_audio(os.path.join(pdir, f))
                feat = extract_enhanced_features(y)
                feats.append(feat)
                all_feats.append(feat)
            except Exception as e:
                print(f"[WARN] 处理文件 {f} 时出错: {e}")
                continue
        if feats:
            enroll_data[pid] = np.vstack(feats)
    return enroll_data, np.vstack(all_feats)

# ================== 训练 GMM-UBM ==================
def train_gmms(enroll_data, ubm_feats):
    print(f"[INFO] Training UBM with {ubm_feats.shape[0]} frames...")
    
    # 标准化特征
    scaler = StandardScaler()
    ubm_feats = scaler.fit_transform(ubm_feats)
    
    # 训练UBM
    ubm = GaussianMixture(
        n_components=UBM_COMPONENTS,
        covariance_type='diag',
        max_iter=200,
        reg_covar=1e-3,
        random_state=42
    )
    ubm.fit(ubm_feats)

    # 训练每只狗的GMM
    dog_gmms = {}
    for pid, feats in enroll_data.items():
        # 标准化特征
        feats = scaler.transform(feats)
        
        gmm = GaussianMixture(
            n_components=DOG_GMM_COMPONENTS,
            covariance_type='diag',
            max_iter=200,
            reg_covar=1e-3,
            random_state=42
        )

        # 用 UBM 参数初始化再适应
        if DOG_GMM_COMPONENTS <= UBM_COMPONENTS:
            gmm.means_init = ubm.means_[:DOG_GMM_COMPONENTS]
        gmm.fit(feats)
        dog_gmms[pid] = (gmm, scaler)  # 保存GMM和对应的标准化器
        print(f"[INFO] built model for {pid} with {feats.shape[0]} frames")
    return ubm, dog_gmms

# ================== 测试 ==================
def test_gmms(test_dir, dog_gmms):
    scores_all = []
    for f in os.listdir(test_dir):
        if not f.lower().endswith(('.wav', '.flac')): 
            continue
        try:
            y = load_audio(os.path.join(test_dir, f))
            feat = extract_enhanced_features(y)
            
            scores = {}
            for pid, (gmm, scaler) in dog_gmms.items():
                # 标准化特征
                scaled_feat = scaler.transform(feat)
                # 计算平均对数似然
                score = gmm.score(scaled_feat)
                scores[pid] = score
            
            # 选择最大得分狗
            best_pid = max(scores, key=scores.get)
            best_score = scores[best_pid]
            
            # 背景判定
            if best_score < BACKGROUND_THRESH:
                best_pid = 'background'
            
            print(f"{f} --> pred={best_pid}, scores={ {k: round(v,4) for k,v in scores.items()} }")
            scores_all.append((f, best_pid, scores))
        except Exception as e:
            print(f"[WARN] 处理测试文件 {f} 时出错: {e}")
            continue
    return scores_all

# ================== 评估指标 ==================
def evaluate(scores_all, true_func):
    y_true, y_score = [], []
    for fname, pred, scores in scores_all:
        true_label = true_func(fname)
        max_score = max(scores.values())
        
        # 对于目标狗，正确预测为1，错误预测为0
        # 对于背景，正确预测为1，错误预测为0
        if true_label == 'background':
            y_true.append(1 if pred == 'background' else 0)
        else:
            y_true.append(1 if pred == true_label else 0)
        y_score.append(max_score)
    
    if len(set(y_true)) < 2:
        print("[WARN] 只有一类标签，无法计算ROC指标")
        return None, None, None
        
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # ROC / EER
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    try:
        idx = np.nanargmin(np.abs(fnr - fpr))
        eer = (fpr[idx] + fnr[idx]) / 2
        thr = thresholds[idx]
    except:
        eer, thr = np.nan, np.nan
    auc = roc_auc_score(y_true, y_score)
    acc = accuracy_score(y_true, [1 if s > thr else 0 for s in y_score])
    print(f"[RESULT] EER = {eer:.4f}, AUC = {auc:.4f}, Accuracy = {acc:.4f}")
    return eer, auc, acc

# ================== CSV 保存 ==================
def save_scores(scores_all, path='scores_enhanced.csv'):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        if scores_all:
            header = ['filename', 'pred'] + list(scores_all[0][2].keys())
            writer.writerow(header)
            for fname, pred, scores in scores_all:
                row = [fname, pred] + [scores[k] for k in scores]
                writer.writerow(row)
    print(f"[INFO] scores saved to {path}")

# ================== 真值函数 ==================
def true_label_func(fname):
    # 文件名规则：dog1_test_01.WAV --> dog1
    if fname.lower().startswith('dog'):
        return fname.split('_')[0]
    else:
        return 'background'

# ================== 主函数 ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--enroll_dir', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    args = parser.parse_args()

    enroll_data, ubm_feats = build_enroll(args.enroll_dir)
    ubm, dog_gmms = train_gmms(enroll_data, ubm_feats)
    scores_all = test_gmms(args.test_dir, dog_gmms)
    save_scores(scores_all)
    evaluate(scores_all, true_label_func)

"""
python gmm_ubm/dog_gmm_ubm_enhanced.py --enroll_dir ./youtube_wav/brakng_dog_datasets --test_dir ./youtube_wav/test
"""