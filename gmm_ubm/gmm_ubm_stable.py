# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/13 11:33
@Author  : Kend
@FileName: gmm_ubm_stable
@Software: PyCharm
@modifier:
"""


import os
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.special import logsumexp
import argparse
import random

# -----------------------------
# 参数设置
# -----------------------------
N_MFCC = 13
DELTA_ORDER = 2
CONTEXT = 5      # 上下文帧拼接
UBM_COMPONENTS = 8
DOG_GMM_COMPONENTS = 2
REG_COVAR = 1e-3
SAMPLE_RATE = 16000

# -----------------------------
# 特征提取
# -----------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    # 数据增强
    if random.random() < 0.5:
        y = y + 0.005*np.random.randn(len(y))   # 加噪声
    if random.random() < 0.5:
        y = np.roll(y, random.randint(-1000,1000))  # 随机时移
    if random.random() < 0.5:
        y = y * random.uniform(0.7,1.3)   # 音量扰动

    # mfcc_feat = librosa.feature.mfcc(y, sr=sr, n_mfcc=N_MFCC)
    mfcc_feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    delta_feat = librosa.feature.delta(mfcc_feat, order=1)
    delta2_feat = librosa.feature.delta(mfcc_feat, order=2)
    feat = np.vstack([mfcc_feat, delta_feat, delta2_feat]).T  # shape: frames x feat_dim

    # 上下文拼接
    frames = []
    for i in range(len(feat)-CONTEXT+1):
        frames.append(feat[i:i+CONTEXT].flatten())
    return np.array(frames)

# -----------------------------
# 构建数据集
# -----------------------------
def build_dataset(enroll_dir):
    enroll_data = {}
    for dog_id in sorted(os.listdir(enroll_dir)):
        dog_path = os.path.join(enroll_dir, dog_id)
        if not os.path.isdir(dog_path):
            continue
        feats_list = []
        for fname in os.listdir(dog_path):
            if fname.lower().endswith('.wav'):
                feats_list.append(extract_features(os.path.join(dog_path, fname)))
        if feats_list:
            enroll_data[dog_id] = np.vstack(feats_list)
    return enroll_data

# -----------------------------
# 训练UBM
# -----------------------------
def train_ubm(enroll_data):
    all_feats = np.vstack(list(enroll_data.values()))
    scaler = StandardScaler().fit(all_feats)
    all_feats = scaler.transform(all_feats)
    ubm = GaussianMixture(n_components=UBM_COMPONENTS, covariance_type='diag',
                          max_iter=200, reg_covar=REG_COVAR, random_state=42)
    ubm.fit(all_feats)
    return ubm, scaler

# -----------------------------
# MAP 微调狗GMM
# -----------------------------
def train_dog_gmms(enroll_data, ubm, scaler):
    dog_gmms = {}
    for dog_id, feats in enroll_data.items():
        feats = scaler.transform(feats)
        gmm = GaussianMixture(n_components=DOG_GMM_COMPONENTS, covariance_type='diag',
                              max_iter=100, reg_covar=REG_COVAR, random_state=42,
                              means_init=ubm.means_[:DOG_GMM_COMPONENTS])
        gmm.fit(feats)
        dog_gmms[dog_id] = gmm
    return dog_gmms

# -----------------------------
# 测试
# -----------------------------
def score_sample(sample_feat, dog_gmms, scaler):
    sample_feat = scaler.transform(sample_feat)
    scores = {}
    for dog_id, gmm in dog_gmms.items():
        log_likelihood = gmm.score(sample_feat)  # 平均log-likelihood
        scores[dog_id] = log_likelihood

    # 将log-likelihood转换为0-1范围得分，1为最大值
    # 方法：先取负值，然后进行min-max标准化
    score_values = np.array(list(scores.values()))
    min_score = np.min(score_values)
    max_score = np.max(score_values)

    # 如果所有得分相同，则都设为1
    if max_score == min_score:
        normalized_scores = {dog_id: 1.0 for dog_id in scores.keys()}
    else:
        # 标准化到0-1范围
        normalized_scores = {
            dog_id: (score - min_score) / (max_score - min_score)
            for dog_id, score in scores.items()
        }
        # 转换为"1为最大"的得分（越高越好）
        # 当前最大值设为1，其他按比例调整
        max_normalized = max(normalized_scores.values())
        normalized_scores = {
            dog_id: score / max_normalized
            for dog_id, score in normalized_scores.items()
        }

    # 预测（仍然是基于原始log-likelihood）
    pred = max(scores, key=scores.get)
    return pred, normalized_scores


# -----------------------------
# 主流程
# -----------------------------
def main(enroll_dir, test_dir):
    print("[INFO] Building enrollments...")
    enroll_data = build_dataset(enroll_dir)
    if not enroll_data:
        print("[ERROR] No enrollment data found!")
        return
    print(f"[INFO] enrolled ids: {list(enroll_data.keys())}")

    print("[INFO] Training UBM...")
    ubm, scaler = train_ubm(enroll_data)

    print("[INFO] Training per-dog GMMs...")
    dog_gmms = train_dog_gmms(enroll_data, ubm, scaler)

    print("[INFO] Building tests...")
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.wav')]
    results = []
    for fname in test_files:
        feat = extract_features(os.path.join(test_dir, fname))
        pred, scores = score_sample(feat, dog_gmms, scaler)
        # 获取true label
        true = fname.split('_')[0] if 'dog' in fname.lower() else 'background'
        print(f"{fname} --> pred={pred}, true={true}, scores={scores}")
        results.append((fname, pred, true, scores))

    # 保存结果
    import csv
    with open('scores_stable.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file','pred','true','scores'])
        for row in results:
            writer.writerow(row)
    print("[INFO] scores saved to scores_stable.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--enroll_dir', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    args = parser.parse_args()
    main(args.enroll_dir, args.test_dir)

"""
python  gmm_ubm/gmm_ubm_stable.py --enroll_dir ./youtube_wav/brakng_dog_datasets --test_dir ./youtube_wav/test

运行结果：
dog1_test_01.WAV --> pred=dog1, true=dog1, scores={'dog1': np.float64(1.0), 'dog2': np.float64(0.8225606452217356), 'dog3': np.float64(0.4783729727364495), 'dog4': np.float64(0.0), 'dog5': np.float64(0.5170179358950694)}
dog4_test_02.WAV --> pred=dog4, true=dog4, scores={'dog1': np.float64(0.5055825952058531), 'dog2': np.float64(0.5451843929493533), 'dog3': np.float64(0.0), 'dog4': np.float64(1.0), 'dog5': np.float64(0.12077160189907014)}
dog5_test_02.WAV --> pred=dog5, true=dog5, scores={'dog1': np.float64(0.7699150019445676), 'dog2': np.float64(0.9963699054441245), 'dog3': np.float64(0.5956563184370872), 'dog4': np.float64(0.0), 'dog5': np.float64(1.0)}
dog2_test_01.WAV --> pred=dog2, true=dog2, scores={'dog1': np.float64(0.720522588265588), 'dog2': np.float64(1.0), 'dog3': np.float64(0.4780998726826498), 'dog4': np.float64(0.0), 'dog5': np.float64(0.6898904656956675)}
bad.WAV --> pred=dog2, true=background, scores={'dog1': np.float64(0.8837768290120195), 'dog2': np.float64(1.0), 'dog3': np.float64(0.7436357715140514), 'dog4': np.float64(0.0), 'dog5': np.float64(0.9747000105629228)}
dog4_test_01.WAV --> pred=dog4, true=dog4, scores={'dog1': np.float64(0.6385572026341149), 'dog2': np.float64(0.7120477537440079), 'dog3': np.float64(0.0), 'dog4': np.float64(1.0), 'dog5': np.float64(0.1660319252120226)}
dog3_test_01.WAV --> pred=dog3, true=dog3, scores={'dog1': np.float64(0.6120028048602165), 'dog2': np.float64(0.4357548136426572), 'dog3': np.float64(1.0), 'dog4': np.float64(0.0), 'dog5': np.float64(0.44092633829334316)}
bad2.WAV --> pred=dog5, true=background, scores={'dog1': np.float64(0.6763214281037861), 'dog2': np.float64(0.9667150449398189), 'dog3': np.float64(0.7538815490229298), 'dog4': np.float64(0.0), 'dog5': np.float64(1.0)}
dog5_test_01.WAV --> pred=dog5, true=dog5, scores={'dog1': np.float64(0.60029814305296), 'dog2': np.float64(0.9153892869611374), 'dog3': np.float64(0.5138544350319714), 'dog4': np.float64(0.0), 'dog5': np.float64(1.0)}

"""