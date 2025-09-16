# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/13 11:20
@Author  : Kend
@FileName: dog_gmm_ubm
@Software: PyCharm
@modifier:
"""


"""
Dog Bark Recognition using GMM-UBM
"""
import os
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import csv

# ================== 配置 ==================
SR = 16000
N_MFCC = 20
FRAME_LEN = 0.025
FRAME_STEP = 0.01
DOG_GMM_COMPONENTS = 2  # 或者 3
UBM_COMPONENTS = 8      # 或者 4
EPS = 1e-8
BACKGROUND_THRESH = -90  #


# ================== 音频加载 ==================
def load_audio(path):
    y, _ = librosa.load(path, sr=SR)
    y, _ = librosa.effects.trim(y, top_db=30)
    return y

# ================== 特征提取 ==================
def extract_mfcc(y):
    mfcc_feat = librosa.feature.mfcc(
        y=y,
        sr=SR,
        n_mfcc=N_MFCC,
        n_fft=int(SR*FRAME_LEN),
        hop_length=int(SR*FRAME_STEP),
        n_mels=40,   # 增加 mel 滤波器数量
        htk=True
    ).T
    return mfcc_feat



# ================== 构建训练集 ==================
def build_enroll(enroll_dir):
    enroll_data = {}
    all_feats = []
    for pid in os.listdir(enroll_dir):
        pdir = os.path.join(enroll_dir, pid)
        if not os.path.isdir(pdir): continue
        feats = []
        for f in os.listdir(pdir):
            if not f.lower().endswith(('.wav', '.flac')): continue
            y = load_audio(os.path.join(pdir, f))
            feat = extract_mfcc(y)
            feats.append(feat)
            all_feats.append(feat)
        if feats:
            enroll_data[pid] = np.vstack(feats)
    return enroll_data, np.vstack(all_feats)



# ================== 训练 GMM ==================
def train_gmms(enroll_data, ubm_feats):
    print(f"[INFO] Training UBM with {ubm_feats.shape[0]} frames...")
    ubm = GaussianMixture(
        n_components=UBM_COMPONENTS,
        covariance_type='diag',
        max_iter=200,
        reg_covar=1e-3,
        random_state=42
    )

    ubm.fit(ubm_feats)

    dog_gmms = {}
    for pid, feats in enroll_data.items():
        gmm = GaussianMixture(
            n_components=DOG_GMM_COMPONENTS,
            covariance_type='diag',
            max_iter=200,
            reg_covar=1e-3,
            random_state=42
        )

        # 用 UBM 参数初始化再适应
        gmm.means_init = ubm.means_[:DOG_GMM_COMPONENTS]
        gmm.fit(feats)
        dog_gmms[pid] = gmm
        print(f"[INFO] built model for {pid} with {feats.shape[0]} frames")
    return ubm, dog_gmms



# ================== 测试 ==================
def test_gmms(test_dir, dog_gmms):
    scores_all = []
    for f in os.listdir(test_dir):
        if not f.lower().endswith(('.wav', '.flac')): continue
        y = load_audio(os.path.join(test_dir, f))
        feat = extract_mfcc(y)
        scores = {}
        for pid, gmm in dog_gmms.items():
            score = gmm.score(feat)  # log-likelihood per frame averaged
            scores[pid] = score
        # 选择最大得分狗
        best_pid = max(scores, key=scores.get)
        best_score = scores[best_pid]
        # 背景判定
        if best_score < BACKGROUND_THRESH:
            best_pid = 'background'
        print(f"{f} --> pred={best_pid}, scores={ {k: round(v,4) for k,v in scores.items()} }")
        scores_all.append((f, best_pid, scores))
    return scores_all


# ================== 评估指标 ==================
def evaluate(scores_all, true_func):
    y_true, y_score = [], []
    for fname, pred, scores in scores_all:
        true_label = true_func(fname)
        max_score = max(scores.values())
        y_true.append(1 if pred == true_label else 0)
        y_score.append(max_score)
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
    acc = accuracy_score(y_true, y_score > thr)
    print(f"[RESULT] EER = {eer:.4f}, AUC = {auc:.4f}, Accuracy = {acc:.4f}")
    return eer, auc, acc

# ================== CSV 保存 ==================
def save_scores(scores_all, path='scores.csv'):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['filename', 'pred'] + list(next(iter(scores_all))[2].keys())
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
    import argparse
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
python  gmm_ubm/dog_gmm_ubm.py --enroll_dir ./youtube_wav/brakng_dog_datasets --test_dir ./youtube_wav/test


运行结果：
dog1_test_01.WAV --> pred=dog1, scores={'dog4': np.float64(-92.0916), 'dog5': np.float64(-82.6699), 'dog1': np.float64(-69.2992), 'dog2': np.float64(-77.5813), 'dog3': np.float64(-80.9089)}
dog4_test_02.WAV --> pred=dog4, scores={'dog4': np.float64(-65.3218), 'dog5': np.float64(-74.0631), 'dog1': np.float64(-77.7713), 'dog2': np.float64(-78.0285), 'dog3': np.float64(-76.1922)}
dog5_test_02.WAV --> pred=dog5, scores={'dog4': np.float64(-87.7004), 'dog5': np.float64(-68.9158), 'dog1': np.float64(-80.6997), 'dog2': np.float64(-76.9892), 'dog3': np.float64(-77.6943)}
dog2_test_01.WAV --> pred=dog2, scores={'dog4': np.float64(-102.7428), 'dog5': np.float64(-83.6731), 'dog1': np.float64(-87.1025), 'dog2': np.float64(-74.9093), 'dog3': np.float64(-89.0318)}
bad.WAV --> pred=dog5, scores={'dog4': np.float64(-82.4953), 'dog5': np.float64(-68.8117), 'dog1': np.float64(-80.7699), 'dog2': np.float64(-76.4289), 'dog3': np.float64(-74.6894)}
dog4_test_01.WAV --> pred=dog4, scores={'dog4': np.float64(-66.9475), 'dog5': np.float64(-76.8648), 'dog1': np.float64(-79.4769), 'dog2': np.float64(-81.0707), 'dog3': np.float64(-80.1913)}
dog3_test_01.WAV --> pred=dog3, scores={'dog4': np.float64(-79.3957), 'dog5': np.float64(-75.7568), 'dog1': np.float64(-74.2947), 'dog2': np.float64(-76.8051), 'dog3': np.float64(-66.446)}
bad2.WAV --> pred=dog5, scores={'dog4': np.float64(-92.6106), 'dog5': np.float64(-68.3545), 'dog1': np.float64(-78.476), 'dog2': np.float64(-75.7369), 'dog3': np.float64(-78.679)}
dog5_test_01.WAV --> pred=dog5, scores={'dog4': np.float64(-96.2504), 'dog5': np.float64(-76.5906), 'dog1': np.float64(-96.1389), 'dog2': np.float64(-85.0718), 'dog3': np.float64(-92.6942)}
[INFO] scores saved to scores.csv
[RESULT] EER = 0.7857, AUC = 0.4286, Accuracy = 0.4444


"""