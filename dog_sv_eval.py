# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/13 10:31
@Author  : Kend
@FileName: dog_sv_eval
@Software: PyCharm
@modifier: GPT 改版，增加每只狗的得分输出
"""

import os
import numpy as np
import librosa
from scipy.stats import skew, kurtosis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

SR = 16000
N_MELS = 80
N_FFT = 1024
HOP = 256
EPS = 1e-8

# ========== 音频加载 ==========
def load_audio(path):
    y, _ = librosa.load(path, sr=SR)
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])  # 预加重
    y, _ = librosa.effects.trim(y, top_db=30)   # 去静音
    return y

# ========== 特征提取 ==========
def extract_logmel_feats(y):
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT,
                                       hop_length=HOP, n_mels=N_MELS, power=2.0)
    logmel = librosa.power_to_db(S)
    m = logmel.T
    d1 = librosa.feature.delta(m.T).T
    d2 = librosa.feature.delta(m.T, order=2).T
    feat = np.concatenate([m, d1, d2], axis=1)  # (T, 3*N_MELS)
    return feat

# ========== 统计池化 ==========
def utt_pooling(feat):
    mu = np.mean(feat, axis=0)
    sd = np.std(feat, axis=0)
    sk = skew(feat, axis=0)
    kt = kurtosis(feat, axis=0)
    emb = np.concatenate([mu, sd, sk, kt], axis=0)
    emb = emb / (np.linalg.norm(emb) + EPS)
    return emb

# ========== 构建注册库 ==========
def build_enroll(enroll_dir):
    enroll = {}
    for pid in os.listdir(enroll_dir):
        pdir = os.path.join(enroll_dir, pid)
        if not os.path.isdir(pdir):
            continue
        embs = []
        for f in os.listdir(pdir):
            if not f.lower().endswith(('.wav', '.flac')): continue
            y = load_audio(os.path.join(pdir, f))
            feat = extract_logmel_feats(y)
            embs.append(utt_pooling(feat))
        if embs:
            enroll[pid] = np.stack(embs, axis=0)
    return enroll

# ========== 构建测试集 ==========
def build_test(test_dir):
    tests = {}
    for f in os.listdir(test_dir):
        if not f.lower().endswith(('.wav', '.flac')): continue
        y = load_audio(os.path.join(test_dir, f))
        feat = extract_logmel_feats(y)
        tests[f] = utt_pooling(feat)
    return tests

# ========== 训练 LDA ==========
def train_lda(enroll):
    X, y = [], []
    for pid, arr in enroll.items():
        for v in arr:
            X.append(v)
            y.append(pid)
    X = np.array(X)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    lda_dim = min(len(np.unique(y)) - 1, Xs.shape[1] - 1)
    lda = LDA(n_components=lda_dim).fit(Xs, y)
    return scaler, lda

# ========== 评分 + 输出所有得分 ==========
def score_eval(enroll, tests):
    scaler, lda = train_lda(enroll)
    enroll_templates = {}
    for pid, arr in enroll.items():
        Xs = scaler.transform(arr)
        Z = lda.transform(Xs)
        tpl = np.mean(Z, axis=0)
        tpl = tpl / (np.linalg.norm(tpl) + EPS)
        enroll_templates[pid] = tpl

    scores, labels = [], []
    for fname, tvec in tests.items():
        zt = lda.transform(scaler.transform(tvec.reshape(1, -1)))[0]
        zt = zt / (np.linalg.norm(zt) + EPS)

        all_scores = {}
        best_pid, best_s = None, -1
        for pid, tpl in enroll_templates.items():
            s = np.dot(zt, tpl)
            all_scores[pid] = s
            if s > best_s:
                best_s, best_pid = s, pid

        # 真值推断：根据文件名前缀匹配注册 ID
        true = None
        for pid in enroll.keys():
            if fname.lower().startswith(pid.lower()):
                true = pid
                break
        if true is None:
            true = "background"


        scores.append(best_s)
        labels.append(1 if true == best_pid else 0)

        # 输出预测结果 + 全部注册狗狗得分
        print(f"{fname} --> pred={best_pid}, score={best_s:.4f}, true={true}")
        print("  All scores:", {k: round(v, 4) for k, v in all_scores.items()})

    return np.array(scores), np.array(labels)

# ========== 计算 EER ==========
def compute_eer(y_true, y_score):
    fpr, tpr, ths = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer, ths[idx], fpr, tpr

# ========== 主函数 ==========
if __name__ == "__main__":
    enroll_dir = "./youtube_wav/brakng_dog_datasets"  # each subfolder = dog id
    test_dir = "youtube_wav/test"

    enroll = build_enroll(enroll_dir)
    tests = build_test(test_dir)
    scores, labels = score_eval(enroll, tests)

    # 计算 ROC / EER / AUC / ACC
    if len(np.unique(labels)) > 1:  # 防止全是背景
        eer, thr, fpr, tpr = compute_eer(labels, scores)
        auc = roc_auc_score(labels, scores)
        acc = accuracy_score(labels, scores > thr)

        print(f"\nFinal EER={eer:.4f}, Thr={thr:.4f}, AUC={auc:.4f}, Acc@thr={acc:.4f}")

        # 画 ROC
        plt.plot(fpr, tpr, label=f'ROC (AUC={auc:.3f})')
        plt.plot([0,1],[0,1],'--',color='gray')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.grid()
        plt.show()
    else:
        print("\n⚠️ Warning: Only one class in labels, ROC/EER 无法计算。")
