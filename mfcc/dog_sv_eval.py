# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/13 10:31
@Author  : Kend
@FileName: dog_sv_eval
@Software: PyCharm
@modifier: 增加每只狗的得分输出
"""


# 2025-09-13 integrated script:
# - features: log-mel(80)+mfcc(40) + deltas + context stacking
# - pooling: mean/std/skew/kurtosis
# - augment: time shift, add noise, volume
# - baseline: scaler + LDA + cosine
# - optional: train small embedding with triplet loss (torch)
# - outputs per-test all scores and saves CSV


"""
（后续调优）

每段是 ~1s，这个脚本用了上下文堆叠（左右 2 帧），帧 hop=256（16 kHz / hop ~16ms），上下文会补时间信息；若需要更长上下文，把 stack_context(... left=4,right=4) 调大。
数据增强 在少样本场景（每只 5–6 条）非常有帮助。--augment 2 会在注册时每条再生成 2 个变体（随机噪声+移位+音量变换）。
若 baseline LDA 性能不好，可考虑 --train_embed 开启 triplet 训练（效果通常比 LDA 好，但需注意过拟合与 batch negative mining 策略）。
把 scores.csv 写出，里面每行含 file, pred, score, true 以及每个注册 score_<id>，方便离线画图 / 统计 max - 2nd_max 差距等。
"""

import os
import argparse
import numpy as np
import librosa
from scipy.stats import skew, kurtosis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import random
import warnings
warnings.filterwarnings("ignore")

# Try import torch for optional embedding training
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
except Exception:
    TORCH_OK = False

# ---------------- Config ----------------
TORCH_OK = False
SR = 16000
N_MELS = 80
N_MFCC = 40
N_FFT = 1024
HOP = 256
EPS = 1e-8

# ---------------- Audio & Augment ----------------
def load_audio(path, sr=SR, trim_db=30):
    y, _ = librosa.load(path, sr=sr)
    if len(y) == 0:
        return None
    # pre-emphasis
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    # trim silence
    y, _ = librosa.effects.trim(y, top_db=trim_db)
    return y

def add_noise(y, snr_db=15):
    rms = np.sqrt(np.mean(y**2))
    snr = 10 ** (snr_db / 10.0)
    noise_rms = rms / np.sqrt(snr)
    noise = np.random.randn(len(y)) * noise_rms
    return y + noise

def time_shift(y, shift_max=0.1):
    # shift_max is fraction of length
    L = len(y)
    shift = int((random.random() * 2 - 1) * shift_max * L)
    if shift > 0:
        return np.concatenate([y[shift:], np.zeros(shift)])
    elif shift < 0:
        return np.concatenate([np.zeros(-shift), y[:shift]])
    else:
        return y

def volume_perturb(y, low=0.7, high=1.3):
    factor = random.uniform(low, high)
    return y * factor

# ---------------- Feature extraction ----------------
def extract_logmel_mfcc(y, sr=SR, n_mels=N_MELS, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP):
    # log-mel
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
    logmel = librosa.power_to_db(S)  # (n_mels, T)
    # mfcc
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    # transpose to (T, dim)
    m = logmel.T
    mm = mfcc.T
    # ensure same time length (should be)
    T = min(m.shape[0], mm.shape[0])
    m = m[:T, :]
    mm = mm[:T, :]
    # deltas
    d1_m = librosa.feature.delta(m.T).T
    d2_m = librosa.feature.delta(m.T, order=2).T
    d1_mf = librosa.feature.delta(mm.T).T
    d2_mf = librosa.feature.delta(mm.T, order=2).T
    # concat per-frame: [logmel, dlogmel, d2logmel, mfcc, dmfcc, d2mfcc]
    feat = np.concatenate([m, d1_m, d2_m, mm, d1_mf, d2_mf], axis=1)
    return feat  # (T, D)

def stack_context(feat, left=2, right=2):
    # stack neighbor frames to give temporal context
    T, D = feat.shape
    frames = []
    for t in range(T):
        parts = []
        for i in range(t-left, t+right+1):
            if i < 0:
                parts.append(np.zeros(D))
            elif i >= T:
                parts.append(np.zeros(D))
            else:
                parts.append(feat[i])
        frames.append(np.concatenate(parts))
    return np.stack(frames, axis=0)  # (T, D*(left+right+1))

# ---------------- Pooling ----------------
def utt_pooling(feat):
    # feat: (T, D)
    if feat is None or feat.shape[0] == 0:
        return None
    mu = np.mean(feat, axis=0)
    sd = np.std(feat, axis=0)
    sk = skew(feat, axis=0)
    kt = kurtosis(feat, axis=0)
    emb = np.concatenate([mu, sd, sk, kt], axis=0)
    emb = emb / (np.linalg.norm(emb) + EPS)
    return emb

# ---------------- Build enroll / test ----------------
def build_enrollments(enroll_dir, augment=0):
    # enroll_dir: subfolders are IDs
    enroll = {}
    ids = [d for d in sorted(os.listdir(enroll_dir)) if os.path.isdir(os.path.join(enroll_dir, d))]
    for pid in ids:
        pdir = os.path.join(enroll_dir, pid)
        embs = []
        for fname in sorted(os.listdir(pdir)):
            if not fname.lower().endswith(('.wav', '.flac', '.mp3')): continue
            p = os.path.join(pdir, fname)
            y = load_audio(p)
            if y is None: continue
            feat = extract_logmel_mfcc(y)
            feat = stack_context(feat, left=2, right=2)
            emb = utt_pooling(feat)
            if emb is not None:
                embs.append(emb)
            # augment copies if requested
            for k in range(augment):
                y2 = time_shift(volume_perturb(add_noise(y, snr_db=random.uniform(10,20)), low=0.8, high=1.2))
                f2 = extract_logmel_mfcc(y2)
                f2 = stack_context(f2, left=2, right=2)
                e2 = utt_pooling(f2)
                if e2 is not None:
                    embs.append(e2)
        if len(embs) > 0:
            enroll[pid] = np.stack(embs, axis=0)
    return enroll

def build_tests(test_dir):
    tests = {}
    for fname in sorted(os.listdir(test_dir)):
        if not fname.lower().endswith(('.wav', '.flac', '.mp3')): continue
        p = os.path.join(test_dir, fname)
        y = load_audio(p)
        if y is None: continue
        feat = extract_logmel_mfcc(y)
        feat = stack_context(feat, left=2, right=2)
        emb = utt_pooling(feat)
        tests[fname] = {'emb': emb, 'path': p}
    return tests

# ---------------- True label inference ----------------
def infer_true_label(fname, enroll_ids):
    # fname like "dog1_test_01.WAV" or "bad.WAV" or "someprefix-dog1.wav"
    base = os.path.splitext(os.path.basename(fname))[0]
    base_low = base.lower()
    # try match any enrolled id as prefix or substring
    for pid in enroll_ids:
        if base_low.startswith(pid.lower()):
            return pid
    for pid in enroll_ids:
        if pid.lower() in base_low:
            return pid
    return "background"

# ---------------- Baseline: scaler + LDA + cosine ----------------
def train_baseline(enroll):
    X, y = [], []
    for pid, arr in enroll.items():
        for v in arr:
            X.append(v); y.append(pid)
    X = np.array(X); y = np.array(y)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    lda_dim = min(len(np.unique(y)) - 1, Xs.shape[1] - 1)
    lda = LDA(n_components=lda_dim).fit(Xs, y)
    # templates: mean of transformed per-id
    templates = {}
    for pid, arr in enroll.items():
        Z = lda.transform(scaler.transform(arr))
        tpl = np.mean(Z, axis=0)
        tpl = tpl / (np.linalg.norm(tpl)+EPS)
        templates[pid] = tpl
    return scaler, lda, templates

# ---------------- Scoring ----------------
def score_and_report(templates, scaler, lda, tests, enroll_ids, out_csv="scores.csv"):
    rows = []
    y_scores_for_eval = []
    y_true_for_eval = []

    for fname, info in tests.items():
        emb = info['emb']
        if emb is None:
            print(f"[WARN] can't extract emb for {fname}")
            continue
        z = lda.transform(scaler.transform(emb.reshape(1,-1)))[0]
        z = z / (np.linalg.norm(z)+EPS)
        all_scores = {}
        best_pid, best_s = None, -1e9
        for pid, tpl in templates.items():
            s = float(np.dot(z, tpl))
            all_scores[pid] = s
            if s > best_s:
                best_s, best_pid = s, pid
        true = infer_true_label(fname, enroll_ids)
        is_genuine = 1 if (true != "background" and best_pid == true) else 0
        rows.append({'file': fname, 'pred': best_pid, 'score': best_s, 'true': true, **{f"score_{pid}": all_scores[pid] for pid in templates}})
        y_scores_for_eval.append(best_s)
        y_true_for_eval.append(is_genuine)
        print(f"{fname} --> pred={best_pid}, score={best_s:.4f}, true={true}")
        print("  All scores:", {k: float(np.round(v,4)) for k,v in all_scores.items()})

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[INFO] scores saved to {out_csv}")

    # ROC/EER if possible
    y_scores = np.array(y_scores_for_eval); y_true = np.array(y_true_for_eval)
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_scores)
        fpr, tpr, ths = roc_curve(y_true, y_scores)
        fnr = 1 - tpr
        idx = np.nanargmin(np.abs(fnr - fpr))
        eer = (fpr[idx] + fnr[idx]) / 2
        thr = ths[idx]
        acc = accuracy_score(y_true, y_scores >= thr)
        print(f"\nAUC={auc:.4f}, EER={eer:.4f}, Thr@EER={thr:.4f}, Acc@thr={acc:.4f}")
        # plot
        plt.figure(); plt.plot(fpr,tpr,label=f"AUC={auc:.3f},EER={eer:.3f}"); plt.plot([0,1],[0,1],'--',color='gray'); plt.legend(); plt.title("ROC"); plt.show()
        # confusion matrix at thr
        preds = (y_scores >= thr).astype(int)
        cm = confusion_matrix(y_true, preds)
        print("Confusion matrix (genuine=1, impostor=0):\n", cm)
    else:
        print("\n[WARN] not enough variety in true labels to compute ROC/EER (need both genuine and impostor).")

    return df

# ---------------- Optional: simple triplet-embedding training (PyTorch) ----------------
class SmallEmbedNet(nn.Module):
    def __init__(self, input_dim, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, emb_dim)
        )
    def forward(self, x):
        return self.net(x)

def train_triplet(enroll, epochs=20, bs=32, lr=1e-3, emb_dim=128, device='cpu'):
    if not TORCH_OK:
        raise RuntimeError("PyTorch not installed")
    # prepare dataset: each sample is pooled emb (already in enroll)
    X = []
    y = []
    for pid, arr in enroll.items():
        for v in arr:
            X.append(v)
            y.append(pid)
    X = np.array(X); y = np.array(y)
    input_dim = X.shape[1]
    # convert to torch
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    # create small net
    model = SmallEmbedNet(input_dim, emb_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    # triplet mining: random batch sampling with per-batch triplet loss
    triplet_loss = nn.TripletMarginLoss(margin=0.3)
    for ep in range(epochs):
        perm = np.random.permutation(len(X))
        losses = []
        for i in range(0, len(X), bs):
            idx = perm[i:i+bs]
            batch_x = X_t[idx]
            batch_y = y[idx]
            # build triplets within batch: for each anchor, pick positive and negative
            anchors = []; positives = []; negatives = []
            for j in range(len(idx)):
                a_x = batch_x[j]
                a_y = batch_y[j]
                # positive: choose another sample with same label
                pos_idx = np.where(batch_y == a_y)[0]
                if len(pos_idx) <= 1:
                    continue
                p_idx = np.random.choice(pos_idx[pos_idx != j])
                n_idx = np.random.choice(np.where(batch_y != a_y)[0])
                anchors.append(a_x); positives.append(batch_x[p_idx]); negatives.append(batch_x[n_idx])
            if len(anchors) < 1:
                continue
            A = torch.stack(anchors)
            P = torch.stack(positives)
            N = torch.stack(negatives)
            opt.zero_grad()
            embA = model(A); embP = model(P); embN = model(N)
            # l2 normalize
            embA = embA / (embA.norm(dim=1, keepdim=True)+1e-8)
            embP = embP / (embP.norm(dim=1, keepdim=True)+1e-8)
            embN = embN / (embN.norm(dim=1, keepdim=True)+1e-8)
            loss = triplet_loss(embA, embP, embN)
            loss.backward(); opt.step()
            losses.append(loss.item())
        if len(losses)>0:
            print(f"[Train] epoch {ep+1}/{epochs}, loss={np.mean(losses):.4f}")
    # Return model in cpu and a function to embed numpy vectors
    model.eval()
    def embed_fn(x_np):
        x = torch.tensor(x_np, dtype=torch.float32).to(device)
        with torch.no_grad():
            z = model(x)
            z = z / (z.norm(dim=1, keepdim=True)+1e-8)
            return z.cpu().numpy()
    return model, embed_fn

# ---------------- Main ----------------
def main(args):
    enroll_dir = args.enroll_dir
    test_dir = args.test_dir
    augment = args.augment  # how many augment copies per enroll utterance
    print("[INFO] Building enrollments...")
    enroll = build_enrollments(enroll_dir, augment=augment)
    enroll_ids = list(enroll.keys())
    print(f"[INFO] enrolled ids: {enroll_ids}")

    print("[INFO] Building tests...")
    tests = build_tests(test_dir)
    print(f"[INFO] test files: {list(tests.keys())}")

    if args.train_embed:
        if not TORCH_OK:
            raise RuntimeError("PyTorch not available; cannot train embedding.")
        print("[INFO] training triplet embedding...")
        # train with pooled features (enroll has multiple pooled emb per id)
        model, embed_fn = train_triplet(enroll, epochs=args.epochs, bs=args.batch_size, lr=args.lr, emb_dim=args.emb_dim, device='cpu')
        # create templates using embed_fn on pooled enroll vectors
        templates = {}
        for pid, arr in enroll.items():
            Z = embed_fn(arr)
            tpl = np.mean(Z, axis=0)
            tpl = tpl / (np.linalg.norm(tpl)+EPS)
            templates[pid] = tpl
        # convert scaler/lda to identity (we will not use lda pipeline)
        scaler = None; lda = None
        # scoring function will use templates and embed_fn directly
        # Build tests embeddings via embedding function
        rows = []
        for fname, info in tests.items():
            emb = info['emb']
            z = embed_fn(emb.reshape(1,-1))[0]
            best_pid, best_s = None, -1e9
            all_scores = {}
            for pid, tpl in templates.items():
                s = float(np.dot(z, tpl))
                all_scores[pid] = s
                if s > best_s:
                    best_s, best_pid = s, pid
            true = infer_true_label(fname, enroll_ids)
            print(f"{fname} -> pred={best_pid}, score={best_s:.4f}, true={true}")
            print("  All scores:", {k:round(v,4) for k,v in all_scores.items()})
        print("[INFO] triplet-trained evaluation done.")
        return

    # baseline pipeline
    scaler, lda, templates = train_baseline(enroll)
    df = score_and_report(templates, scaler, lda, tests, enroll_ids, out_csv=args.out_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enroll_dir", type=str, default="./person", help="enroll dir (subfolders per id)")
    parser.add_argument("--test_dir", type=str, default="./test_person_flac", help="test files dir")
    parser.add_argument("--augment", type=int, default=0, help="number of augmentation per enroll utt")
    parser.add_argument("--train_embed", action="store_true", help="train triplet embedding (requires torch)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--out_csv", type=str, default="scores.csv")
    args = parser.parse_args()
    main(args)


"""
不训练 embedding，只跑基线（LDA）并保存 scores.csv, --augment 2 表示每条 enrollment 生成 2 个增强样本，推荐在样本少时使用
python mfcc/dog_sv_eval.py --enroll_dir ./youtube_wav/brakng_dog_datasets --test_dir ./youtube_wav/test
若开启 triplet 训练（需要 PyTorch）这里不考虑
python dog_sv_eval.py --enroll_dir ./person --test_dir ./test_person_flac --train_embed --epochs 30 --batch_size 64
"""

"""
[INFO] Building enrollments...
[INFO] enrolled ids: ['dog1', 'dog2', 'dog3', 'dog4', 'dog5']
[INFO] Building tests...
[INFO] test files: ['bad.WAV', 'bad2.WAV', 'dog1_test_01.WAV', 'dog2_test_01.WAV', 'dog3_test_01.WAV', 'dog4_test_01.WAV', 'dog4_test_02.WAV', 'dog5_test_01.WAV', 'dog5_test_02.WAV']
bad.WAV --> pred=dog4, score=0.5766, true=background
  All scores: {'dog1': 0.1806, 'dog2': 0.0375, 'dog3': -0.6631, 'dog4': 0.5766, 'dog5': -0.6299}
bad2.WAV --> pred=dog4, score=0.4281, true=background
  All scores: {'dog1': 0.259, 'dog2': 0.2031, 'dog3': -0.6948, 'dog4': 0.4281, 'dog5': -0.7423}
dog1_test_01.WAV --> pred=dog5, score=0.6278, true=dog1
  All scores: {'dog1': -0.2922, 'dog2': 0.1155, 'dog3': -0.1115, 'dog4': -0.3657, 'dog5': 0.6278}
dog2_test_01.WAV --> pred=dog2, score=0.9837, true=dog2
  All scores: {'dog1': 0.3475, 'dog2': 0.9837, 'dog3': -0.3927, 'dog4': -0.7835, 'dog5': -0.5745}
dog3_test_01.WAV --> pred=dog3, score=0.6818, true=dog3
  All scores: {'dog1': -0.4573, 'dog2': -0.5557, 'dog3': 0.6818, 'dog4': 0.5948, 'dog5': 0.2025}
dog4_test_01.WAV --> pred=dog4, score=0.9013, true=dog4
  All scores: {'dog1': -0.2016, 'dog2': -0.3946, 'dog3': -0.2633, 'dog4': 0.9013, 'dog5': -0.2223}
dog4_test_02.WAV --> pred=dog4, score=0.8220, true=dog4
  All scores: {'dog1': -0.5413, 'dog2': -0.2465, 'dog3': -0.0943, 'dog4': 0.822, 'dog5': -0.0585}
dog5_test_01.WAV --> pred=dog5, score=0.9513, true=dog5
  All scores: {'dog1': -0.758, 'dog2': -0.3745, 'dog3': 0.6472, 'dog4': 0.0324, 'dog5': 0.9513}
dog5_test_02.WAV --> pred=dog5, score=0.9487, true=dog5
  All scores: {'dog1': -0.858, 'dog2': -0.5116, 'dog3': 0.5807, 'dog4': 0.2994, 'dog5': 0.9487}
[INFO] scores saved to scores.csv

特征提取：MFCC（+ 一阶/二阶差分，DTM）
相似度：余弦相似度
评估：AUC、EER

结论：单纯增加 log-Mel、PLP 之类的特征并没有让效果更好，反而波动比较大。

（每只狗 5-6 条，1s 音频）太少，额外特征容易引入噪声；
余弦相似度对噪声很敏感，且小样本时决策边界不稳定；
EER/AUC 虽然看起来很好，但测试集少时容易“过拟合”。
"""