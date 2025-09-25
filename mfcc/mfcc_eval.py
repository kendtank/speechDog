# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/20 下午2:08
@Author  : Kend
@FileName: mfcc_eval.py
@Software: PyCharm
"""

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# -----------------------------
# 参数配置
# -----------------------------
DATA_DIR = r"D:\kend\myPython\speechDog-master\tiny\bark_segments"
NOISE_DIR = r"D:\work\datasets\tinyML\no_bark"
REPORT_DIR = r"D:\kend\myPython\speechDog-master\tiny\reports"
SR = 16000
N_MFCC = 13
THRESHOLD = 50  # DTW 距离阈值，越小越相似

os.makedirs(REPORT_DIR, exist_ok=True)

# -----------------------------
# 功能函数
# -----------------------------
def extract_mfcc(file_path):
    y, _ = librosa.load(file_path, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    w = min(3, mfcc.shape[1])  # 防止短序列报错
    delta = librosa.feature.delta(mfcc, width=w)
    delta2 = librosa.feature.delta(mfcc, order=2, width=w)
    feat = np.vstack([mfcc, delta, delta2])
    return feat.T


def dtw_distance(seq1, seq2):
    distance, _ = fastdtw(seq1, seq2, dist=euclidean)
    return distance

def build_templates(dog_files, test_index=0):
    template_files = dog_files[:test_index] + dog_files[test_index + 1:]
    templates = [extract_mfcc(f) for f in template_files]
    return templates

def tes_dog(dog_name, dog_files, other_dogs_files, noise_files):
    test_index = 0  # 可改为随机选择
    test_file = dog_files[test_index]
    template_features = build_templates(dog_files, test_index)

    # 测试文件包含该狗测试片段、其他狗及噪声
    test_files = [test_file] + other_dogs_files + noise_files
    results = []

    for f in test_files:
        feat_test = extract_mfcc(f)
        # 计算与模板库每个片段的 DTW 距离，取最小
        distances = [dtw_distance(feat_test, tpl) for tpl in template_features]
        min_distance = np.min(distances)
        label = "MATCH" if min_distance <= THRESHOLD else "NO_MATCH"
        results.append({
            "test_file": os.path.basename(f),
            "min_dtw_distance": min_distance,
            "label": label
        })
    df = pd.DataFrame(results)
    df["dog_id"] = dog_name
    return df, test_file

# -----------------------------
# 数据整理
# -----------------------------
dog_groups = {}
for f in os.listdir(DATA_DIR):
    if f.endswith(".wav"):
        dog_id = f.split("_")[0]
        dog_groups.setdefault(dog_id, []).append(os.path.join(DATA_DIR, f))

noise_files = [os.path.join(NOISE_DIR, f) for f in os.listdir(NOISE_DIR) if f.endswith(".wav")]

# -----------------------------
# 循环测试所有狗
# -----------------------------
all_results = []
summary_data = []
dog_ids = list(dog_groups.keys())

for dog_id in dog_ids:
    dog_files = dog_groups[dog_id]
    other_dogs_files = []
    for other_id in dog_ids:
        if other_id != dog_id:
            other_dogs_files.extend(dog_groups[other_id])

    df, test_file = tes_dog(dog_id, dog_files, other_dogs_files, noise_files)
    all_results.append(df)

    own_test = df[df["test_file"] == os.path.basename(test_file)]
    own_acc = (own_test["label"] == "MATCH").mean()

    other_tests = df[df["test_file"] != os.path.basename(test_file)]
    false_alarm = (other_tests["label"] == "MATCH").mean()

    summary_data.append({
        "dog_id": dog_id,
        "test_file": os.path.basename(test_file),
        "own_acc": own_acc,
        "false_alarm": false_alarm
    })

summary_df = pd.DataFrame(summary_data)
all_results_df = pd.concat(all_results, ignore_index=True)

# -----------------------------
# 可视化和保存报告
# -----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_md = os.path.join(REPORT_DIR, f"dog_bark_test_report_{timestamp}.md")

with open(report_md, "w", encoding="utf-8") as f:
    f.write(f"# 狗吠特征匹配测试报告\n\n")
    f.write(f"生成时间：{datetime.now()}\n\n")
    f.write("## 1. 总体统计\n\n")
    f.write(summary_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## 2. 各狗 DTW 距离柱状图\n\n")
    for dog_id in dog_ids:
        df = all_results_df[all_results_df["dog_id"] == dog_id]
        plt.figure(figsize=(10, 4))
        plt.bar(df["test_file"], df["min_dtw_distance"])
        plt.axhline(y=THRESHOLD, color='r', linestyle='--', label="Threshold")
        plt.xticks(rotation=90)
        plt.title(f"{dog_id} DTW 距离")
        plt.ylabel("DTW Distance")
        plt.legend()
        img_path = os.path.join(REPORT_DIR, f"{dog_id}_dtw_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()
        f.write(f"### {dog_id}\n\n")
        f.write(f"![{dog_id}]({os.path.basename(img_path)})\n\n")

    f.write("## 3. 所有测试片段结果\n\n")
    f.write(all_results_df.to_markdown(index=False))

print(f"测试完成，报告生成：{report_md}")
