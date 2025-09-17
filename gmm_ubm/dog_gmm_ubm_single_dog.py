# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/16 16:30
@Author  : Kend & Assistant
@FileName: dog_gmm_ubm_single_dog.py
@Software: PyCharm
"""

"""
Single Dog Verification using GMM-UBM with intuitive similarity scores
"""

import os
import numpy as np
import librosa
import time
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

# ================== 构建目标狗模型 ==================
def build_target_model(enroll_dir, target_dog_id):
    start_time = time.time()
    
    # 收集目标狗的特征
    target_feats = []
    # 收集其他狗的特征（用于UBM训练）
    other_feats = []
    
    for pid in os.listdir(enroll_dir):
        pdir = os.path.join(enroll_dir, pid)
        if not os.path.isdir(pdir): 
            continue
            
        for f in os.listdir(pdir):
            if not f.lower().endswith(('.wav', '.flac')): 
                continue
            try:
                y = load_audio(os.path.join(pdir, f))
                feat = extract_enhanced_features(y)
                
                if pid == target_dog_id:
                    target_feats.append(feat)
                else:
                    other_feats.append(feat)
            except Exception as e:
                print(f"[WARN] 处理文件 {f} 时出错: {e}")
                continue
    
    if not target_feats:
        raise ValueError(f"没有找到目标狗 {target_dog_id} 的样本")
    
    target_feats = np.vstack(target_feats)
    print(f"[INFO] 目标狗 {target_dog_id} 样本特征: {target_feats.shape[0]} 帧")
    
    # 如果有其他狗的样本，也加入UBM训练
    if other_feats:
        other_feats = np.vstack(other_feats)
        ubm_feats = np.vstack([target_feats, other_feats])
        print(f"[INFO] UBM训练数据: {ubm_feats.shape[0]} 帧")
    else:
        ubm_feats = target_feats
        print(f"[INFO] UBM训练数据: {ubm_feats.shape[0]} 帧 (仅目标狗数据)")
    
    # 标准化特征
    scaler = StandardScaler()
    ubm_feats = scaler.fit_transform(ubm_feats)
    target_feats = scaler.transform(target_feats)
    
    # 训练UBM
    print("[INFO] 训练UBM模型...")
    ubm = GaussianMixture(
        n_components=UBM_COMPONENTS,
        covariance_type='diag',
        max_iter=200,
        reg_covar=1e-3,
        random_state=42
    )
    ubm.fit(ubm_feats)
    
    # 训练目标狗GMM
    print(f"[INFO] 训练目标狗 {target_dog_id} 的GMM模型...")
    dog_gmm = GaussianMixture(
        n_components=DOG_GMM_COMPONENTS,
        covariance_type='diag',
        max_iter=200,
        reg_covar=1e-3,
        random_state=42
    )
    
    # 用 UBM 参数初始化再适应
    if DOG_GMM_COMPONENTS <= UBM_COMPONENTS:
        dog_gmm.means_init = ubm.means_[:DOG_GMM_COMPONENTS]
    dog_gmm.fit(target_feats)
    
    end_time = time.time()
    print(f"[INFO] 模型构建完成，耗时 {end_time - start_time:.2f} 秒")
    
    return dog_gmm, ubm, scaler

# ================== 相似度计算 ==================
def compute_similarity(test_feat, dog_gmm, ubm, scaler):
    # 标准化特征
    scaled_feat = scaler.transform(test_feat)
    
    # 计算对数似然
    dog_likelihood = dog_gmm.score(scaled_feat)
    ubm_likelihood = ubm.score(scaled_feat)
    
    # 计算似然比（更直观的相似度指标）
    likelihood_ratio = dog_likelihood - ubm_likelihood
    
    # 转换为0-1范围的相似度分数
    # 使用sigmoid函数将似然比映射到0-1范围
    similarity = 1 / (1 + np.exp(-likelihood_ratio/10))
    
    return similarity, dog_likelihood, ubm_likelihood

# ================== 解析文件名 ==================
def parse_id_from_filename(fname, target_dog_id=None):
    """
    根据目标狗ID解析文件名，判断是否为目标狗
    """
    fname = fname.lower()
    if target_dog_id:
        if fname.startswith(target_dog_id):
            return "target"
        else:
            return "non-target"
    else:
        # 全狗模式下，返回实际的狗ID或background
        if fname.startswith("dog"):
            return fname.split("_")[0]
        else:
            return "background"

# ================== 单狗测试 ==================
def test_single_dog(test_dir, target_dog_id, dog_gmm, ubm, scaler, threshold=0.5):
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.wav', '.flac'))]
    test_files.sort()
    
    print(f"[INFO] 测试目标狗: {target_dog_id}")
    print(f"[INFO] 测试文件列表: {test_files}")
    print(f"[INFO] 使用阈值: {threshold}")
    print("[INFO] 识别结果:")
    
    scores = []
    labels = []
    
    start_time = time.time()
    
    for f in test_files:
        try:
            y = load_audio(os.path.join(test_dir, f))
            feat = extract_enhanced_features(y)
            
            similarity, dog_likelihood, ubm_likelihood = compute_similarity(feat, dog_gmm, ubm, scaler)
            
            # 根据阈值判断是否为目标狗
            pred_result = "是目标狗" if similarity >= threshold else "非目标狗"
            
            # 获取真实标签
            true_id = parse_id_from_filename(f, target_dog_id)
            true_label = "是目标狗" if true_id == "target" else "非目标狗"
            
            # 保存分数和标签
            scores.append(similarity)
            labels.append(int(true_label == "是目标狗"))
            
            # 打印结果
            is_correct = "✓" if pred_result == true_label else "✗"
            print(f"  {f}: {is_correct} 识别为[{pred_result}] (相似度={similarity:.4f}) 实际为[{true_label}]")
            
        except Exception as e:
            print(f"[WARN] 处理测试文件 {f} 时出错: {e}")
            continue
    
    end_time = time.time()
    print(f"[INFO] 识别完成，耗时 {end_time - start_time:.2f} 秒")
    
    # 计算评估指标
    if len(set(labels)) < 2:
        print("[WARN] 只有一类样本，无法计算ROC-AUC。")
        # 简单准确率计算
        binary_preds = [1 if s >= threshold else 0 for s in scores]
        acc = accuracy_score(labels, binary_preds)
        print(f"[RESULT] 准确率 = {acc:.4f}")
        return

    # 计算 AUC
    auc = roc_auc_score(labels, scores)
    
    # 计算准确率
    binary_preds = [1 if s >= threshold else 0 for s in scores]
    acc = accuracy_score(labels, binary_preds)
    
    # 计算 EER
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_threshold = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[eer_threshold]
    
    print(f"[RESULT] 性能指标: EER={eer:.4f}, AUC={auc:.4f}, 准确率={acc:.4f}")
    print(f"[INFO] 推荐阈值: {thresholds[eer_threshold]:.4f}")

# ================== 构建所有狗的模型 ==================
def build_all_models(enroll_dir):
    # 获取所有狗的ID
    dog_ids = []
    for pid in os.listdir(enroll_dir):
        pdir = os.path.join(enroll_dir, pid)
        if os.path.isdir(pdir):
            dog_ids.append(pid)
    
    dog_ids.sort()
    print(f"[INFO] 构建所有狗的模型: {dog_ids}")
    
    models = {}
    start_time = time.time()
    
    for dog_id in dog_ids:
        try:
            dog_gmm, ubm, scaler = build_target_model(enroll_dir, dog_id)
            models[dog_id] = (dog_gmm, ubm, scaler)
            print(f"[INFO] 狗 {dog_id} 模型构建完成")
        except Exception as e:
            print(f"[WARN] 构建狗 {dog_id} 模型时出错: {e}")
            continue
    
    end_time = time.time()
    print(f"[INFO] 所有模型构建完成，耗时 {end_time - start_time:.2f} 秒")
    
    return models

# ================== 全狗测试 ==================
def test_all_dogs(test_dir, enroll_dir, threshold=0.5):
    # 获取测试文件
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.wav', '.flac'))]
    test_files.sort()
    
    print(f"[INFO] 测试文件列表: {test_files}")
    print(f"[INFO] 使用阈值: {threshold}")
    
    # 先构建所有狗的模型
    models = build_all_models(enroll_dir)
    
    if not models:
        print("[ERROR] 没有成功构建任何狗的模型")
        return
    
    print("[INFO] 开始测试...")
    
    results = []
    start_time = time.time()
    
    for f in test_files:
        try:
            print(f"\n{f}:")
            file_results = {}
            
            # 获取真实标签
            true_label = parse_id_from_filename(f)
            
            # 对每个狗模型计算相似度
            for dog_id, (dog_gmm, ubm, scaler) in models.items():
                try:
                    y = load_audio(os.path.join(test_dir, f))
                    feat = extract_enhanced_features(y)
                    
                    similarity, _, _ = compute_similarity(feat, dog_gmm, ubm, scaler)
                    file_results[dog_id] = similarity
                    
                    print(f"  对狗 {dog_id} 的相似度 = {similarity:.4f}")
                except Exception as e:
                    print(f"[WARN] 计算狗 {dog_id} 相似度时出错: {e}")
                    file_results[dog_id] = 0.0
                    continue
            
            # 选择最高相似度的狗
            if file_results:
                best_dog = max(file_results, key=file_results.get)
                best_score = file_results[best_dog]
                pred_result = best_dog if best_score >= threshold else "background"
                
                is_correct = "✓" if pred_result == true_label else "✗"
                print(f"  --> {is_correct} 预测为: {pred_result} (最高相似度={best_score:.4f}) 实际为: {true_label}")
                results.append((f, pred_result, true_label, best_score, file_results))
                
        except Exception as e:
            print(f"[WARN] 处理测试文件 {f} 时出错: {e}")
            continue
    
    end_time = time.time()
    print(f"\n[INFO] 所有测试完成，耗时 {end_time - start_time:.2f} 秒")
    
    # 输出汇总结果
    print("\n[INFO] 汇总结果:")
    correct_count = 0
    for f, pred, true, score, all_scores in results:
        is_correct = "✓" if pred == true else "✗"
        if pred == true:
            correct_count += 1
        print(f"  {f}: {is_correct} 预测={pred} 实际={true} 最高相似度={score:.4f}")
    
    accuracy = correct_count / len(results) if results else 0
    print(f"\n[RESULT] 总体准确率: {accuracy:.4f} ({correct_count}/{len(results)})")

# ================== 主函数 ==================
def main():
    parser = argparse.ArgumentParser(description="Single Dog Verification using GMM-UBM")
    parser.add_argument('--enroll_dir', type=str, required=True, help='注册样本目录')
    parser.add_argument('--test_dir', type=str, required=True, help='测试样本目录')
    parser.add_argument('--target_dog', type=str, help='目标狗ID（单狗验证模式）')
    parser.add_argument('--all', action='store_true', help='对所有狗进行测试（全狗测试模式）')
    parser.add_argument('--threshold', type=float, default=0.5, help='分类阈值')
    
    args = parser.parse_args()
    
    if args.all:
        # 全狗测试模式
        test_all_dogs(args.test_dir, args.enroll_dir, args.threshold)
    elif args.target_dog:
        # 单狗验证模式
        print(f"[INFO] 为狗 {args.target_dog} 构建模型...")
        dog_gmm, ubm, scaler = build_target_model(args.enroll_dir, args.target_dog)
        test_single_dog(args.test_dir, args.target_dog, dog_gmm, ubm, scaler, args.threshold)
    else:
        print("请指定 --target_dog（单狗验证）或 --all（全狗测试）")

if __name__ == "__main__":
    main()

"""
单狗验证模式：
python gmm_ubm/dog_gmm_ubm_single_dog.py --enroll_dir ./youtube_wav/brakng_dog_datasets --test_dir ./youtube_wav/test --target_dog dog1 --threshold 0.5

全狗测试模式：
python gmm_ubm/dog_gmm_ubm_single_dog.py --enroll_dir ./youtube_wav/brakng_dog_datasets --test_dir ./youtube_wav/test --all --threshold 0.5


bad.WAV:
  对狗 dog1 的相似度 = 0.1853
  对狗 dog2 的相似度 = 0.2755
  对狗 dog3 的相似度 = 0.6411
  对狗 dog4 的相似度 = 0.1510
  对狗 dog5 的相似度 = 0.0134
  --> ✗ 预测为: dog3 (最高相似度=0.6411) 实际为: background

bad2.WAV:
  对狗 dog1 的相似度 = 0.3444
  对狗 dog2 的相似度 = 0.2428
  对狗 dog3 的相似度 = 0.4236
  对狗 dog4 的相似度 = 0.1449
  对狗 dog5 的相似度 = 0.0903
  --> ✓ 预测为: background (最高相似度=0.4236) 实际为: background

dog1_test_01.WAV:
  对狗 dog1 的相似度 = 0.8430
  对狗 dog2 的相似度 = 0.1082
  对狗 dog3 的相似度 = 0.0462
  对狗 dog4 的相似度 = 0.0016
  对狗 dog5 的相似度 = 0.0011
  --> ✓ 预测为: dog1 (最高相似度=0.8430) 实际为: dog1

dog2_test_01.WAV:
  对狗 dog1 的相似度 = 0.1395
  对狗 dog2 的相似度 = 0.3964
  对狗 dog3 的相似度 = 0.0506
  对狗 dog4 的相似度 = 0.0002
  对狗 dog5 的相似度 = 0.0010
  --> ✗ 预测为: background (最高相似度=0.3964) 实际为: dog2

dog3_test_01.WAV:
  对狗 dog1 的相似度 = 0.2071
  对狗 dog2 的相似度 = 0.1459
  对狗 dog3 的相似度 = 0.4898
  对狗 dog4 的相似度 = 0.0549
  对狗 dog5 的相似度 = 0.0320
  --> ✗ 预测为: background (最高相似度=0.4898) 实际为: dog3

dog4_test_01.WAV:
  对狗 dog1 的相似度 = 0.1017
  对狗 dog2 的相似度 = 0.1133
  对狗 dog3 的相似度 = 0.1210
  对狗 dog4 的相似度 = 0.5060
  对狗 dog5 的相似度 = 0.0221
  --> ✓ 预测为: dog4 (最高相似度=0.5060) 实际为: dog4

dog4_test_02.WAV:
  对狗 dog1 的相似度 = 0.0384
  对狗 dog2 的相似度 = 0.0314
  对狗 dog3 的相似度 = 0.0648
  对狗 dog4 的相似度 = 0.5784
  对狗 dog5 的相似度 = 0.0218
  --> ✓ 预测为: dog4 (最高相似度=0.5784) 实际为: dog4

dog5_test_01.WAV:
  对狗 dog1 的相似度 = 0.0437
  对狗 dog2 的相似度 = 0.1255
  对狗 dog3 的相似度 = 0.0530
  对狗 dog4 的相似度 = 0.0041
  对狗 dog5 的相似度 = 0.5996
  --> ✓ 预测为: dog5 (最高相似度=0.5996) 实际为: dog5

dog5_test_02.WAV:
  对狗 dog1 的相似度 = 0.1143
  对狗 dog2 的相似度 = 0.1606
  对狗 dog3 的相似度 = 0.1718
  对狗 dog4 的相似度 = 0.0106
  对狗 dog5 的相似度 = 0.5220
  --> ✓ 预测为: dog5 (最高相似度=0.5220) 实际为: dog5

[INFO] 所有测试完成，耗时 0.27 秒

[INFO] 汇总结果:
  bad.WAV: ✗ 预测=dog3 实际=background 最高相似度=0.6411
  bad2.WAV: ✓ 预测=background 实际=background 最高相似度=0.4236
  dog1_test_01.WAV: ✓ 预测=dog1 实际=dog1 最高相似度=0.8430
  dog2_test_01.WAV: ✗ 预测=background 实际=dog2 最高相似度=0.3964
  dog3_test_01.WAV: ✗ 预测=background 实际=dog3 最高相似度=0.4898
  dog4_test_01.WAV: ✓ 预测=dog4 实际=dog4 最高相似度=0.5060
  dog4_test_02.WAV: ✓ 预测=dog4 实际=dog4 最高相似度=0.5784
  dog5_test_01.WAV: ✓ 预测=dog5 实际=dog5 最高相似度=0.5996
  dog5_test_02.WAV: ✓ 预测=dog5 实际=dog5 最高相似度=0.5220

[RESULT] 总体准确率: 0.6667 (6/9)

"""