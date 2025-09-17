# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/16 18:00
@Author  : Kend & Assistant
@FileName: dog_ivector_sv_eval.py
@Software: PyCharm
"""

"""
Dog Speaker Verification using i-vector
专门为单只狗的声纹验证设计，适用于端侧部署

改进点：
1. 特征 = MFCC + Δ + ΔΔ + log-Mel + 谱质心等增强特征
2. CMVN 标准化
3. i-vector 降维
4. 余弦相似度计算
5. 阈值机制，支持"目标狗"和"非目标"分类
6. 专为单狗验证设计
"""

import os
import argparse
import numpy as np
import librosa
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from ivector_extractor import IVectorExtractor

# ========== 特征提取 ==========
def extract_features(wav_path, sr=16000, n_mfcc=20, n_mels=40):
    y, sr = librosa.load(wav_path, sr=sr)
    
    # 如果音频太短，进行重采样延长
    if len(y) < sr * 0.1:  # 如果小于0.1秒
        # 重复音频直到达到最小长度
        repeat_times = int(np.ceil((sr * 0.1) / len(y)))
        y = np.tile(y, repeat_times)
        y = y[:int(sr * 0.1)]  # 裁剪到精确长度

    # MFCC - 保持时间维度
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=256, n_mels=n_mels)
    
    # 确保有足够的帧来计算delta特征
    if mfcc.shape[1] < 9:  # librosa默认窗口宽度为9
        # 通过重复帧来增加帧数
        repeat_times = int(np.ceil(9 / mfcc.shape[1]))
        mfcc = np.tile(mfcc, (1, repeat_times))
        mfcc = mfcc[:, :9]  # 裁剪到精确帧数
    
    delta1 = librosa.feature.delta(mfcc, order=1, width=min(5, mfcc.shape[1]//2*2+1))
    delta2 = librosa.feature.delta(mfcc, order=2, width=min(5, mfcc.shape[1]//2*2+1))

    # log-Mel - 保持时间维度
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=512, hop_length=256)
    logmel = librosa.power_to_db(mel)
    
    # 对log-Mel特征也进行帧数检查
    if logmel.shape[1] < 9:
        repeat_times = int(np.ceil(9 / logmel.shape[1]))
        logmel = np.tile(logmel, (1, repeat_times))
        logmel = logmel[:, :9]
    
    # 谱质心 - 保持时间维度
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=512, hop_length=256)
    if spectral_centroids.shape[1] < 9:
        repeat_times = int(np.ceil(9 / spectral_centroids.shape[1]))
        spectral_centroids = np.tile(spectral_centroids, (1, repeat_times))
        spectral_centroids = spectral_centroids[:, :9]
    
    # 零交叉率 - 保持时间维度
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=512, hop_length=256)
    if zcr.shape[1] < 9:
        repeat_times = int(np.ceil(9 / zcr.shape[1]))
        zcr = np.tile(zcr, (1, repeat_times))
        zcr = zcr[:, :9]
    
    # 拼接所有特征，保持时间维度
    feat = np.vstack([
        mfcc, delta1, delta2, 
        logmel,
        spectral_centroids,
        zcr
    ])

    # 转置为 [帧数, 特征维度] 格式
    feat = feat.T

    return feat

# ========== 构建目标狗的模型 ==========
def build_target_model(file_list, target_dog_id, tv_dim=32):
    start_time = time.time()
    
    # 收集目标狗的特征
    target_feats = []
    # 收集非目标狗的特征（用于UBM训练）
    non_target_feats = []
    
    for f in file_list:
        fname = os.path.basename(f).lower()
        try:
            x = extract_features(f)
            # 分类目标和非目标样本
            if fname.startswith(target_dog_id):
                target_feats.append(x)
            elif fname.startswith("dog"):  # 其他狗的样本作为负样本
                non_target_feats.append(x)
        except Exception as e:
            print(f"[WARN] 处理文件 {f} 时出错: {e}")
            continue
    
    if len(target_feats) == 0:
        raise ValueError(f"No samples found for target dog: {target_dog_id}")
    
    print(f"[INFO] 目标狗 {target_dog_id} 样本数: {len(target_feats)}")
    print(f"[INFO] 其他狗样本数: {len(non_target_feats)}")
    
    # 合并特征用于UBM训练
    ubm_feats_list = target_feats + non_target_feats
    
    if not ubm_feats_list:
        raise ValueError("No features collected for UBM training")
    
    # 训练i-vector提取器
    feature_dim = ubm_feats_list[0].shape[1] if ubm_feats_list else 102
    # 确保i-vector维度不超过样本数和特征维度
    max_tv_dim = min(tv_dim, len(ubm_feats_list), feature_dim // 2)
    adaptive_tv_dim = max(1, max_tv_dim)  # 至少为1
    
    print(f"[INFO] 特征维度: {feature_dim}, i-vector维度: {adaptive_tv_dim}")
    
    ivec_extractor = IVectorExtractor(n_components=8, tv_dim=adaptive_tv_dim)
    ivec_extractor.fit(ubm_feats_list)
    
    # 提取目标狗的i-vectors
    target_ivectors = []
    for feat in target_feats:
        ivec = ivec_extractor.transform(feat)
        target_ivectors.append(ivec)
    target_ivectors = np.array(target_ivectors)
    
    end_time = time.time()
    print(f"[INFO] 模型构建完成，耗时 {end_time - start_time:.2f} 秒")
    
    return ivec_extractor, target_ivectors

# ========== 相似度计算 ==========
def cosine_score(x, y):
    # 确保输入是1维向量
    if x.ndim > 1:
        x = x.flatten()
    if y.ndim > 1:
        y = y.flatten()
        
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    
    # 避免除零错误
    if norm_x == 0 or norm_y == 0:
        return 0.0
        
    result = dot_product / (norm_x * norm_y)
    return result

# ========== 解析文件名 ==========
def parse_id_from_filename(fname, target_dog_id):
    """
    根据目标狗ID解析文件名，判断是否为目标狗
    - dog1_xxx.WAV --> 如果target_dog_id是dog1则为target，否则为non-target
    - bad.WAV / noise.WAV --> non-target
    """
    fname = fname.lower()
    if fname.startswith(target_dog_id):
        return "target"
    else:
        return "non-target"

# ========== 评估 ==========
def evaluate(ivec_extractor, target_ivectors, test_files, target_dog_id, threshold=0.0):
    # 打印测试信息
    test_filenames = [os.path.basename(f) for f in test_files]
    print("[INFO] 构建测试集...")
    print(f"[INFO] 测试文件列表: {test_filenames}")
    
    scores = []
    labels = []
    
    start_time = time.time()
    
    print(f"[INFO] 使用阈值: {threshold}")
    print(f"[INFO] 识别结果 (目标狗为{target_dog_id}):")
    
    for i, f in enumerate(test_files):
        try:
            # 提取特征
            feat = extract_features(f)
            
            # 提取i-vector
            test_ivec = ivec_extractor.transform(feat)
            
            # 计算与目标狗i-vectors的平均相似度
            similarities = []
            for target_ivec in target_ivectors:
                sim = cosine_score(test_ivec, target_ivec)
                similarities.append(sim)
            
            score = np.mean(similarities)
            
            # 根据阈值判断是否为目标狗
            pred_result = "是目标狗" if score >= threshold else "非目标狗"
            
            # 获取真实标签
            true_id = parse_id_from_filename(os.path.basename(f), target_dog_id)
            true_label = "是目标狗" if true_id == "target" else "非目标狗"
            
            # 保存分数和标签
            scores.append(score)
            labels.append(int(true_label == "是目标狗"))
            
            # 打印当前测试文件的结果
            filename = test_filenames[i]
            is_correct = "✓" if pred_result == true_label else "✗"
            print(f"  {filename}: {is_correct} 识别为[{pred_result}] (相似度={score:.4f}) 实际为[{true_label}]")
            
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


# ========== 主程序 ==========
def collect_wavs(root_dir):
    wavs = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                wavs.append(os.path.join(root, f))
    return wavs

def main():
    parser = argparse.ArgumentParser(description="Dog Speaker Verification using i-vector - Single dog verification")
    parser.add_argument("--enroll_dir", type=str, required=True, 
                        help="Directory containing enrollment samples")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Directory containing test samples")
    parser.add_argument("--target_dog", type=str, required=True,
                        help="Target dog ID (e.g., dog1, dog2)")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Threshold for verification. Scores >= threshold are classified as target dog")
    parser.add_argument("--tv_dim", type=int, default=16,
                        help="i-vector dimension")
    args = parser.parse_args()

    # 注册目标狗模型
    enroll_files = collect_wavs(args.enroll_dir)
    if len(enroll_files) == 0:
        raise RuntimeError(f"[ERROR] No .wav files found in enroll_dir: {args.enroll_dir}")
    
    print(f"[INFO] 为狗 {args.target_dog} 构建声纹模型，使用 {len(enroll_files)} 个样本...")
    ivec_extractor, target_ivectors = build_target_model(enroll_files, args.target_dog, args.tv_dim)

    # 测试
    test_files = collect_wavs(args.test_dir)
    if len(test_files) == 0:
        raise RuntimeError(f"[ERROR] No .wav files found in test_dir: {args.test_dir}")
    
    print(f"[INFO] 准备测试，共 {len(test_files)} 个文件...")
    
    # 评估
    evaluate(ivec_extractor, target_ivectors, test_files, args.target_dog, args.threshold)


if __name__ == "__main__":
    main()

"""
python ivector/dog_ivector_sv_eval.py --enroll_dir ./youtube_wav/brakng_dog_datasets --test_dir ./youtube_wav/test --target_dog dog1 --threshold 0.5



python ivector/dog_ivector_sv_eval.py --enroll_dir ./youtube_wav/brakng_dog_datasets --test_dir ./youtube_wav/test --target_dog dog1 --threshold 0.5 --tv_dim 8

"""