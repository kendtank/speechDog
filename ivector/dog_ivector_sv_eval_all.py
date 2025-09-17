# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/16 18:25
@Author  : Kend & Assistant
@FileName: dog_ivector_sv_eval_all.py
@Software: PyCharm
"""

"""
Dog Speaker Verification using i-vector - All dogs evaluation
对所有已注册的狗进行声纹验证测试

改进点：
1. 特征 = MFCC + Δ + ΔΔ + log-Mel + 谱质心等增强特征
2. CMVN 标准化
3. i-vector 降维
4. 余弦相似度计算
5. 对所有狗进行测试
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

    # 统一帧参数
    n_fft = 512
    hop_length = 256
    
    # MFCC - 保持时间维度
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    
    # Delta和Delta-Delta特征
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # log-Mel
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    logmel = librosa.power_to_db(mel)
    
    # 谱质心
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    
    # 零交叉率
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)
    
    # 确保所有特征具有相同的帧数
    min_frames = min(
        mfcc.shape[1], 
        delta1.shape[1], 
        delta2.shape[1],
        logmel.shape[1],
        spectral_centroids.shape[1],
        zcr.shape[1]
    )
    
    # 截取所有特征到相同的帧数
    mfcc = mfcc[:, :min_frames]
    delta1 = delta1[:, :min_frames]
    delta2 = delta2[:, :min_frames]
    logmel = logmel[:, :min_frames]
    spectral_centroids = spectral_centroids[:, :min_frames]
    zcr = zcr[:, :min_frames]
    
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

# ========== 构建单个狗的模型 ==========
def build_single_dog_model(file_list, target_dog_id, tv_dim=16):
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
    
    # 合并特征用于UBM训练
    ubm_feats_list = target_feats + non_target_feats
    
    if not ubm_feats_list:
        raise ValueError("No features collected for UBM training")
    
    # 训练i-vector提取器
    feature_dim = ubm_feats_list[0].shape[1] if ubm_feats_list else 102
    # 确保i-vector维度不超过样本数和特征维度
    max_tv_dim = min(tv_dim, len(ubm_feats_list), feature_dim // 2)
    adaptive_tv_dim = max(1, max_tv_dim)  # 至少为1
    
    ivec_extractor = IVectorExtractor(n_components=8, tv_dim=adaptive_tv_dim)
    ivec_extractor.fit(ubm_feats_list)
    
    # 提取目标狗的i-vectors
    target_ivectors = []
    for feat in target_feats:
        ivec = ivec_extractor.transform(feat)
        target_ivectors.append(ivec)
    target_ivectors = np.array(target_ivectors)
    
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

# ========== 获取所有狗的ID ==========
def get_dog_ids(enroll_dir):
    dog_ids = []
    for item in os.listdir(enroll_dir):
        item_path = os.path.join(enroll_dir, item)
        if os.path.isdir(item_path):
            dog_ids.append(item)
    return sorted(dog_ids)

# ========== 解析文件名 ==========
def parse_id_from_filename(fname):
    """
    解析文件名，返回实际的狗ID或background
    """
    fname = fname.lower()
    if fname.startswith("dog"):
        return fname.split("_")[0]
    else:
        return "background"

# ========== 全狗测试 ==========
def test_all_dogs(enroll_dir, test_dir, tv_dim=16, threshold=0.5):
    # 获取所有狗的ID
    dog_ids = get_dog_ids(enroll_dir)
    print(f"[INFO] 所有已注册的狗: {dog_ids}")
    
    # 获取测试文件
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(".wav")]
    test_files.sort()
    print(f"[INFO] 测试文件列表: {test_files}")
    
    # 为每只狗构建模型
    dog_models = {}
    print("[INFO] 构建所有狗的模型...")
    
    # 收集所有注册文件
    enroll_files = []
    for root, _, files in os.walk(enroll_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                enroll_files.append(os.path.join(root, f))
    
    if len(enroll_files) == 0:
        raise RuntimeError(f"[ERROR] No .wav files found in enroll_dir: {enroll_dir}")
    
    start_time = time.time()
    for dog_id in dog_ids:
        try:
            print(f"[INFO] 构建狗 {dog_id} 的模型...")
            ivec_extractor, target_ivectors = build_single_dog_model(enroll_files, dog_id, tv_dim)
            dog_models[dog_id] = (ivec_extractor, target_ivectors)
            print(f"[INFO] 狗 {dog_id} 模型构建完成")
        except Exception as e:
            print(f"[ERROR] 构建狗 {dog_id} 模型时出错: {e}")
            continue
    
    if not dog_models:
        raise RuntimeError("没有成功构建任何狗的模型")
    
    model_time = time.time()
    print(f"[INFO] 所有模型构建完成，耗时 {model_time - start_time:.2f} 秒")
    
    # 测试每个文件
    print("[INFO] 开始测试...")
    results = []
    correct = 0
    total = 0
    
    for f in test_files:
        print(f"\n{f}:")
        file_scores = {}
        
        # 获取真实标签
        true_label = parse_id_from_filename(f)
        
        try:
            # 对每只狗计算相似度
            for dog_id, (ivec_extractor, target_ivectors) in dog_models.items():
                try:
                    fpath = os.path.join(test_dir, f)
                    feat = extract_features(fpath)
                    test_ivec = ivec_extractor.transform(feat)
                    
                    # 计算相似度
                    similarities = []
                    for target_ivec in target_ivectors:
                        sim = cosine_score(test_ivec, target_ivec)
                        similarities.append(sim)
                    
                    avg_similarity = np.mean(similarities)
                    file_scores[dog_id] = avg_similarity
                    print(f"  对狗 {dog_id} 的相似度 = {avg_similarity:.4f}")
                    
                except Exception as e:
                    print(f"[WARN] 计算狗 {dog_id} 相似度时出错: {e}")
                    file_scores[dog_id] = -1.0
                    continue
            
            # 选择最高相似度的狗
            if file_scores:
                best_dog = max(file_scores, key=file_scores.get)
                best_score = file_scores[best_dog]
                pred_label = best_dog if best_score >= threshold else "background"
                
                is_correct = (pred_label == true_label)
                if is_correct:
                    correct += 1
                total += 1
                
                status = "✓" if is_correct else "✗"
                print(f"  {status} 预测为: {pred_label} (最高相似度={best_score:.4f}) 实际为: {true_label}")
                results.append((f, pred_label, true_label, best_score, file_scores))
                
        except Exception as e:
            print(f"[ERROR] 处理文件 {f} 时出错: {e}")
            continue
    
    # 输出汇总结果
    test_time = time.time()
    print(f"\n[INFO] 所有测试完成，耗时 {test_time - model_time:.2f} 秒")
    print(f"\n[INFO] 汇总结果:")
    for fname, pred, true_label, score, all_scores in results:
        status = "✓" if pred == true_label else "✗"
        print(f"  {fname}: {status} 预测={pred} 实际={true_label} 最高相似度={score:.4f}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\n[RESULT] 总体准确率: {accuracy:.4f} ({correct}/{total})")

# ========== 主程序 ==========
def main():
    parser = argparse.ArgumentParser(description="Dog Speaker Verification using i-vector - All dogs evaluation")
    parser.add_argument("--enroll_dir", type=str, required=True, 
                        help="Directory containing enrollment samples")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Directory containing test samples")
    parser.add_argument("--tv_dim", type=int, default=16,
                        help="i-vector dimension")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for verification")
    args = parser.parse_args()

    # 全狗测试
    test_all_dogs(args.enroll_dir, args.test_dir, args.tv_dim, args.threshold)


if __name__ == "__main__":
    main()

"""
python ivector/dog_ivector_sv_eval_all.py --enroll_dir ./youtube_wav/brakng_dog_datasets --test_dir ./youtube_wav/test --tv_dim 16 --threshold 0.5

"""