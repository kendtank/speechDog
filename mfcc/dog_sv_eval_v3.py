# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/16 14:15
@Author  : Kend
@FileName: dog_sv_eval_v3.py
@Software: PyCharm
"""

"""
Dog Speaker Verification v3
专门为单只狗的声纹验证设计，适用于端侧部署

改进点：
1. 特征 = MFCC + Δ + ΔΔ + log-Mel
2. CMVN 标准化
3. PCA 降维
4. 支持多种分类器（SVM、余弦相似度）
5. 阈值机制，支持"目标狗"和"非目标"分类
6. 专为单狗验证设计
"""

import os
import argparse
import numpy as np
import librosa
import time  # Add time module for timing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.svm import SVC

# ========== 特征提取 ==========
def extract_features(wav_path, sr=16000, n_mfcc=20, n_mels=40):
    y, sr = librosa.load(wav_path, sr=sr)
    
    # 如果音频太短，进行重采样延长
    if len(y) < sr * 0.1:  # 如果小于0.1秒
        # 重复音频直到达到最小长度
        repeat_times = int(np.ceil((sr * 0.1) / len(y)))
        y = np.tile(y, repeat_times)
        y = y[:int(sr * 0.1)]  # 裁剪到精确长度

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # 确保有足够的帧来计算delta特征
    if mfcc.shape[1] < 9:  # librosa默认窗口宽度为9
        # 通过重复帧来增加帧数
        repeat_times = int(np.ceil(9 / mfcc.shape[1]))
        mfcc = np.tile(mfcc, (1, repeat_times))
        mfcc = mfcc[:, :9]  # 裁剪到精确帧数
    
    delta1 = librosa.feature.delta(mfcc, order=1, width=min(5, mfcc.shape[1]//2*2+1))
    delta2 = librosa.feature.delta(mfcc, order=2, width=min(5, mfcc.shape[1]//2*2+1))

    # log-Mel
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    logmel = librosa.power_to_db(mel)
    
    # 对log-Mel特征也进行帧数检查
    if logmel.shape[1] < 9:
        repeat_times = int(np.ceil(9 / logmel.shape[1]))
        logmel = np.tile(logmel, (1, repeat_times))
        logmel = logmel[:, :9]

    # 拼接所有特征
    feat = np.vstack([
        mfcc, delta1, delta2, 
        logmel
    ])

    # 时间平均池化
    feat = np.mean(feat, axis=1)

    return feat

# ========== 构建目标狗的模型 ==========
def build_target_model(file_list, target_dog_id, classifier_type='cosine', pca=None, scaler=None):
    start_time = time.time()  # Start timing
    
    # 收集目标狗的特征
    target_feats = []
    # 收集非目标狗的特征（用于SVM训练）
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
    
    target_feats = np.array(target_feats)
    non_target_feats = np.array(non_target_feats) if len(non_target_feats) > 0 else np.array([])

    # 标准化（使用目标狗的特征拟合标准化参数）
    if scaler is None:
        scaler = StandardScaler().fit(target_feats)
    
    # 标准化所有特征
    target_feats = scaler.transform(target_feats)
    if len(non_target_feats) > 0:
        non_target_feats = scaler.transform(non_target_feats)

    # PCA（使用目标狗的特征拟合PCA参数）
    if pca is None:
        n_components = min(32, len(target_feats), target_feats.shape[1])
        if n_components < 1:
            n_components = 1
        pca = PCA(n_components=n_components)
        target_feats_transformed = pca.fit_transform(target_feats)
        if len(non_target_feats) > 0:
            non_target_feats_transformed = pca.transform(non_target_feats)
        else:
            non_target_feats_transformed = np.array([])
    else:
        target_feats_transformed = pca.transform(target_feats)
        if len(non_target_feats) > 0:
            non_target_feats_transformed = pca.transform(non_target_feats)
        else:
            non_target_feats_transformed = np.array([])

    # 根据分类器类型构建模型
    if classifier_type == 'cosine':
        # 目标模型是目标狗特征的均值向量
        model = np.mean(target_feats_transformed, axis=0)
    elif classifier_type == 'svm':
        # 使用SVM分类器
        # 准备训练数据
        X_train = target_feats_transformed
        y_train = [1] * len(target_feats_transformed)  # 正样本标签
        
        # 如果有非目标狗的样本，也加入训练集作为负样本
        if len(non_target_feats_transformed) > 0:
            X_train = np.vstack([X_train, non_target_feats_transformed])
            y_train.extend([0] * len(non_target_feats_transformed))  # 负样本标签
        else:
            # 如果没有负样本，创建一些人工负样本（通过添加噪声）
            noise_samples = []
            for feat in target_feats_transformed:
                noise = np.random.normal(0, 0.1, feat.shape)
                noise_samples.append(feat + noise)
            noise_samples = np.array(noise_samples)
            X_train = np.vstack([X_train, noise_samples])
            y_train.extend([0] * len(noise_samples))
        
        # 训练SVM
        model = SVC(kernel='rbf', probability=True)
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")
    
    end_time = time.time()  # End timing
    print(f"[INFO] 模型构建完成，耗时 {end_time - start_time:.2f} 秒")
    
    return model, pca, scaler

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
def evaluate(model, classifier_type, test_feats, test_ids, test_files, target_dog_id, threshold=0.0):
    # 打印测试信息
    test_filenames = [os.path.basename(f) for f in test_files]
    print("[INFO] 构建测试集...")
    print(f"[INFO] 测试文件列表: {test_filenames}")
    
    scores = []
    labels = []
    
    start_time = time.time()
    
    print(f"[INFO] 使用阈值: {threshold}")
    print(f"[INFO] 识别结果 (目标狗为{target_dog_id}):")
    
    for i, feat in enumerate(test_feats):
        # 根据分类器类型计算分数
        if classifier_type == 'cosine':
            score = cosine_score(model, feat)
        elif classifier_type == 'svm':
            score = model.predict_proba([feat])[0][1]  # 正类概率
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
        
        # 根据阈值判断是否为目标狗
        pred_result = "是目标狗" if score >= threshold else "非目标狗"
        
        # 获取真实标签
        true_id = test_ids[i]
        true_label = "是目标狗" if true_id == "target" else "非目标狗"
        
        # 保存分数和标签
        scores.append(score)
        labels.append(int(true_label == "是目标狗"))
        
        # 打印当前测试文件的结果
        filename = test_filenames[i]
        is_correct = "✓" if pred_result == true_label else "✗"
        print(f"  {filename}: {is_correct} 识别为[{pred_result}] (相似度={score:.4f}) 实际为[{true_label}]")

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
    parser = argparse.ArgumentParser(description="Dog Speaker Verification v3 - Single dog verification")
    parser.add_argument("--enroll_dir", type=str, required=True, 
                        help="Directory containing enrollment samples")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Directory containing test samples")
    parser.add_argument("--target_dog", type=str, required=True,
                        help="Target dog ID (e.g., dog1, dog2)")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Threshold for verification. Scores >= threshold are classified as target dog")
    parser.add_argument("--classifier", type=str, default='cosine', choices=['cosine', 'svm'],
                        help="Classifier type for verification")
    args = parser.parse_args()

    # 注册目标狗模型
    enroll_files = collect_wavs(args.enroll_dir)
    if len(enroll_files) == 0:
        raise RuntimeError(f"[ERROR] No .wav files found in enroll_dir: {args.enroll_dir}")
    
    print(f"[INFO] 为狗 {args.target_dog} 构建声纹模型，使用 {len(enroll_files)} 个样本...")
    model, pca, scaler = build_target_model(enroll_files, args.target_dog, args.classifier)

    # 测试
    test_files = collect_wavs(args.test_dir)
    if len(test_files) == 0:
        raise RuntimeError(f"[ERROR] No .wav files found in test_dir: {args.test_dir}")
    
    print(f"[INFO] 准备测试，共 {len(test_files)} 个文件...")
    
    # 提取测试特征
    test_feats = []
    test_ids = []
    for f in test_files:
        try:
            x = extract_features(f)
            # 使用注册时的标准化和PCA
            x = scaler.transform([x])[0]
            x = pca.transform([x])[0]
            test_feats.append(x)
            # 根据目标狗ID解析文件标签
            test_ids.append(parse_id_from_filename(os.path.basename(f), args.target_dog))
        except Exception as e:
            print(f"[WARN] 处理测试文件 {f} 时出错: {e}")
            continue
    
    test_feats = np.array(test_feats)

    # 评估
    evaluate(model, args.classifier, test_feats, test_ids, test_files, args.target_dog, args.threshold)


if __name__ == "__main__":
    main()

"""
# 余弦相似度方法
python mfcc/dog_sv_eval_v3.py --enroll_dir ./youtube_wav/brakng_dog_datasets --test_dir ./youtube_wav/test --target_dog dog1 --threshold 0.3 --classifier cosine

# SVM方法
python mfcc/dog_sv_eval_v3.py --enroll_dir ./youtube_wav/brakng_dog_datasets --test_dir ./youtube_wav/test --target_dog dog4 --threshold 0.5 --classifier svm



(ai) D:\kend\myPython\speechDog-master>python mfcc/dog_sv_eval_v3.py --enroll_dir ./youtube_wav/brakng_dog_datasets --test_dir ./youtube_wav/test --target_dog dog1 --threshold 0.3 --classifier cosine
[INFO] 为狗 dog1 构建声纹模型，使用 25 个样本...
[INFO] 模型构建完成，耗时 0.89 秒
[INFO] 准备测试，共 9 个文件...
[INFO] 构建测试集...
[INFO] 测试文件列表: ['bad.WAV', 'bad2.WAV', 'dog1_test_01.WAV', 'dog2_test_01.WAV', 'dog3_test_01.WAV', 'dog4_test_01.WAV', 'dog4_test_02.WAV', 'dog5_test_01.WAV', 'dog5_test_02.WAV']
[INFO] 使用阈值: 0.3
[INFO] 识别结果 (目标狗为dog1):
  bad.WAV: ✓ 识别为[非目标狗] (相似度=-0.6711) 实际为[非目标狗]
  bad2.WAV: ✓ 识别为[非目标狗] (相似度=-0.6799) 实际为[非目标狗]
  dog1_test_01.WAV: ✗ 识别为[非目标狗] (相似度=-0.3313) 实际为[是目标狗]
  dog2_test_01.WAV: ✓ 识别为[非目标狗] (相似度=-0.7501) 实际为[非目标狗]
  dog3_test_01.WAV: ✓ 识别为[非目标狗] (相似度=-0.6803) 实际为[非目标狗]
  dog4_test_01.WAV: ✓ 识别为[非目标狗] (相似度=-0.2865) 实际为[非目标狗]
  dog4_test_02.WAV: ✓ 识别为[非目标狗] (相似度=-0.6352) 实际为[非目标狗]
  dog5_test_01.WAV: ✓ 识别为[非目标狗] (相似度=-0.6633) 实际为[非目标狗]
  dog5_test_02.WAV: ✓ 识别为[非目标狗] (相似度=-0.6925) 实际为[非目标狗]
[INFO] 识别完成，耗时 0.00 秒
[RESULT] 性能指标: EER=0.1250, AUC=0.8750, 准确率=0.8889
[INFO] 推荐阈值: -0.3313

(ai) D:\kend\myPython\speechDog-master>
(ai) D:\kend\myPython\speechDog-master>
(ai) D:\kend\myPython\speechDog-master>
(ai) D:\kend\myPython\speechDog-master>python mfcc/dog_sv_eval_v3.py --enroll_dir ./youtube_wav/brakng_dog_datasets --test_dir ./youtube_wav/test --target_dog dog4 --threshold 0.3  --classifier svm   
[INFO] 为狗 dog4 构建声纹模型，使用 25 个样本...
[INFO] 模型构建完成，耗时 0.89 秒
[INFO] 准备测试，共 9 个文件...
[INFO] 构建测试集...
[INFO] 测试文件列表: ['bad.WAV', 'bad2.WAV', 'dog1_test_01.WAV', 'dog2_test_01.WAV', 'dog3_test_01.WAV', 'dog4_test_01.WAV', 'dog4_test_02.WAV', 'dog5_test_01.WAV', 'dog5_test_02.WAV']
[INFO] 使用阈值: 0.3
[INFO] 识别结果 (目标狗为dog4):
  bad.WAV: ✓ 识别为[非目标狗] (相似度=0.2779) 实际为[非目标狗]
  bad2.WAV: ✓ 识别为[非目标狗] (相似度=0.2798) 实际为[非目标狗]
  dog1_test_01.WAV: ✓ 识别为[非目标狗] (相似度=0.2282) 实际为[非目标狗]
  dog2_test_01.WAV: ✓ 识别为[非目标狗] (相似度=0.2299) 实际为[非目标狗]
  dog3_test_01.WAV: ✓ 识别为[非目标狗] (相似度=0.2466) 实际为[非目标狗]
  dog4_test_01.WAV: ✓ 识别为[是目标狗] (相似度=0.3016) 实际为[是目标狗]
  dog4_test_02.WAV: ✗ 识别为[非目标狗] (相似度=0.2610) 实际为[是目标狗]
  dog5_test_01.WAV: ✓ 识别为[非目标狗] (相似度=0.2385) 实际为[非目标狗]
  dog5_test_02.WAV: ✓ 识别为[非目标狗] (相似度=0.2775) 实际为[非目标狗]
[INFO] 识别完成，耗时 0.00 秒
[RESULT] 性能指标: EER=0.4286, AUC=0.7857, 准确率=0.8889
[INFO] 推荐阈值: 0.2775

实验得出：使用传统的物理信号和数学方法区分度不够高， 这种模版匹配的鲁棒性太弱
"""