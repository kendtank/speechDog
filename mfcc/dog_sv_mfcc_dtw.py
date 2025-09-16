# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/12 18:22
@Author  : Kend
@FileName: dog_sv_mfcc_dtw
@Software: PyCharm
@modifier:
"""

# -*- coding: utf-8 -*-
"""
基于时序特征和DTW的犬吠声纹识别测试脚本
保留时序信息，使用DTW进行序列匹配
"""

import os
import numpy as np
import librosa
from scipy.signal import butter, lfilter
from scipy.spatial.distance import euclidean
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

try:
    from fastdtw import fastdtw

    HAS_DTW = True
except ImportError:
    HAS_DTW = False
    print("警告：未安装fastdtw，将使用scipy.spatial.distance.cdist作为替代")

# ========== CONFIG ==========
enroll_dir = r"../youtube_wav/brakng_dog_datasets"
test_dir = r"../youtube_wav/test"
sr = 16000
n_mfcc = 13  # MFCC维度 (经典13维)
frame_length = 1024  # 帧长 (64ms @ 16kHz)
hop_length = 512  # 帧移 (32ms @ 16kHz)
fmin = 200.0  # 带通下限 (Hz)
fmax = 8000.0  # 带通上限 (Hz)
vad_rms_threshold = 0.001  # VAD阈值
target_rms = 0.1  # 音量归一化目标
trim_top_db = None  # librosa trim阈值
cache_path = "emb_sequence_cache.npz"
min_frames = 20  # 最少帧数
# ==========================

EPS = 1e-9


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, lowcut=fmin, highcut=fmax, fs=sr):
    try:
        b, a = butter_bandpass(lowcut, highcut, fs, order=4)
        y = lfilter(b, a, data)
        return y
    except Exception:
        return data


def normalize_rms(y, target_rms=target_rms):
    rms = np.sqrt(np.mean(y ** 2) + EPS)
    if rms < EPS:
        return y
    return y * (target_rms / (rms + EPS))


def simple_vad_frames(y, frame_length=frame_length, hop_length=hop_length, threshold=vad_rms_threshold):
    # 确保y长度足够
    if len(y) < frame_length:
        return y

    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + EPS)
    keep = rms >= threshold
    if np.sum(keep) == 0:
        return None
    kept = frames[keep]
    # 重构音频
    return kept.flatten()


def load_preprocess(path):
    print(f"[DEBUG] 加载并预处理音频: {path}")
    y, _ = librosa.load(path, sr=sr, mono=True)
    if y is None or y.size == 0:
        print(f"[WARN] 音频为空: {path}")
        return None
    print(f"[DEBUG] 原始音频长度: {len(y)}")

    # 带通滤波
    y = bandpass_filter(y)
    print(f"[DEBUG] 带通滤波后长度: {len(y)}")

    # 可选的静音修剪
    if trim_top_db is not None:
        y, _ = librosa.effects.trim(y, top_db=trim_top_db)
        print(f"[DEBUG] 静音修剪后长度: {len(y)}")

    # VAD: 保留有声音的帧
    y_vad = simple_vad_frames(y)
    if y_vad is None or y_vad.size < 256:
        print(f"[DEBUG] VAD后音频过短，使用原始音频归一化")
        y = normalize_rms(y)
        print(f"[DEBUG] 归一化后RMS: {np.sqrt(np.mean(y ** 2))}")
        return y
    y_vad = normalize_rms(y_vad)
    print(f"[DEBUG] VAD后长度: {len(y_vad)}, 归一化后RMS: {np.sqrt(np.mean(y_vad ** 2))}")
    return y_vad


def extract_features_sequence(y):
    """
    提取时序特征序列，保留时序信息
    返回: (T, D) 的特征序列
    """
    if y is None or len(y) < frame_length:
        print("[WARN] 输入音频过短，无法提取特征")
        return None

    print(f"[DEBUG] 提取时序特征，音频长度: {len(y)}")

    # 提取MFCC特征 (13维)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                n_fft=frame_length, hop_length=hop_length)
    print(f"[DEBUG] MFCC特征形状: {mfcc.shape}")

    # 计算一阶差分
    delta = librosa.feature.delta(mfcc)
    # 计算二阶差分
    delta2 = librosa.feature.delta(mfcc, order=2)

    # 拼接特征 [MFCC + Δ + ΔΔ = 39维]
    features = np.vstack([mfcc, delta, delta2])  # (39, T)
    feat_seq = features.T  # (T, 39)

    print(f"[DEBUG] 特征序列形状: {feat_seq.shape}")

    # 检查帧数
    if feat_seq.shape[0] < min_frames:
        print(f"[WARN] 特征帧数不足 ({feat_seq.shape[0]} < {min_frames})")
        return None

    # 检查无效值
    if np.isnan(feat_seq).any() or np.isinf(feat_seq).any():
        print("[ERROR] 特征序列包含无效值")
        return None

    # L2归一化每帧特征
    norm = np.linalg.norm(feat_seq, axis=1, keepdims=True) + EPS
    feat_seq = feat_seq / norm

    return feat_seq


def dtw_distance(seq1, seq2):
    """
    计算两个特征序列之间的DTW距离
    """
    if not HAS_DTW:
        # 简单替代方案：计算欧氏距离矩阵的平均值
        from scipy.spatial.distance import cdist
        distances = cdist(seq1, seq2, metric='euclidean')
        return np.mean(distances)

    try:
        distance, path = fastdtw(seq1, seq2, dist=euclidean)
        # 归一化距离
        normalized_distance = distance / (len(path) + EPS)
        return normalized_distance
    except Exception as e:
        print(f"[WARN] DTW计算失败: {str(e)}")
        # 备用方案：使用序列长度归一化的欧氏距离
        min_len = min(len(seq1), len(seq2))
        max_len = max(len(seq1), len(seq2))
        # 简单的长度对齐和距离计算
        dist = np.linalg.norm(seq1[:min_len] - seq2[:min_len]) / min_len
        return dist


def dtw_similarity(seq1, seq2):
    """
    将DTW距离转换为相似度
    """
    distance = dtw_distance(seq1, seq2)
    # 转换为相似度 (距离越小相似度越高)
    similarity = 1.0 / (1.0 + distance)
    return similarity


def load_or_build_enroll_cache(enroll_dir, cache_path=cache_path):
    """
    加载或构建注册模板缓存
    """
    if os.path.exists(cache_path):
        try:
            print(f"[DEBUG] 尝试加载缓存文件: {cache_path}")
            d = np.load(cache_path, allow_pickle=True)
            enroll_data = dict(d['enroll'])
            meta_data = dict(d['meta'])
            print("[DEBUG] 缓存加载成功")
            return enroll_data, meta_data
        except Exception as e:
            print(f"[WARN] 缓存加载失败: {str(e)}")
            pass

    print(f"[DEBUG] 构建注册模板库: {enroll_dir}")
    enroll = {}
    meta = {}
    total_files = 0
    processed_files = 0

    for entry in sorted(os.listdir(enroll_dir)):
        p = os.path.join(enroll_dir, entry)
        if not os.path.isdir(p) or entry.lower() == 'test':
            continue
        print(f"[DEBUG] 处理犬只目录: {entry}")

        feat_sequences = []
        file_count = 0
        for fn in sorted(os.listdir(p)):
            if not fn.lower().endswith(('.wav', '.flac', '.mp3')):
                continue
            file_count += 1
            total_files += 1
            fp = os.path.join(p, fn)
            print(f"[DEBUG] 处理注册文件: {fp}")

            y = load_preprocess(fp)
            if y is None:
                print(f"[WARN] 音频预处理失败: {fp}")
                continue

            feat_seq = extract_features_sequence(y)
            if feat_seq is None:
                print(f"[WARN] 特征序列提取失败: {fp}")
                continue

            feat_sequences.append(feat_seq)
            processed_files += 1

        if len(feat_sequences) > 0:
            enroll[entry] = feat_sequences
            meta[entry] = {'n_utts': len(feat_sequences), 'seq_shapes': [seq.shape for seq in feat_sequences]}
            print(f"[ENROLL] {entry}: {len(feat_sequences)} 个样本")
            for i, seq in enumerate(feat_sequences):
                print(f"         样本{i}特征维度: {seq.shape}")
        else:
            print(f"[WARN] 犬只 {entry} 没有有效的注册样本")

    print(
        f"[SUMMARY] 注册阶段: 总文件数={total_files}, 成功处理={processed_files}, 失败={total_files - processed_files}")

    # 保存缓存
    print(f"[DEBUG] 保存缓存到: {cache_path}")
    np.savez_compressed(cache_path, enroll=enroll, meta=meta)
    return enroll, meta


def build_tests(test_dir):
    """
    构建测试集
    """
    print(f"[DEBUG] 构建测试集: {test_dir}")
    tests = {}
    total_files = 0
    processed_files = 0

    for fn in sorted(os.listdir(test_dir)):
        if not fn.lower().endswith(('.wav', '.flac', '.mp3')):
            continue
        total_files += 1
        fp = os.path.join(test_dir, fn)
        print(f"[DEBUG] 处理测试文件: {fp}")

        y = load_preprocess(fp)
        if y is None:
            print(f"[WARN] 音频预处理失败: {fp}")
            continue

        feat_seq = extract_features_sequence(y)
        if feat_seq is None:
            print(f"[WARN] 特征序列提取失败: {fp}")
            continue

        # 推断标签
        label = None
        base = os.path.basename(fn).lower()
        for token in base.replace('-', '_').split('_'):
            if token.startswith('dog'):
                label = token
                break
        tests[fn] = {'feat_seq': feat_seq, 'label': label}
        print(f"[TEST] {fn}: 标签={label}, 特征序列形状={feat_seq.shape}")
        processed_files += 1

    print(
        f"[SUMMARY] 测试阶段: 总文件数={total_files}, 成功处理={processed_files}, 失败={total_files - processed_files}")
    return tests


def score_and_eval(enroll, tests):
    """
    评分和评估
    """
    enroll_ids = list(enroll.keys())
    print("已注册犬只ID:", enroll_ids)

    y_scores = []
    y_true = []
    details = []

    # 用于区分度分析的详细数据
    genuine_scores = []  # 真正例得分
    impostor_scores = []  # 冒充者得分
    per_dog_scores = {dog_id: {'genuine': [], 'impostor': []} for dog_id in enroll_ids}

    for fname, t in tests.items():
        test_seq = t['feat_seq']
        true_label = t['label']

        if test_seq is None:
            print(f"[WARN] 测试文件 {fname} 没有特征序列")
            continue

        print(f"\n[DEBUG] 为测试文件 {fname} 计算匹配分数")
        print(f"[DEBUG] 真实标签: {true_label}")
        print(f"[DEBUG] 测试序列形状: {test_seq.shape}")

        best_id = None
        best_score = -1e9
        per_class_scores = {}

        # 对每个注册犬只计算匹配分数
        for pid, feat_sequences in enroll.items():
            print(f"[DEBUG] 与犬只 {pid} 比较，该犬只有 {len(feat_sequences)} 个注册样本")

            # 与该犬只的所有注册样本比较，取最高分
            similarity_scores = []
            for i, enroll_seq in enumerate(feat_sequences):
                print(f"[DEBUG]   与样本{i}比较，形状: {enroll_seq.shape}")
                try:
                    sim = dtw_similarity(test_seq, enroll_seq)
                    similarity_scores.append(sim)
                    print(f"[DEBUG]   样本{i}相似度: {sim:.4f}")
                except Exception as e:
                    print(f"[WARN]   样本{i}DTW计算失败: {str(e)}")
                    similarity_scores.append(0.0)

            # 取最高相似度作为该犬只的匹配分数
            pid_score = max(similarity_scores) if similarity_scores else 0.0
            per_class_scores[pid] = pid_score
            print(f"[DEBUG] {pid} 的最高相似度: {pid_score:.4f}")

            if pid_score > best_score:
                best_score = pid_score
                best_id = pid

        print(f"[TEST] {fname}")
        for pid, s in per_class_scores.items():
            marker = "★" if pid == true_label else " "
            print(f"   {marker} {pid}: score={s:.4f}")

        if best_id is not None:
            is_correct = best_id == true_label if true_label is not None else False
            print(f"  → 预测 = {best_id}, 得分={best_score:.4f}, 真实={true_label}, 正确={is_correct}")
        else:
            print(f"  → ❌ 未找到匹配, 真实={true_label}")

        # 收集评分用于分析
        if true_label is not None:
            is_genuine = 1 if (best_id == true_label) else 0
            y_scores.append(best_score)
            y_true.append(is_genuine)
            details.append((fname, best_id, best_score, true_label))

            # 按犬只分类收集得分
            if true_label in enroll_ids:
                if is_genuine:
                    genuine_scores.append(best_score)
                    per_dog_scores[true_label]['genuine'].append(best_score)
                else:
                    impostor_scores.append(best_score)
                    if best_id:  # 确保best_id不为None
                        per_dog_scores[best_id]['impostor'].append(best_score)
            else:
                # 未知犬只，视为冒充者
                if impostor_scores is not None:
                    impostor_scores.append(best_score)

    # 详细分析区分度
    print("\n=== 区分度分析 ===")
    if genuine_scores:
        print(
            f"真正例得分统计: 最小={min(genuine_scores):.4f}, 最大={max(genuine_scores):.4f}, 平均={np.mean(genuine_scores):.4f}")
    if impostor_scores:
        print(
            f"冒充者得分统计: 最小={min(impostor_scores):.4f}, 最大={max(impostor_scores):.4f}, 平均={np.mean(impostor_scores):.4f}")

    # 按犬只分析
    for dog_id, scores in per_dog_scores.items():
        print(f"\n犬只 {dog_id} 的得分分布:")
        if scores['genuine']:
            print(f"  真正例: 平均={np.mean(scores['genuine']):.4f}, 标准差={np.std(scores['genuine']):.4f}")
        if scores['impostor']:
            print(f"  冒充者: 平均={np.mean(scores['impostor']):.4f}, 标准差={np.std(scores['impostor']):.4f}")

    # 计算评估指标
    if len(y_scores) > 0 and len(set(y_true)) > 1:
        y_scores = np.array(y_scores)
        y_true = np.array(y_true)

        auc = roc_auc_score(y_true, y_scores)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1 - tpr
        idx = np.nanargmin(np.abs(fnr - fpr))
        eer = (fpr[idx] + fnr[idx]) / 2
        eer_th = thresholds[idx]

        print(f"\n=== 评估结果 ===")
        print(f"AUC={auc:.4f}, EER={eer:.4f}, EER阈值={eer_th:.4f}")

        # 显示不同阈值下的准确率
        print("\n不同阈值下的准确率:")
        for thr in np.linspace(np.min(y_scores), np.max(y_scores), 9):
            preds = (y_scores >= thr).astype(int)
            acc = accuracy_score(y_true, preds)
            print(f"阈值={thr:.4f} 准确率={acc:.4f}")

        # 绘制分数分布直方图
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        if genuine_scores:
            plt.hist(genuine_scores, bins=20, alpha=0.7, label='真正例', color='blue')
        if impostor_scores:
            plt.hist(impostor_scores, bins=20, alpha=0.7, label='冒充者', color='red')
        plt.legend()
        plt.title("分数分布")
        plt.xlabel("匹配分数")
        plt.ylabel("样本数量")
        plt.grid(True, alpha=0.3)

        # 绘制ROC曲线
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}, EER={eer:.3f}", linewidth=2)
        plt.plot([0, 1], [0, 1], '--', color='gray', label='随机分类器')
        plt.xlabel("假正例率")
        plt.ylabel("真正例率")
        plt.title("ROC曲线")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return details, y_scores, y_true
    else:
        print("⚠️ 无法计算评估指标：缺少足够的测试样本或类别")
        return details, None, None


def analyze_sequence_similarity(enroll):
    """
    分析注册样本间的序列相似度
    """
    print("\n=== 注册样本序列相似度分析 ===")

    for dog_id, sequences in enroll.items():
        if len(sequences) > 1:
            print(f"\n犬只 {dog_id} 内部相似度:")
            similarities = []
            for i in range(len(sequences)):
                for j in range(i + 1, len(sequences)):
                    try:
                        sim = dtw_similarity(sequences[i], sequences[j])
                        similarities.append(sim)
                        print(f"  样本{i} vs 样本{j}: 相似度={sim:.4f}")
                    except Exception as e:
                        print(f"  样本{i} vs 样本{j}: 计算失败 ({str(e)})")
            if similarities:
                print(f"  平均相似度: {np.mean(similarities):.4f} ± {np.std(similarities):.4f}")


if __name__ == "__main__":
    print("=== 基于时序特征和DTW的犬吠声纹识别测试 ===")

    if not HAS_DTW:
        print("⚠️  警告：未安装fastdtw库，将使用简化距离计算")
        print("   建议安装: pip install fastdtw")

    # 加载或构建注册模板
    print("\n1. 构建注册模板库...")
    enroll, meta = load_or_build_enroll_cache(enroll_dir)

    # 构建测试集
    print("\n2. 构建测试集...")
    tests = build_tests(test_dir)

    # 分析注册样本相似度
    print("\n3. 分析注册样本相似度...")
    analyze_sequence_similarity(enroll)

    # 评分和评估
    print("\n4. 开始评分和评估...")
    details, scores, labels = score_and_eval(enroll, tests)

    print("\n=== 测试完成 ===")



"""
[TEST] dog5_test_02.WAV
     dog1: score=0.7723
     dog2: score=0.7450
     dog3: score=0.8099
     dog4: score=0.7946
   ★ dog5: score=0.8561
  → 预测 = dog5, 得分=0.8561, 真实=dog5, 正确=True

=== 区分度分析 ===
真正例得分统计: 最小=0.7826, 最大=1.0000, 平均=0.8752
总结：有提升，但是提升不大
"""