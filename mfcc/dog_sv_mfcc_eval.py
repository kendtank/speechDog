# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/12 下午4:54
@Author  : Kend
@FileName: dog_sv_mfcc_robust.py（debug）
@Software: PyCharm
@modifier:
"""

"""
更鲁棒且轻量的传统特征声纹识别流程（对于端侧友好）
调试增强版 - 提供详细的区分度分析
"""

import os
import numpy as np
import librosa
from scipy.signal import butter, lfilter
from scipy.stats import skew, kurtosis
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ========== CONFIG ==========
enroll_dir = r"./youtube_wav/brakng_dog_datasets"  # each subfolder = dog id
test_dir = r"./youtube_wav/test"
sr = 16000
n_mels = 40  # mel bins (40 is a good compromise)
use_mfcc = False  # 若 True 使用 MFCC (n_mfcc=n_mels)
fmin = 200.0  # 带通下限 (Hz)
fmax = 12000.0  # 带通上限 (Hz)
vad_rms_threshold = 0.001  # VAD 阈值（RMS），可调
target_rms = 0.1  # 音量归一化目标 RMS（避免极小值）
trim_top_db = None  # 若不想用 librosa trim, 设为 None
alpha = 0.75  # 融合权重 (cos vs euc)
cache_path = "emb_cache_debug.npz"
min_frames = 10  # 增加最少帧数阈值
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


def simple_vad_frames(y, frame_length=1024, hop_length=512, threshold=vad_rms_threshold):
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + EPS)
    keep = rms >= threshold
    if np.sum(keep) == 0:
        return None
    kept = frames[keep]
    # reconstruct by concatenation (approx)
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


def compute_feats(y):
    # 返回帧级特征 shape (T, D)
    if y is None or y.size == 0:
        print("[WARN] 输入音频为空，无法提取特征")
        return None
    print(f"[DEBUG] 提取特征，音频长度: {len(y)}")

    if use_mfcc:
        # MFCC路径
        print("[DEBUG] 使用MFCC特征")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mels)
        feat = mfcc.T  # (T, n_mels)
    else:
        # log-mel路径（对噪声更鲁棒）
        print("[DEBUG] 使用Log-Mel特征")
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0)
        logS = np.log(S + 1e-9)
        feat = logS.T  # (T, n_mels)

    print(f"[DEBUG] 基础特征形状: {feat.shape}")

    # 检查基础特征是否有效
    if np.isnan(feat).any() or np.isinf(feat).any():
        print("[ERROR] 基础特征包含NaN或无穷大值")
        return None

    # 添加差分特征
    delta = librosa.feature.delta(feat.T).T
    delta2 = librosa.feature.delta(feat.T, order=2).T

    # 检查差分特征是否有效
    if np.isnan(delta).any() or np.isinf(delta).any():
        print("[ERROR] 一阶差分特征包含NaN或无穷大值")
        return None

    if np.isnan(delta2).any() or np.isinf(delta2).any():
        print("[ERROR] 二阶差分特征包含NaN或无穷大值")
        return None

    feat_full = np.concatenate([feat, delta, delta2], axis=1)  # (T, 3*n_mels)
    print(f"[DEBUG] 添加差分后特征形状: {feat_full.shape}")

    if feat_full.shape[0] < min_frames:
        print(f"[WARN] 特征帧数不足 ({feat_full.shape[0]} < {min_frames})")
        return None

    # 检查最终特征是否有效
    if np.isnan(feat_full).any() or np.isinf(feat_full).any():
        print("[ERROR] 最终特征包含NaN或无穷大值")
        return None

    return feat_full


def extract_pitch_stats(y):
    # 使用librosa.pyin提取基频统计信息
    try:
        print("[DEBUG] 提取基频特征")
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=5000, sr=sr)
        # 移除NaN值
        f0_valid = f0[~np.isnan(f0)]
        if f0_valid.size == 0:
            print("[DEBUG] 未检测到有效基频")
            return 0.0, 0.0
        pmean = float(np.mean(f0_valid))
        pstd = float(np.std(f0_valid))
        print(f"[DEBUG] 基频统计 - 均值: {pmean:.2f}, 标准差: {pstd:.2f}")
        return pmean, pstd
    except Exception as e:
        print(f"[DEBUG] 基频提取失败: {str(e)}")
        return 0.0, 0.0


def safe_stats_computation(feat_seq):
    """安全地计算统计特征，避免NaN和无穷大值"""
    print(f"[DEBUG] 计算统计特征，输入形状: {feat_seq.shape}")

    # 检查输入
    if feat_seq is None or feat_seq.shape[0] == 0:
        print("[ERROR] 输入特征序列为空")
        return None, None, None, None

    if np.isnan(feat_seq).any() or np.isinf(feat_seq).any():
        print("[ERROR] 输入特征包含NaN或无穷大值")
        return None, None, None, None

    # 计算统计特征，增加异常处理
    try:
        mean_ = np.mean(feat_seq, axis=0)
        std_ = np.std(feat_seq, axis=0)

        # 对于偏度和峰度，需要足够的数据点
        if feat_seq.shape[0] < 3:
            print(f"[WARN] 特征帧数过少 ({feat_seq.shape[0]} < 3)，使用零值填充偏度和峰度")
            skew_ = np.zeros_like(mean_)
            kurt_ = np.zeros_like(mean_)
        else:
            # 使用有限值计算
            skew_ = skew(feat_seq, axis=0, nan_policy='omit')
            kurt_ = kurtosis(feat_seq, axis=0, nan_policy='omit')

        # 处理可能的NaN值
        mean_ = np.nan_to_num(mean_, nan=0.0, posinf=0.0, neginf=0.0)
        std_ = np.nan_to_num(std_, nan=0.0, posinf=0.0, neginf=0.0)
        skew_ = np.nan_to_num(skew_, nan=0.0, posinf=0.0, neginf=0.0)
        kurt_ = np.nan_to_num(kurt_, nan=0.0, posinf=0.0, neginf=0.0)

        return mean_, std_, skew_, kurt_

    except Exception as e:
        print(f"[ERROR] 统计特征计算异常: {str(e)}")
        # 返回零向量作为后备
        dim = feat_seq.shape[1] if len(feat_seq.shape) > 1 else feat_seq.shape[0]
        return np.zeros(dim), np.zeros(dim), np.zeros(dim), np.zeros(dim)


def embed_from_feat(feat_seq, y=None):
    # 从特征序列生成嵌入向量
    if feat_seq is None:
        print("[WARN] 特征序列为空，无法生成嵌入")
        return None
    print(f"[DEBUG] 从特征生成嵌入，特征形状: {feat_seq.shape}")

    # 安全计算统计特征
    mean_, std_, skew_, kurt_ = safe_stats_computation(feat_seq)

    if mean_ is None:
        print("[ERROR] 统计特征计算失败")
        return None

    vec = np.concatenate([mean_, std_, skew_, kurt_], axis=0)
    print(f"[DEBUG] 统计特征维度: {len(vec)}")

    # 检查拼接后的向量
    if np.isnan(vec).any() or np.isinf(vec).any():
        print("[ERROR] 拼接后的统计特征包含NaN或无穷大值")
        return None

    # 添加基频统计信息
    if y is not None:
        pmean, pstd = extract_pitch_stats(y)
    else:
        pmean, pstd = 0.0, 0.0
    vec = np.concatenate([vec, np.array([pmean, pstd])], axis=0)
    print(f"[DEBUG] 添加基频特征后维度: {len(vec)}")

    # L2归一化
    norm = np.linalg.norm(vec) + EPS
    if norm < EPS:
        print("[WARN] 向量范数过小，可能导致数值不稳定")
        norm = 1.0
    normalized_vec = vec / norm
    print(f"[DEBUG] 归一化后向量范数: {np.linalg.norm(normalized_vec)}")

    return normalized_vec


def load_or_build_enroll_cache(enroll_dir, cache_path=cache_path):
    if os.path.exists(cache_path):
        try:
            print(f"[DEBUG] 尝试加载缓存文件: {cache_path}")
            d = np.load(cache_path, allow_pickle=True)
            print("[DEBUG] 缓存加载成功")
            return dict(d['enroll']), dict(d['meta'])
        except Exception as e:
            print(f"[WARN] 缓存加载失败: {str(e)}")
            pass

    print(f"[DEBUG] 构建注册模板库: {enroll_dir}")
    enroll = {}
    meta = {}
    total_files = 0
    processed_files = 0
    failed_reasons = {}  # 记录失败原因

    for entry in sorted(os.listdir(enroll_dir)):
        p = os.path.join(enroll_dir, entry)
        if not os.path.isdir(p) or entry.lower() == 'test':
            continue
        print(f"[DEBUG] 处理犬只目录: {entry}")

        embs = []
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
                failed_reasons[fp] = "预处理失败"
                continue

            feat = compute_feats(y)
            if feat is None:
                print(f"[WARN] 特征提取失败: {fp}")
                failed_reasons[fp] = "特征提取失败"
                continue

            emb = embed_from_feat(feat, y)
            if emb is None:
                print(f"[WARN] 嵌入向量生成失败: {fp}")
                failed_reasons[fp] = "嵌入向量生成失败"
                continue

            embs.append(emb)
            processed_files += 1

        if len(embs) > 0:
            enroll[entry] = embs
            meta[entry] = {'n_utts': len(embs)}
            print(f"[ENROLL] {entry}: {len(embs)} 个样本, 嵌入维度={len(embs[0])}")
        else:
            print(f"[WARN] 犬只 {entry} 没有有效的注册样本")

    print(
        f"[SUMMARY] 注册阶段: 总文件数={total_files}, 成功处理={processed_files}, 失败={total_files - processed_files}")

    # 打印失败原因统计
    if failed_reasons:
        print("[FAILURES] 失败原因统计:")
        reason_count = {}
        for reason in failed_reasons.values():
            reason_count[reason] = reason_count.get(reason, 0) + 1
        for reason, count in reason_count.items():
            print(f"  {reason}: {count} 次")

    # 保存缓存
    print(f"[DEBUG] 保存缓存到: {cache_path}")
    np.savez_compressed(cache_path, enroll=enroll, meta=meta)
    return enroll, meta


def build_tests(test_dir):
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

        feat = compute_feats(y)
        if feat is None:
            print(f"[WARN] 特征提取失败: {fp}")
            continue

        emb = embed_from_feat(feat, y)
        if emb is None:
            print(f"[WARN] 嵌入向量生成失败: {fp}")
            continue

        # 推断标签
        label = None
        base = os.path.basename(fn).lower()
        for token in base.replace('-', '_').split('_'):
            if token.startswith('dog'):
                label = token
                break
        tests[fn] = {'emb': emb, 'y': y, 'label': label}
        print(f"[TEST] {fn}: 标签={label}, 嵌入生成={'成功' if emb is not None else '失败'}")
        if emb is not None:
            print(f"[DEBUG] {fn}: 嵌入范数={np.linalg.norm(emb):.6f}")
        processed_files += 1

    print(
        f"[SUMMARY] 测试阶段: 总文件数={total_files}, 成功处理={processed_files}, 失败={total_files - processed_files}")
    return tests


def analyze_embeddings(enroll, tests):
    """分析嵌入向量的统计特性"""
    print("\n=== 嵌入向量分析 ===")

    # 分析注册嵌入
    all_enroll_embs = []
    for dog_id, embs in enroll.items():
        all_enroll_embs.extend(embs)
        print(f"[ANALYSIS] {dog_id} - 样本数: {len(embs)}, 嵌入维度: {len(embs[0]) if embs else 0}")

    if all_enroll_embs:
        all_enroll_embs = np.array(all_enroll_embs)
        print(f"[ANALYSIS] 总注册样本数: {len(all_enroll_embs)}")
        print(f"[ANALYSIS] 嵌入向量统计 - 均值: {np.mean(all_enroll_embs):.6f}, "
              f"标准差: {np.std(all_enroll_embs):.6f}")

    # 分析测试嵌入
    test_embs = [t['emb'] for t in tests.values() if t['emb'] is not None]
    if test_embs:
        test_embs = np.array(test_embs)
        print(f"[ANALYSIS] 测试样本数: {len(test_embs)}")
        print(f"[ANALYSIS] 测试嵌入统计 - 均值: {np.mean(test_embs):.6f}, "
              f"标准差: {np.std(test_embs):.6f}")


def score_and_eval(enroll, tests, alpha=alpha):
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
        temb = t['emb']
        true_label = t['label']

        if temb is None:
            print(f"[WARN] 测试文件 {fname} 没有嵌入向量")
            continue

        best_id = None
        best_score = -1e9
        best_cos, best_euc = 0.0, 0.0
        per_class_scores = {}

        print(f"\n[DEBUG] 为测试文件 {fname} 计算匹配分数")
        print(f"[DEBUG] 真实标签: {true_label}")

        for pid, embs in enroll.items():
            if not embs:
                continue
            print(f"[DEBUG] 与犬只 {pid} 比较")

            # 计算余弦相似度
            cos_vals = []
            for e in embs:
                if temb is not None and e is not None:
                    norm_temb = np.linalg.norm(temb) + EPS
                    norm_e = np.linalg.norm(e) + EPS
                    if norm_temb < EPS or norm_e < EPS:
                        cos_vals.append(0.0)
                    else:
                        cos_val = np.dot(temb, e) / (norm_temb * norm_e)
                        cos_vals.append(cos_val)
                else:
                    cos_vals.append(0.0)
            cos_vals = [np.nan_to_num(v) for v in cos_vals]
            pid_cos = max(cos_vals) if cos_vals else 0.0
            print(f"[DEBUG] {pid} 的最大余弦相似度: {pid_cos:.4f}")

            # 计算欧氏距离相似度
            euc_vals = []
            for e in embs:
                if temb is not None and e is not None:
                    dist = np.linalg.norm(temb - e)
                    euc_vals.append(dist)
                else:
                    euc_vals.append(1e9)
            euc_vals = [np.nan_to_num(v) for v in euc_vals]
            euc_min = min(euc_vals) if euc_vals else 1e9
            euc_sim = 1.0 / (1.0 + euc_min)
            print(f"[DEBUG] {pid} 的欧氏相似度: {euc_sim:.4f}")

            # 综合评分
            score = alpha * pid_cos + (1 - alpha) * euc_sim
            per_class_scores[pid] = score
            print(f"[DEBUG] {pid} 的综合评分: {score:.4f}")

            if score > best_score:
                best_score = score
                best_id = pid
                best_cos = pid_cos
                best_euc = euc_sim

        print(f"[TEST] {fname}")
        for pid, s in per_class_scores.items():
            marker = "★" if pid == true_label else " "
            print(f"   {marker} {pid}: score={s:.4f}")

        if best_id is not None:
            is_correct = best_id == true_label if true_label is not None else False
            print(
                f"  → 预测 = {best_id}, 得分={best_score:.4f}, 余弦={best_cos:.4f}, 欧氏={best_euc:.4f}, 真实={true_label}, 正确={is_correct}")
        else:
            print(f"  → ❌ 未找到匹配, 真实={true_label}")

        # 收集评分用于分析
        if true_label is not None:
            is_genuine = 1 if (best_id == true_label) else 0
            y_scores.append(best_score)
            y_true.append(is_genuine)
            details.append((fname, best_id, best_score, best_cos, best_euc, true_label))

            # 按犬只分类收集得分
            if true_label in enroll_ids:
                if is_genuine:
                    genuine_scores.append(best_score)
                    per_dog_scores[true_label]['genuine'].append(best_score)
                else:
                    impostor_scores.append(best_score)
                    per_dog_scores[best_id]['impostor'].append(best_score)
            else:
                # 未知犬只，视为冒充者
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


def compare_embedding_similarity(enroll):
    """分析注册样本间的相似度矩阵"""
    print("\n=== 注册样本相似度分析 ===")

    for dog_id, embs in enroll.items():
        if len(embs) > 1:
            print(f"\n犬只 {dog_id} 内部相似度:")
            similarities = []
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    # 确保向量有效
                    if embs[i] is None or embs[j] is None:
                        continue
                    norm_i = np.linalg.norm(embs[i])
                    norm_j = np.linalg.norm(embs[j])
                    if norm_i < EPS or norm_j < EPS:
                        cos_sim = 0.0
                    else:
                        cos_sim = np.dot(embs[i], embs[j]) / (norm_i * norm_j)
                    euc_dist = np.linalg.norm(embs[i] - embs[j])
                    similarities.append(cos_sim)
                    print(f"  样本{i} vs 样本{j}: 余弦={cos_sim:.4f}, 欧氏={euc_dist:.4f}")
            if similarities:
                print(f"  平均余弦相似度: {np.mean(similarities):.4f} ± {np.std(similarities):.4f}")


if __name__ == "__main__":
    print("=== 开始调试分析 ===")

    # 加载或构建注册模板
    print("\n1. 构建注册模板库...")
    enroll, meta = load_or_build_enroll_cache(enroll_dir)

    # 构建测试集
    print("\n2. 构建测试集...")
    tests = build_tests(test_dir)

    # 分析嵌入向量
    print("\n3. 分析嵌入向量...")
    analyze_embeddings(enroll, tests)

    # 分析注册样本相似度
    print("\n4. 分析注册样本相似度...")
    compare_embedding_similarity(enroll)

    # 评分和评估
    print("\n5. 开始评分和评估...")
    details, scores, labels = score_and_eval(enroll, tests, alpha=alpha)

    print("\n=== 调试分析完成 ===")


"""
结论: 区分效果还是不好
"""