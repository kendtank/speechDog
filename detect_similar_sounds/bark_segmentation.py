# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/17 上午10:31
@Author  : Kend
@FileName: bark_segmentation.py
@Software: PyCharm
@modifier:
"""

"""
只对"尖锐的狗吠瞬时"做模板和比对，比把整条音频（包含大量静音/背景）直接做匹配要可靠得多。

流程：
    先检测"候选短段"（candidate peaks）：自动从连续录音里找到"尖、短、有能量突变"的段落（0.1–2s）。
    判定是否为狗叫（bark detector）：对候选段用一些轻量规则或一个小分类器判断是不是狗叫（而非门声、拍手、鸟叫）。（这里待定， 可能不需要做，直接去验证是不是狗吠声音）
    把合格的段落当作 enrollment/verification 的基本单位：只以这些段落提特征（MFCC / log-mel / i-vector / MFCC统计），并与模板比对。
    多段融合：一次检测到多个短段，可能把多个段的 embedding 求均值或取最大分数作为最终判定。

用到信号学指标（对"尖"有鉴别力）
    短时能量 / RMS（能量高）
    谱流（spectral flux）：频谱突变大说明"尖"或突发声。
    谱质心（spectral centroid）：高说明频谱集中在高频，狗吠常有高频成分。
    零交叉率（ZCR）：尖锐脉冲型信号通常 ZCR 较高或变化剧烈。
    短时熵 / 谱熵：低熵代表脉冲峰值；高熵代表噪声。
    持续时间：观察到狗吠 0.2–1s，很重要作为过滤条件。
    短时峰值检测 / 均方根门限 + 滑窗平滑 + 双阈值（hysteresis）。
    
前端处理流程：
    预处理：高通或带通（例如 200–8000 Hz）以去掉低频风噪与直流。
    分帧（25ms/10ms），计算每帧的 RMS、谱流、谱质心、ZCR。
    组合得分：score = w1*norm(RMS) + w2*norm(spectral_flux) + w3*norm(centroid)。
    双阈值峰检测：上阈值触发 start，下阈值触发 end（避免抖动）。
    合并短间隔段（若两段间隔 < 50–100ms，则合并）。
    按时长过滤：保留 0.1s–0.6s 的段。
    对每段做"狗叫判定"规则：例如 centroid_mean > Cc && flux_mean > Cf && rms_mean > Cr → 通过；否则丢弃或送小分类器。
    输出候选段时间戳并保存短音频片段用于后续提特征/比对。
"""


import numpy as np
import librosa
import scipy.signal as sps
import soundfile as sf  # 使用soundfile替代librosa.output

SR = 16000  # 采样率，一秒钟采 16000 个点来表示声音波形。采样率越高，能表示的声音细节越多。普通语音识别、声纹：16kHz（足够捕捉人声和狗吠）。
FRAME_LEN = 0.025
FRAME_HOP = 0.01

def bandpass(y, sr=SR, low=200, high=8000, order=4):
    """
    带通滤波器，用于去除低频噪声和直流分量 这里做的一个巴特沃斯带通滤波器
    Args:
        y: 音频信号-一堆采样点
        sr: 采样率-16000 Hz
        low: 低频截止频率-200 Hz 以下的声音去掉，比如风声、车轱辘 rumble
        high: 高频截止频率-8000 Hz 以上去掉，因为采样率是 16kHz，奈奎斯特频率一半 = 8000Hz，已经到极限了
        order: 滤波器阶数-（阶数越高，滤波越陡峭，但计算量更大，也容易失真）
    Returns:
        滤波后的音频信号 NOTE: (时间长度保持不变, 被过滤掉的声音不会消失在时间上, 只是被削弱（趋近于 0)
    """
    nyq = 0.5 * sr  # 计算奈奎斯特频率
    # 确保频率在有效范围内
    low = max(low, 1)  # 最小频率不能为0
    high = min(high, nyq - 1)  # 最大频率必须小于奈奎斯特频率
    
    # 检查频率是否有效
    if low >= high or low <= 0 or high >= nyq:
        raise ValueError(f"Invalid frequency range: low={low}, high={high}, nyquist={nyq}")
    
    # 归一化频率
    low_norm = low / nyq
    high_norm = high / nyq

    # 生成滤波器系数。
    b, a = sps.butter(order, [low_norm, high_norm], btype='band')
    # 对信号前后滤波，去除低频噪声和高频杂音
    return sps.filtfilt(b, a, y)

def frame_signal(y, sr=SR, frame_len=FRAME_LEN, frame_hop=FRAME_HOP):
    """
    将音频信号分帧
    Args:
        y: 音频信号
        sr: 采样率
        frame_len: 帧长度(秒)
        frame_hop: 帧移(秒)
    Returns:
        分帧后的信号矩阵，每行是一帧
    """
    fl = int(frame_len*sr)
    fh = int(frame_hop*sr)
    frames = librosa.util.frame(y, frame_length=fl, hop_length=fh).T
    return frames

def spectral_flux(frames, n_fft=512):
    """
    计算谱流特征，衡量频谱变化程度
    Args:
        frames: 分帧后的信号
        n_fft: FFT点数
    Returns:
        每帧的谱流值
    """
    # frames: (T, frame_len)
    S = np.abs(np.fft.rfft(frames * np.hanning(frames.shape[1]), n=n_fft, axis=1))
    S = S / (np.sum(S, axis=1, keepdims=True) + 1e-8)
    flux = np.sqrt(np.sum((np.diff(S, axis=0))**2, axis=1))
    flux = np.concatenate([[0.0], flux])
    return flux

def spectral_entropy(frames, n_fft=512):
    """
    计算谱熵，用于区分脉冲信号和噪声
    Args:
        frames: 分帧后的信号
        n_fft: FFT点数
    Returns:
        每帧的谱熵值
    """
    S = np.abs(np.fft.rfft(frames * np.hanning(frames.shape[1]), n=n_fft, axis=1))
    # 归一化功率谱
    P = S / (np.sum(S, axis=1, keepdims=True) + 1e-8)
    # 计算熵
    entropy = -np.sum(P * np.log(P + 1e-8), axis=1)
    return entropy

def compute_frame_features(y, sr=SR):
    """
    计算所有帧级特征
    Args:
        y: 音频信号
        sr: 采样率
    Returns:
        rms, flux, cent, zcr, entropy: 各种声学特征
    """
    frames = frame_signal(y, sr)
    # RMS (能量特征)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    # spectral centroid (频谱质心)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=512, hop_length=int(FRAME_HOP*sr)).flatten()
    # zcr (零交叉率)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=int(FRAME_LEN*sr), hop_length=int(FRAME_HOP*sr)).flatten()
    # spectral flux (谱流)
    flux = spectral_flux(frames, n_fft=512)
    # spectral entropy (谱熵)
    entropy = spectral_entropy(frames, n_fft=512)
    
    # 确保所有特征具有相同的帧数
    min_frames = min(len(rms), len(flux), len(cent), len(zcr), len(entropy))
    rms = rms[:min_frames]
    flux = flux[:min_frames]
    cent = cent[:min_frames]
    zcr = zcr[:min_frames]
    entropy = entropy[:min_frames]
    
    # normalize each feature to 0-1 by robust scaling
    def rob_norm(x):
        med = np.median(x); iqr = np.percentile(x,75)-np.percentile(x,25)
        if iqr < 1e-6: iqr = np.std(x) + 1e-6
        return np.clip((x - med) / (3*iqr) + 0.5, 0.0, 1.0)
    return rob_norm(rms), rob_norm(flux), rob_norm(cent), rob_norm(zcr), rob_norm(entropy)

def is_dog_bark_candidate(rms, flux, cent, zcr, entropy, sr=SR):
    """
    判断是否为狗吠候选段，减少人声误检
    根据用户观察：狗吠具有短时间上升下降的脉冲特性，峰值较高但持续时间短
    Args:
        rms: 能量特征
        flux: 谱流特征
        cent: 谱质心特征
        zcr: 零交叉率特征
        entropy: 谱熵特征
        sr: 采样率
    Returns:
        bool: 是否为狗吠候选
    """
    # 狗吠特征判断规则
    # 1. 能量不能太低也不能太高（过滤静音和爆音）
    rms_mean = np.mean(rms)
    # 2. 谱流要足够大（脉冲特性）
    flux_mean = np.mean(flux)
    # 3. 谱质心适中（狗吠有高频成分但不是纯高频）
    cent_mean = np.mean(cent)
    # 4. 零交叉率适中（尖锐但不过高）
    zcr_mean = np.mean(zcr)
    # 5. 谱熵较低（脉冲而非噪声）
    entropy_mean = np.mean(entropy)
    
    # 增加对脉冲特性的检测
    # 计算能量的峰值和持续时间
    peak_ratio = np.sum(rms > 0.7) / len(rms)  # 高能量帧占比
    peak_duration = peak_ratio * len(rms) * FRAME_HOP  # 高能量持续时间
    
    # 狗吠判断条件（平衡检测准确率和召回率）
    is_bark = (
        rms_mean > 0.1 and rms_mean < 0.9 and        # 能量适中
        flux_mean > 0.15 and                         # 谱流足够大（更强的脉冲特性）
        cent_mean > 0.3 and cent_mean < 0.9 and      # 谱质心中等偏高
        zcr_mean > 0.2 and zcr_mean < 0.8 and        # 零交叉率适中
        entropy_mean < 0.6 and                       # 谱熵较低（更强的脉冲特性）
        peak_duration < 0.3                          # 高能量持续时间短（脉冲特性）
    )
    
    return is_bark

def detect_bark_segments(y, sr=SR,
                         bp_low=200, bp_high=7500,  # 带通滤波器参数
                         rms_w=0.4, flux_w=0.3, cent_w=0.2, zcr_w=0.1,  # 特征权重
                         up_thresh=0.6, down_thresh=0.4,  # 提高阈值
                         min_dur=0.05, max_dur=0.5,  # 调整持续时间范围，更适合脉冲型狗吠
                         merge_gap=0.05):  # 减小合并间隙
    """
    检测音频中的狗吠片段
    根据用户观察优化：狗吠是短时间上升下降的脉冲信号
    Args:
        y: 音频信号
        sr: 采样率
        bp_low: 带通滤波器低频截止
        bp_high: 带通滤波器高频截止
        rms_w, flux_w, cent_w, zcr_w: 各特征权重
        up_thresh: 双阈值检测的上限（提高到0.6）
        down_thresh: 双阈值检测的下限（提高到0.4）
        min_dur: 最小持续时间（减少到0.05秒）
        max_dur: 最大持续时间（减少到0.5秒，更适合脉冲型狗吠）
        merge_gap: 合并相邻片段的时间间隙（减少到0.05秒）
    Returns:
        list: 检测到的片段列表，每个元素为(起始采样点, 结束采样点)
    """
    print(f"[DEBUG] 开始检测狗吠片段，音频长度: {len(y)/sr:.2f}秒")
    
    # 1. bandpass（带通滤波，去除低频噪声）
    y_bp = bandpass(y, sr, low=bp_low, high=bp_high)
    print(f"[DEBUG] 带通滤波完成")
    
    # 2. frame-level feats（计算帧级特征）
    rms, flux, cent, zcr, entropy = compute_frame_features(y_bp, sr)
    print(f"[DEBUG] 特征计算完成，帧数: {len(rms)}")
    
    # 3. combined score（组合得分，用于检测候选片段）
    # 核心算法：将多个声学特征加权组合，形成一个综合得分
    # RMS（能量）权重最高，因为狗吠是高能量事件
    # Spectral Flux（谱流）权重次之，衡量频谱变化程度
    # Spectral Centroid（谱质心）权重再次之，反映频率分布
    # ZCR（零交叉率）权重最低，辅助判断信号尖锐程度
    score = rms_w*rms + flux_w*flux + cent_w*cent + zcr_w*zcr
    print(f"[DEBUG] 组合得分计算完成，得分范围: [{np.min(score):.3f}, {np.max(score):.3f}]")
    
    # 4. hysteresis thresholding (双阈值检测)
    # 核心算法：使用双阈值避免检测结果抖动
    # 上阈值：触发检测开始
    # 下阈值：维持检测状态直到得分低于下阈值
    voiced = np.zeros_like(score, dtype=bool)
    state = False  # 检测状态标志
    for i,s in enumerate(score):
        if not state:
            # 当前未在检测状态，如果得分超过上阈值则开始检测
            if s >= up_thresh:
                state = True
                voiced[i] = True
        else:
            # 当前在检测状态，只有得分低于下阈值才停止检测
            if s >= down_thresh:
                voiced[i] = True
            else:
                state = False
    print(f"[DEBUG] 双阈值检测完成，检测到{np.sum(voiced)}个帧")
    
    # 5. convert frame indices to time intervals（将帧索引转换为时间区间）
    # 核心算法：将连续的帧转换为连续的时间段
    # fh: 帧移（帧之间的时间间隔）
    # fl: 帧长（每帧的时间长度）
    fh = int(FRAME_HOP*sr)
    fl = int(FRAME_LEN*sr)
    segs = []
    i = 0
    while i < len(voiced):
        if voiced[i]:
            # 找到一个连续的检测段
            j = i
            while j < len(voiced) and voiced[j]:
                j += 1
            # 计算起始和结束时间（采样点）
            start = max(0, i*fh)  # 起始点
            end = min(len(y), (j-1)*fh + fl)  # 结束点
            segs.append((start, end))
            i = j
        else:
            i += 1
    print(f"[DEBUG] 转换为时间区间完成，检测到{len(segs)}个片段")
    
    # 6. merge close segments（合并相近的片段）
    # 核心算法：如果两个片段之间的时间间隔小于merge_gap，则合并为一个片段
    # 这样可以捕获连续的狗吠声
    merged = []
    for s,e in segs:
        if not merged:
            merged.append([s,e])
        else:
            # 如果当前片段起始时间与上一个片段结束时间的间隔小于merge_gap
            if s - merged[-1][1] <= int(merge_gap*sr):
                # 合并两个片段
                merged[-1][1] = e
            else:
                # 保留为独立片段
                merged.append([s,e])
    print(f"[DEBUG] 合并相邻片段完成，合并后{len(merged)}个片段")
    
    # 7. duration filter + dog bark verification（持续时间过滤+狗吠验证）
    # 核心算法：过滤掉持续时间过短或过长的片段，并进行狗吠验证
    final = []
    for s,e in merged:
        # 计算片段持续时间
        dur = (e-s)/sr
        # 持续时间过滤（更适合脉冲型狗吠）
        if dur < min_dur or dur > max_dur:
            print(f"[DEBUG] 片段({s}, {e})持续时间{dur:.3f}秒不符合要求，跳过")
            continue
        # per-seg checks (狗吠验证)
        seg = y_bp[s:e]
        seg_rms, seg_flux, seg_cent, seg_zcr, seg_entropy = compute_frame_features(seg, sr)
        
        # 狗吠验证，减少人声误检
        if is_dog_bark_candidate(seg_rms, seg_flux, seg_cent, seg_zcr, seg_entropy, sr):
            final.append((s,e))
            print(f"[DEBUG] 片段({s}, {e})通过狗吠验证，持续时间{dur:.3f}秒")
        else:
            print(f"[DEBUG] 片段({s}, {e})未通过狗吠验证，持续时间{dur:.3f}秒")
    print(f"[DEBUG] 狗吠验证完成，最终保留{len(final)}个片段")
    return final

# example usage:
if __name__ == "__main__":
    import sys
    import os
    # 确保输出目录存在
    output_dir = "youtube_wav/post_bark_segmentation"
    os.makedirs(output_dir, exist_ok=True)
    
    p = sys.argv[1]  # wav path
    print(f"[INFO] 加载音频文件: {p}")
    y, sr = librosa.load(p, sr=SR)
    print(f"[INFO] 音频加载完成，长度: {len(y)/sr:.2f}秒, 采样率: {sr}Hz")
    
    segs = detect_bark_segments(y, sr)
    print("Detected segments (samp):", segs)
    
    if len(segs) == 0:
        print("[WARN] 未检测到任何狗吠片段，请检查音频或调整参数")
    
    for i,(s,e) in enumerate(segs):
        # 使用soundfile保存音频文件
        output_path = os.path.join(output_dir, f"seg_{i}.wav")
        sf.write(output_path, y[s:e], sr)
        print(f"Saved segment {i} to {output_path}")

"""
python detect_similar_sounds/bark_segmentation.py youtube_wav/bark_segmentation/processed_template_preserving/outdoor_braking_clean_preserving.wav

"""