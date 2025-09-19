# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/19 下午2:56
@Author  : Kend
@FileName: del_back_and_save_audio.py
@Software: PyCharm
@modifier:
"""


# bark_soft_mask_preserve.py
import os
import numpy as np
import librosa
import soundfile as sf
import scipy.signal as sg

# ---------- 配置 ----------
INPUT_DIR = r"D:\work\datasets\tinyML\bark_16k"
OUTPUT_DIR = r"D:\work\datasets\tinyML\bark_clean_soft"
SR = 16000
LOW = 300
HIGH = 7900
TRANSITION_MS = 5     # 前后保留过渡(ms)
ROLL_HZ = 200         # 边缘平滑宽度 (Hz)
N_FFT = 1024
HOP = 256
PRESERVATION_THRESH = 0.95
FALLBACK_MIN_MASK = 0.35  # 如果第一次相似度不够，第二次用更保守的最小掩码

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- 辅助函数 ----------
def detect_core_region(y, sr, frame_length=512, hop_length=128, energy_ratio=0.2, noise_mult=3.0):
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    if len(rms) == 0:
        return 0, len(y)
    max_r = np.max(rms)
    median_r = np.median(rms)
    thresh = max(median_r * noise_mult, max_r * energy_ratio)
    frames = np.nonzero(rms >= thresh)[0]
    if len(frames) == 0:
        return 0, len(y)
    start_frame = frames[0]
    end_frame = frames[-1]
    start_sample = max(0, librosa.frames_to_samples(start_frame, hop_length=hop_length))
    end_sample = min(len(y), librosa.frames_to_samples(end_frame, hop_length=hop_length) + frame_length)
    trans_samp = int(sr * TRANSITION_MS / 1000)
    start_sample = max(0, start_sample - trans_samp)
    end_sample = min(len(y), end_sample + trans_samp)
    return start_sample, end_sample

def make_soft_mask(freqs, low, high, roll_hz, floor=0.0):
    """
    产生一个对频率平滑衰减的掩码：低于 (low-roll) -> floor; low-roll->low: 从 floor->1 (cosine ramp)
                                               low->high: 1
                                               high->high+roll: 1->floor (cosine)
                                               >high+roll -> floor
    floor: 最低保留系数（0 = 完全衰减；>0 = 只弱衰减）
    """
    mask = np.zeros_like(freqs)
    low_start = max(0.0, low - roll_hz)
    low_end = low
    high_start = high
    high_end = min(freqs.max(), high + roll_hz)

    for i, f in enumerate(freqs):
        if f < low_start:
            mask[i] = floor
        elif f < low_end:
            x = (f - low_start) / (low_end - low_start)
            mask[i] = floor + (1.0 - floor) * 0.5 * (1 - np.cos(np.pi * x))
        elif f <= high_start:
            mask[i] = 1.0
        elif f <= high_end:
            x = (f - high_start) / (high_end - high_start)
            mask[i] = floor + (1.0 - floor) * 0.5 * (1 + np.cos(np.pi * x))
        else:
            mask[i] = floor
    return mask

def spectral_cosine_sim(y1, y2, sr, n_mels=40, n_fft=1024, hop_length=256, fmin=300, fmax=8000):
    s1 = librosa.feature.melspectrogram(y=y1, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                       n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0)
    s2 = librosa.feature.melspectrogram(y=y2, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                       n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0)
    log1 = np.log1p(s1)
    log2 = np.log1p(s2)
    min_t = min(log1.shape[1], log2.shape[1])
    if min_t == 0:
        return 0.0
    a = (log1[:, :min_t] - np.mean(log1)) / (np.std(log1) + 1e-12)
    b = (log2[:, :min_t] - np.mean(log2)) / (np.std(log2) + 1e-12)
    af = a.flatten()
    bf = b.flatten()
    denom = (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12)
    cos = float(np.dot(af, bf) / denom)
    return max(0.0, min(1.0, (cos + 1.0) / 2.0))

def energy_ratio(y_ref, y_proc):
    e_ref = np.sum(y_ref.astype(np.float64)**2) + 1e-12
    e_proc = np.sum(y_proc.astype(np.float64)**2) + 1e-12
    return float(e_proc / e_ref)

# ---------- 处理单文件 ----------
def process_one(file_in, file_out):
    y, sr = librosa.load(file_in, sr=SR, mono=True)
    orig_dur = len(y) / sr

    # 检测主体区间（含 transition）
    start_samp, end_samp = detect_core_region(y, sr)
    y_core = y[start_samp:end_samp]
    if len(y_core) == 0:
        print(f"{os.path.basename(file_in)}: 未检测到主体，跳过")
        return

    # STFT
    D = librosa.stft(y_core, n_fft=N_FFT, hop_length=HOP)
    mag = np.abs(D)
    ph = np.angle(D)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    # 生成掩码（第一次使用 floor=0: 边缘完全弱化）
    mask = make_soft_mask(freqs, LOW, HIGH, ROLL_HZ, floor=0.0)[:, None]  # shape (freq,1)
    mag_masked = mag * mask

    # iSTFT
    D_masked = mag_masked * np.exp(1j*ph)
    y_proc = librosa.istft(D_masked, hop_length=HOP, length=len(y_core))

    # crossfade 前后 transition_ms（线性渐变）
    keep = int(sr * TRANSITION_MS / 1000)
    if keep > 0 and len(y_core) > 2*keep:
        w = np.linspace(1.0, 0.0, keep)
        y_out = y_proc.copy()
        y_out[:keep] = y_core[:keep] * w + y_proc[:keep] * (1.0 - w)
        y_out[-keep:] = y_proc[-keep:] * (1.0 - w[::-1]) + y_core[-keep:] * w[::-1]
    else:
        y_out = y_proc

    # 中央区域（排除过渡）用于评估与能量匹配
    if len(y_core) > 2*keep:
        ref_eval = y_core[keep:-keep]
        proc_eval = y_out[keep:-keep]
    else:
        ref_eval = y_core
        proc_eval = y_out

    spec_sim = spectral_cosine_sim(ref_eval, proc_eval, sr, n_fft=N_FFT, hop_length=HOP)
    e_ratio = energy_ratio(ref_eval, proc_eval)

    # 如果谱相似度不达标，尝试保守模式（边缘最小保留非0）
    if spec_sim < PRESERVATION_THRESH:
        mask2 = make_soft_mask(freqs, LOW, HIGH, ROLL_HZ, floor=FALLBACK_MIN_MASK)[:, None]
        mag_masked2 = mag * mask2
        D_masked2 = mag_masked2 * np.exp(1j*ph)
        y_proc2 = librosa.istft(D_masked2, hop_length=HOP, length=len(y_core))
        # crossfade again
        if keep > 0 and len(y_core) > 2*keep:
            y_out2 = y_proc2.copy()
            w = np.linspace(1.0, 0.0, keep)
            y_out2[:keep] = y_core[:keep] * w + y_proc2[:keep] * (1.0 - w)
            y_out2[-keep:] = y_proc2[-keep:] * (1.0 - w[::-1]) + y_core[-keep:] * w[::-1]
        else:
            y_out2 = y_proc2
        # evaluate
        if len(y_core) > 2*keep:
            proc_eval2 = y_out2[keep:-keep]
        else:
            proc_eval2 = y_out2
        spec_sim2 = spectral_cosine_sim(ref_eval, proc_eval2, sr, n_fft=N_FFT, hop_length=HOP)
        e_ratio2 = energy_ratio(ref_eval, proc_eval2)
        # 如果改进了谱相似度则采纳
        if spec_sim2 > spec_sim:
            y_out = y_out2
            spec_sim = spec_sim2
            e_ratio = e_ratio2

    # 最后做能量匹配（在中央评估区），避免听感变小
    # 计算中央区域 RMS 并缩放整段 y_out 使中央区域能量匹配 ref_eval
    eps = 1e-12
    e_ref = np.sum(ref_eval**2) + eps
    e_proc_final = np.sum((y_out[keep:-keep] if len(y_core)>2*keep else y_out)**2) + eps
    gain = np.sqrt(e_ref / e_proc_final)
    y_out = y_out * gain

    # 保存：把处理后的主体写成文件（注意：我们只保存主体区间）
    sf.write(file_out, y_out, sr, subtype='PCM_16')

    proc_dur = len(y_out) / sr
    status = "OK" if spec_sim >= PRESERVATION_THRESH else "WARN"
    print(f"{os.path.basename(file_in)}: 原始={orig_dur:.3f}s 主体_samples={len(y_core)} proc={proc_dur:.3f}s | "
          f"谱相似度={spec_sim:.4f} 能量比={e_ratio:.4f} => {status}")
    if spec_sim < PRESERVATION_THRESH:
        print(f"  >>> WARN: 建议人工复核 {os.path.basename(file_in)} (谱相似度 {spec_sim:.3f})")

# ---------- 批量执行 ----------
if __name__ == "__main__":
    for fn in sorted(os.listdir(INPUT_DIR)):
        if not fn.lower().endswith(".wav"):
            continue
        inpth = os.path.join(INPUT_DIR, fn)
        outpth = os.path.join(OUTPUT_DIR, fn.replace(".wav", "_core.wav"))
        process_one = globals()['process_one'] = process_one if 'process_one' in globals() else None
        # call processing
        try:
            process_one(inpth, outpth)
        except Exception as e:
            print("处理失败:", fn, e)

