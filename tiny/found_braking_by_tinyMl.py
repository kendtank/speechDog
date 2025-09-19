import os
import librosa
import numpy as np
import tensorflow as tf
import soundfile as sf
import time
import argparse

# ================== 配置 ==================
TFLITE_MODEL_PATH = "tiny_cnn_bark.tflite"  # TFLite模型路径
INPUT_PATH = r"D:\kend\myPython\speechDog-master\tiny\test_audio"          # 待检测音频文件或文件夹
OUTPUT_DIR = "bark_segments"                # 输出目录
SAMPLE_RATE = 16000
WINDOW_LENGTH = 0.2      # 滑窗长度 (秒)
HOP_LENGTH = 0.1         # 滑窗步长 (秒)
NUM_MFCC = 40
DOG_CLASS_ID = 0         # 狗吠类别索引
THRESHOLD = 0.6          # 判定为狗吠的概率阈值

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================== 加载 TFLite 模型 ==================
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ================== MFCC 提取 ==================
def extract_mfcc(y):
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=NUM_MFCC)
    mfcc = mfcc[np.newaxis, ..., np.newaxis].astype(np.float32)
    return mfcc

# ================== 音频裁剪函数 ==================
def process_audio_file(audio_file, out_dir):
    y, sr = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)
    win_samples = int(WINDOW_LENGTH * sr)
    hop_samples = int(HOP_LENGTH * sr)

    start_time = time.time()
    segments = []
    current_segment = None

    for start in range(0, len(y), hop_samples):
        end = start + win_samples
        if end > len(y):
            segment = np.pad(y[start:], (0, end - len(y)))
        else:
            segment = y[start:end]

        mfcc = extract_mfcc(segment)
        interpreter.set_tensor(input_details[0]['index'], mfcc)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])[0]
        prob = pred[DOG_CLASS_ID]

        if prob >= THRESHOLD:
            if current_segment is None:
                current_segment = [start, end]
            else:
                if start <= current_segment[1]:
                    current_segment[1] = end
                else:
                    segments.append(current_segment)
                    current_segment = [start, end]
        else:
            if current_segment is not None:
                segments.append(current_segment)
                current_segment = None

    if current_segment is not None:
        segments.append(current_segment)

    segment_count = 0
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    for seg in segments:
        seg_start, seg_end = seg
        seg_start_sec = seg_start / sr
        seg_end_sec = min(seg_end, len(y)) / sr
        audio_seg = y[seg_start:seg_end]
        segment_count += 1
        out_path = os.path.join(out_dir, f"{base_name}_bark_{segment_count:03d}_{seg_start_sec:.2f}-{seg_end_sec:.2f}.wav")
        sf.write(out_path, audio_seg, sr)
        print(f"[保存] {out_path} | 长度={seg_end_sec - seg_start_sec:.3f}s")

    end_time = time.time()
    print(f"✅ 完成 {audio_file} | 总裁剪片段数: {segment_count} | 处理时长: {end_time - start_time:.3f}s\n")


# ================== 主函数 ==================
def main():
    if os.path.isfile(INPUT_PATH):
        process_audio_file(INPUT_PATH, OUTPUT_DIR)
    elif os.path.isdir(INPUT_PATH):
        files = [os.path.join(INPUT_PATH, f) for f in os.listdir(INPUT_PATH) if f.lower().endswith((".wav", ".mp3"))]

        files.sort()
        for f in files:
            process_audio_file(f, OUTPUT_DIR)
    else:
        print(f"路径不存在: {INPUT_PATH}")

if __name__ == "__main__":
    main()
