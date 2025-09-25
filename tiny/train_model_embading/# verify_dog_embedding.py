# batch_verify_from_wav.py
import os
import numpy as np
import torch
import librosa
import csv
from model import TinyDogEmbeddingNet
from scipy.spatial.distance import cosine


# ---------------------------
# 配置（请根据你的路径修改）
# ---------------------------
MODEL_PATH = "best_dog_embedding.pth"
DATA_ROOT = r"D:\kend\myPython\speechDog-master\datasets\dog_tiny_verification"
REGISTER_DOG_ID = "dog01"  # 要验证的目标狗ID
TEST_WAV_FOLDER = r"D:\work\datasets\tinyML\bark_origion"  # 待测试的 WAV 文件夹
OUTPUT_CSV = r"verification_scores.csv"

# 预处理参数（必须与训练时一致！）
SAMPLE_RATE = 16000
TARGET_DURATION = 0.4
N_MELS = 32
N_FFT = 400
HOP_LENGTH = 200
TARGET_TIME_FRAMES = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# 复用你的预处理逻辑（WAV → log-mel）
# ---------------------------
def wav_to_logmel_fixed(wav_path):
    """将 WAV 文件转为 32x32 log-mel，与训练预处理完全一致"""
    # 1. 加载并以能量峰值为中心裁剪
    y, orig_sr = librosa.load(wav_path, sr=None, mono=True)
    if orig_sr != SAMPLE_RATE:
        y = librosa.resample(y=y, orig_sr=orig_sr, target_sr=SAMPLE_RATE)

    target_samples = int(TARGET_DURATION * SAMPLE_RATE)
    if len(y) < target_samples:
        pad_len = target_samples - len(y)
        left = pad_len // 2
        right = pad_len - left
        y = np.pad(y, (left, right), mode="constant")
    else:
        rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
        if len(rms) == 0:
            center = len(y) // 2
        else:
            max_frame = np.argmax(rms)
            center = int(max_frame * HOP_LENGTH + N_FFT // 2)
        start = max(0, center - target_samples // 2)
        end = start + target_samples
        if end > len(y):
            end = len(y)
            start = max(0, end - target_samples)
        y = y[start:end]
    y = y[:target_samples]

    # 2. 提取 log-mel 并 resize 到 32x32
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

    if mel_db.shape[1] > TARGET_TIME_FRAMES:
        mel_db = mel_db[:, :TARGET_TIME_FRAMES]
    else:
        mel_db = librosa.util.fix_length(mel_db, size=TARGET_TIME_FRAMES, axis=1)

    return mel_db.astype(np.float32)


# ---------------------------
# 模型 & 注册库
# ---------------------------
def load_model():
    model = TinyDogEmbeddingNet(embedding_dim=16)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def build_register_db(model, dog_id):
    """从 DATA_ROOT/dog_id 中加载所有 *_orig.npy 作为注册库"""
    dog_dir = os.path.join(DATA_ROOT, dog_id)
    embeddings = []
    for f in os.listdir(dog_dir):
        if f.endswith('_orig.npy'):
            mel = np.load(os.path.join(dog_dir, f)).astype(np.float32)
            mel = np.expand_dims(mel, axis=(0, 1))
            with torch.no_grad():
                emb = model(torch.tensor(mel, device=DEVICE))
            embeddings.append(emb.cpu().numpy().flatten())

    if not embeddings:
        raise ValueError(f"未找到 {dog_id} 的注册样本 (*_orig.npy)")

    # 使用平均 embedding 作为注册模板（更鲁棒）
    template = np.mean(embeddings, axis=0)
    template = template / np.linalg.norm(template)
    print(f"✅ 注册模板: {dog_id} (基于 {len(embeddings)} 个原始样本)")
    return template


# ---------------------------
# 批量测试 WAV 文件夹
# ---------------------------
def batch_verify(model, register_template, wav_folder, output_csv):
    results = []
    wav_files = [f for f in os.listdir(wav_folder) if f.lower().endswith('.wav')]

    if not wav_files:
        print(f"⚠️  {wav_folder} 中没有 WAV 文件")
        return

    print(f"🔍 批量测试 {len(wav_files)} 个 WAV 文件...")
    for fname in sorted(wav_files):
        wav_path = os.path.join(wav_folder, fname)
        try:
            # WAV → log-mel
            mel = wav_to_logmel_fixed(wav_path)
            mel = np.expand_dims(mel, axis=(0, 1))

            # 提取 embedding
            with torch.no_grad():
                emb = model(torch.tensor(mel, device=DEVICE))
            emb = emb.cpu().numpy().flatten()
            emb = emb / np.linalg.norm(emb)

            # 计算相似度
            similarity = 1 - cosine(emb, register_template)
            results.append((fname, similarity))
            print(f"{fname}: {similarity:.4f}")

        except Exception as e:
            print(f"❌ {fname} 处理失败: {e}")
            results.append((fname, -1.0))

    # 保存 CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'similarity_to_' + REGISTER_DOG_ID])
        for fname, score in results:
            writer.writerow([fname, f"{score:.4f}"])

    print(f"\n✅ 结果已保存至: {output_csv}")
    return results


# ---------------------------
# 主流程
# ---------------------------
if __name__ == "__main__":
    # 1. 加载模型
    model = load_model()

    # 2. 构建注册模板（基于 dog_01 的原始样本）
    register_template = build_register_db(model, REGISTER_DOG_ID)

    # 3. 批量验证 WAV 文件夹
    batch_verify(model, register_template, TEST_WAV_FOLDER, OUTPUT_CSV)