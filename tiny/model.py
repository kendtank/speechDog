import os
import glob
import random
import numpy as np
import tensorflow as tf
import librosa
from tensorflow.keras import layers, models

# ================== 配置 ==================
DATA_DIR = r"D:\kend\myPython\speechDog-master\datasets"
SAMPLE_RATE = 16000
WINDOW_LENGTH = 0.2       # 每帧音频长度（秒）
STEP_LENGTH = WINDOW_LENGTH / 2  # 滑窗步长，50%重叠
NUM_MFCC = 40
BATCH_SIZE = 16
EPOCHS = 30
VAL_RATIO = 0.33
SEED = 42
MIN_LEN = 0.1    # 秒
MAX_LEN = 1.0    # 秒
ENERGY_RATIO_THRESHOLD = 0.95  # 300-8000Hz 能量 / 总能量

MODEL_PATH = "tiny_cnn_bark_tf_savedmodel"

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# ================== 工具函数 ==================
def check_energy(y, sr=SAMPLE_RATE):
    """检查主体完整度: 300-8000Hz 能量 / 总能量 ≥ threshold"""
    fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    total_energy = np.sum(np.abs(fft)**2)
    band_energy = np.sum(np.abs(fft[(freqs>=300)&(freqs<=8000)])**2)
    ratio = band_energy / (total_energy + 1e-8)
    return ratio >= ENERGY_RATIO_THRESHOLD

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    orig_len = len(y)/sr
    if orig_len < MIN_LEN:
        print(f"[丢弃音频] {os.path.basename(file_path)} | 时长 {orig_len:.3f}s < {MIN_LEN}s")
        return None
    if orig_len > MAX_LEN:
        y = y[:int(MAX_LEN*sr)]
    return y

def sliding_windows(y, window_len=WINDOW_LENGTH, step_len=STEP_LENGTH, sr=SAMPLE_RATE):
    """生成滑窗音频片段"""
    target_len = int(window_len * sr)
    step = int(step_len * sr)
    segments = []
    for start in range(0, len(y), step):
        end = start + target_len
        seg = y[start:end]
        # 尾部不足，零填充
        if len(seg) < target_len:
            seg = np.pad(seg, (0, target_len - len(seg)))
        # 能量检查
        if check_energy(seg):
            segments.append(seg)
    return segments

def extract_mfcc(y):
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=NUM_MFCC)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]  # (1, n_mfcc, time, 1)
    return mfcc.astype(np.float32)

# ================== 数据集 ==================
all_files = []
class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print("类别:", class_names)

for class_id, class_name in enumerate(class_names):
    class_dir = os.path.join(DATA_DIR, class_name)
    files = glob.glob(os.path.join(class_dir, "**", "*.wav"), recursive=True)
    valid_files = []
    for f in files:
        y = load_audio(f)
        if y is not None:
            segments = sliding_windows(y)
            for seg in segments:
                valid_files.append((seg, class_id))
    print(f"{class_name} 类有效片段: {len(valid_files)}")
    for seg, label in valid_files:
        all_files.append((seg, label))

if len(all_files) == 0:
    raise RuntimeError("没有找到有效音频片段")

random.shuffle(all_files)
val_size = int(len(all_files) * VAL_RATIO)
train_files = all_files[val_size:]
val_files = all_files[:val_size]

print(f"总音频片段数量: {len(all_files)}")
print(f"训练集: {len(train_files)}条, 验证集: {len(val_files)}条")

# ================== 数据生成器 ==================
def data_generator(file_list):
    while True:
        random.shuffle(file_list)
        batch_mfcc = []
        batch_labels = []
        for y, label in file_list:
            mfcc = extract_mfcc(y)
            label_onehot = tf.keras.utils.to_categorical(label, num_classes=len(class_names))
            batch_mfcc.append(mfcc)
            batch_labels.append(label_onehot)
            if len(batch_mfcc) == BATCH_SIZE:
                yield np.vstack(batch_mfcc), np.vstack(batch_labels)
                batch_mfcc = []
                batch_labels = []

# ================== 模型 ==================
sample_input = train_files[0][0]
sample_input = extract_mfcc(sample_input)
input_shape = sample_input.shape[1:]  # (NUM_MFCC, time, 1)

model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(16, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ================== 日志 ==================
class BatchLogger(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        if batch % 10 == 0:
            print(f"[Batch {batch}] loss={logs['loss']:.4f}, acc={logs['accuracy']:.4f}")
raise RuntimeError("请先运行此文件")
# ================== 训练 ==================
train_gen = data_generator(train_files)
val_gen = data_generator(val_files)

steps_per_epoch = max(1, len(train_files) // BATCH_SIZE)
validation_steps = max(1, len(val_files) // BATCH_SIZE)

model.fit(
    train_gen,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    callbacks=[BatchLogger()]
)

# ================== 保存模型 ==================
model.export(MODEL_PATH)
print(f"训练完成，模型已保存为 {MODEL_PATH} （SavedModel 格式）")
