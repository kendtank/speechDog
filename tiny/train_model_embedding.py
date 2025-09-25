# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/20 下午6:17
@Author  : Kend
@FileName: train_model_embedding.py
@Software: PyCharm
@modifier:
"""


import os
import random
import numpy as np
import tensorflow as tf
import librosa
from tensorflow.keras import layers, models

# ========== 1. 配置 ==========
DATASET_PATH = r"D:\kend\myPython\speechDog-master\datasets\compare_dog"  # 数据集目录，每只狗一个子文件夹 dog01, dog02, ...
SR = 16000                  # 采样率
DURATION = 2.0              # 每段裁剪时长 (秒)
N_MELS = 64                 # Mel频谱维度
EMBED_DIM = 128             # 声纹嵌入维度
BATCH_SIZE = 16
EPOCHS = 20

# ========== 2. 数据处理 ==========
def load_wav(path, sr=SR, duration=DURATION):
    wav, _ = librosa.load(path, sr=sr)
    # 固定长度，短则补零，长则裁剪
    target_len = int(sr * duration)
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    else:
        wav = wav[:target_len]
    return wav

def wav_to_mel(wav, sr=SR, n_mels=N_MELS):
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_mels=n_mels, n_fft=400, hop_length=160
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # 归一化
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db.astype(np.float32)

def build_dataset(dataset_path):
    dataset = []
    dog_ids = sorted(os.listdir(dataset_path))
    for dog_id in dog_ids:
        dog_path = os.path.join(dataset_path, dog_id)
        if not os.path.isdir(dog_path):
            continue
        for wav_file in os.listdir(dog_path):
            if wav_file.endswith(".wav"):
                dataset.append((os.path.join(dog_path, wav_file), dog_id))
    return dataset, dog_ids

all_data, all_dogs = build_dataset(DATASET_PATH)
dog_to_idx = {dog: i for i, dog in enumerate(all_dogs)}

# ========== 3. Triplet 数据生成 ==========
def generate_triplet(batch_size=16):
    while True:
        batch_anchor, batch_pos, batch_neg = [], [], []
        for _ in range(batch_size):
            # 选一只狗做 anchor/positive
            dog = random.choice(all_dogs)
            dog_files = [f for f, d in all_data if d == dog]
            if len(dog_files) < 2:  # 至少2个样本
                continue
            anchor_file, pos_file = random.sample(dog_files, 2)

            # 选一只不同的狗做 negative
            neg_dog = random.choice([d for d in all_dogs if d != dog])
            neg_file = random.choice([f for f, d in all_data if d == neg_dog])

            # 特征提取
            anchor = wav_to_mel(load_wav(anchor_file))
            pos = wav_to_mel(load_wav(pos_file))
            neg = wav_to_mel(load_wav(neg_file))

            batch_anchor.append(anchor)
            batch_pos.append(pos)
            batch_neg.append(neg)

        yield (
            np.expand_dims(np.array(batch_anchor), -1),
            np.expand_dims(np.array(batch_pos), -1),
            np.expand_dims(np.array(batch_neg), -1),
        )

# ========== 4. 模型 ==========
def build_embedding_model(input_shape=(N_MELS, int(SR*DURATION/160), 1)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    x = layers.DepthwiseConv2D(3, activation="relu", padding="same")(x)
    x = layers.Conv2D(32, 1, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(EMBED_DIM)(x)
    outputs = tf.nn.l2_normalize(x, axis=1)  # L2归一化
    return models.Model(inputs, outputs, name="EmbeddingModel")

embedding_model = build_embedding_model()

# Triplet Loss
def triplet_loss(anchor, positive, negative, margin=0.3):
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
    return tf.reduce_mean(loss)

# ========== 5. 训练包装 ==========
class TripletModel(tf.keras.Model):
    def __init__(self, embedding_model):
        super().__init__()
        self.embedding = embedding_model

    def train_step(self, data):
        anchor, positive, negative = data
        with tf.GradientTape() as tape:
            emb_a = self.embedding(anchor, training=True)
            emb_p = self.embedding(positive, training=True)
            emb_n = self.embedding(negative, training=True)
            loss = triplet_loss(emb_a, emb_p, emb_n)
        grads = tape.gradient(loss, self.embedding.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.embedding.trainable_variables))
        return {"loss": loss}

triplet_model = TripletModel(embedding_model)
triplet_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

# ========== 6. 开始训练 ==========
train_gen = generate_triplet(BATCH_SIZE)
triplet_model.fit(train_gen, steps_per_epoch=50, epochs=EPOCHS)

# 保存 TFLite 模型
embedding_model.save("dog_speaker_embedding.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(embedding_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # int8量化
tflite_model = converter.convert()
with open("dog_speaker_embedding.tflite", "wb") as f:
    f.write(tflite_model)

print("训练完成，模型已导出！")
