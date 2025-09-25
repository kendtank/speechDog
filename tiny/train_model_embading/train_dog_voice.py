# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/22 下午3:10
@Author  : Kend
@FileName: train_dog_voice.py
@Software: PyCharm
@modifier:
"""


import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from sklearn.model_selection import train_test_split
import datetime


# ===============================
# 数据加载函数
# ===============================
def load_data(feature_dir):
    X, y = [], []
    label_map = {}
    label_id = 0

    for dog_id in sorted(os.listdir(feature_dir)):
        dog_path = os.path.join(feature_dir, dog_id)
        if not os.path.isdir(dog_path):
            continue

        if dog_id not in label_map:
            label_map[dog_id] = label_id
            label_id += 1

        for npy_file in os.listdir(dog_path):
            if npy_file.endswith(".npy"):
                feat = np.load(os.path.join(dog_path, npy_file))
                X.append(feat)
                y.append(label_map[dog_id])

    X = np.array(X)
    y = np.array(y)
    return X, y, label_map


# ===============================
# 模型构建函数
# ===============================
def build_model(input_shape, num_classes, arch="cnn", embedding_dim=128):
    """
    根据 arch 构建模型。输出 embedding 层 + softmax 层
    """
    if arch == "cnn":
        x_in = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, (3, 3), activation="relu")(x_in)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation="relu")(x)
        x = layers.GlobalAveragePooling2D()(x)
        embedding = layers.Dense(embedding_dim, activation=None, name="embedding")(x)

    elif arch == "lstm":
        x_in = layers.Input(shape=input_shape)
        x = layers.Reshape((input_shape[0], input_shape[1]))(x_in)
        x = layers.LSTM(128)(x)
        embedding = layers.Dense(embedding_dim, activation=None, name="embedding")(x)

    elif arch == "cnn_lstm":
        x_in = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, (3, 3), activation="relu")(x_in)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Reshape((-1, 64))(x)
        x = layers.LSTM(128)(x)
        embedding = layers.Dense(embedding_dim, activation=None, name="embedding")(x)

    else:
        raise ValueError("Unsupported architecture")

    # 分类头
    logits = layers.Dense(num_classes, activation="softmax", name="softmax")(embedding)

    return tf.keras.Model(inputs=x_in, outputs=[embedding, logits])


# ===============================
# 自定义 Loss
# ===============================
class CenterLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes, embedding_dim, alpha=0.5):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        # 初始化中心
        self.centers = tf.Variable(
            initial_value=tf.zeros([num_classes, embedding_dim]),
            trainable=False,
            dtype=tf.float32,
            name="centers"
        )

    def call(self, y_true, embeddings):
        y_true = tf.cast(y_true, tf.int32)
        centers_batch = tf.gather(self.centers, y_true)
        diff = embeddings - centers_batch
        # 更新中心
        self.centers.assign_sub(self.alpha * tf.reduce_mean(diff, axis=0))
        return tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=1))




# ===============================
# 训练入口
# ===============================
def main(args):
    # 数据加载
    X, y, label_map = load_data(args.data_dir)
    print(f"Loaded data shape: {X.shape}, labels: {len(label_map)} classes")

    # 拆分数据
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if len(X_train.shape) == 3:  # 补通道
        X_train = np.expand_dims(X_train, -1)
        X_val = np.expand_dims(X_val, -1)

    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y))

    # 构建模型
    model = build_model(input_shape, num_classes, arch=args.arch, embedding_dim=args.embedding_dim)
    model.summary()

    # ===============================
    # 损失函数 & 训练模式
    # ===============================
    if args.loss == "ce":
        loss_fn = losses.SparseCategoricalCrossentropy()
        loss_outputs = ["softmax"]
    elif args.loss == "triplet":
        loss_fn = losses.TripletSemiHardLoss()
        loss_outputs = ["embedding"]
    elif args.loss == "center":
        loss_fn = CenterLoss(num_classes, args.embedding_dim)
        loss_outputs = ["embedding"]
    else:
        raise ValueError("Unsupported loss function")

    # ===============================
    # 日志 & 回调
    # ===============================
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    ckpt_path = os.path.join(args.output_dir, "ckpt")
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_loss")

    # ===============================
    # 编译
    # ===============================
    model.compile(
        optimizer=optimizers.Adam(args.lr),
        loss={k: loss_fn for k in loss_outputs},
        metrics={"softmax": "accuracy"} if args.loss == "ce" else {}
    )

    # ===============================
    # 训练
    # ===============================
    history = model.fit(
        X_train, {"embedding": y_train, "softmax": y_train},
        validation_data=(X_val, {"embedding": y_val, "softmax": y_val}),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[tensorboard_cb, ckpt_cb]
    )

    # 保存原始模型
    saved_model_path = os.path.join(args.output_dir, "saved_model")
    # model.save(saved_model_path)
    model.export(saved_model_path)
    print(f"✅ Saved model at {saved_model_path}")

    # 导出 TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    tflite_model = converter.convert()
    tflite_path = os.path.join(args.output_dir, "model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"✅ Exported TFLite model at {tflite_path}")


# ===============================
# 参数解析
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to feature directory (logmel / mfcc / spectral)")
    parser.add_argument("--arch", type=str, default="cnn",
                        choices=["cnn", "lstm", "cnn_lstm"], help="Model architecture")
    parser.add_argument("--loss", type=str, default="ce",
                        choices=["ce", "triplet", "center"], help="Loss function")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save model & logs & tflite")

    args = parser.parse_args()
    main(args)

"""
python train_dog_voice.py \
  --data_dir ./logmel \
  --arch cnn \
  --loss ce \
  --epochs 30 \
  --output_dir ./exp_logmel_cnn



python train_dog_voice.py \
  --data_dir ./mfcc \
  --arch lstm \
  --loss triplet \
  --embedding_dim 128 \
  --epochs 50 \
  --output_dir ./exp_mfcc_triplet


"""