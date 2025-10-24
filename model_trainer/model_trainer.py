import json
from typing import Dict, Optional
import os
import random
import sys
import traceback
from typing import Tuple

import numpy as np
import tensorflow
from sklearn.metrics import log_loss, top_k_accuracy_score
import keras
from keras import callbacks, layers, mixed_precision, regularizers
from tensorflowjs.converters import save_keras_model

from futurize import futurize
from scaling_feature import scaling_feature
from utils import ioutils, logutils, utils

# GPUを使わない設定
# tensorflow.config.set_visible_devices([], 'GPU')
tensorflow.config.run_functions_eagerly(False)

IDENTITY = utils.source_path_identity(__file__)
OUT_DIR = utils.setup_out_dir(__file__)
logger = logutils.get_logger(OUT_DIR)

def load_ids_and_feature() -> Tuple[np.ndarray, np.ndarray]:
    npz_file = os.path.join(scaling_feature.OUT_DIR, "scaled_features.npz")
    data = np.load(npz_file)
    ids = data['ids']
    features = data['features']
    return ids, features

def load_ids_and_y() -> Tuple[np.ndarray, np.ndarray]:
    # y_values テーブル(vector_id, y)から、vector_id, y の配列を取得する
    con = scaling_feature.connect()
    cur = con.cursor()
    cur.execute("SELECT vector_id, y FROM y_values")
    result = cur.fetchall()
    ids = []
    ys = []
    for r in result:
        ids.append(r[0])
        ys.append(int(r[1]))
    con.close()
    return np.array(ids), np.array(ys)


def vector_to_str(info: futurize.CharacterInfo, vector: np.ndarray) -> str:
    indexes = np.where(vector > 0)[0]
    widxs = np.argsort(-vector[indexes])
    indexes = indexes[widxs]
    result = ""
    for i in indexes:
        char = info.chars[i]
        result += char
    return result

def train(x, y,  hidden=256, max_iter=200, batch_size=1024):
    # モデル定義
    model = keras.Sequential()
    model.add(layers.Input(shape=(x.shape[1],)))
    model.add(layers.Dense(hidden, activation='relu'))
    model.add(layers.Dense(hidden, activation='relu'))
    model.add(layers.Dense(y.shape[1], activation='softmax'))

    # コンパイル
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 学習
    model.fit(x, y, epochs=max_iter, batch_size=batch_size, verbose=2)

    return model

# 各層を調整する等して高速化したバージョン
def train_fast(x, y,  hidden=256, batch_size=1024, max_iter=200):
    mixed_precision.set_global_policy("mixed_float16")
    # データパイプライン
    ds = tensorflow.data.Dataset.from_tensor_slices((x, y)) \
        .shuffle(min(len(x), 10000)) \
        .batch(batch_size) \
        .prefetch(tensorflow.data.AUTOTUNE)

    # モデル定義
    inputs = keras.Input(shape=(x.shape[1],))
    h = layers.Dense(hidden, activation='relu', input_shape=(x.shape[1],),
                     kernel_regularizer=regularizers.l2(0.001))(inputs)
    h = layers.Dense(hidden, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))(h)
    # 出力は fp32 に戻す（softmaxもfp32で）
    outputs = layers.Dense(y.shape[1], activation='softmax', dtype="float32")(h)
    model = keras.Model(inputs, outputs)

    # コンパイル：steps_per_execution で呼び出し削減。XLAはお好みで。
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        steps_per_execution=100,
        jit_compile=True,   # 初回だけ重い場合あり。重すぎたら False に。
    )

    es = callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(ds, epochs=max_iter, verbose=2, callbacks=[es])

    return model

def train_fast_sparse(x, y_int, num_classes, hidden=256, max_iter=200, batch_size=1024):
    mixed_precision.set_global_policy("mixed_float16")

    ds = (tensorflow.data.Dataset.from_tensor_slices((x, y_int))
            .shuffle(min(len(x), 10000))
            .batch(batch_size)
            .prefetch(tensorflow.data.AUTOTUNE))

    # 正則化済みのデータを渡しているが、過学習防止のためにL2正則化を追加

    inputs = keras.Input((x.shape[1],))
    h = layers.Dense(hidden, activation="relu", kernel_regularizer=regularizers.l2(1e-3), name="hidden_1")(inputs)
    h = layers.Dense(hidden, activation="relu", kernel_regularizer=regularizers.l2(1e-3), name="hidden_2")(h)
    logits = layers.Dense(num_classes, dtype="float32", name="logits")(h)  # ← softmax付けない

    model = keras.Model(inputs, logits, name="logistic_regression_model")
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        steps_per_execution=10,     # ← ここは 1 で安定を優先。後で段階的に上げる
        jit_compile=False          # ← まずは False。安定後に True を試す
    )

    es = callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
    model.fit(ds, epochs=max_iter, callbacks=[es], verbose=2)
    return model


def train_softmax_class_weighted(x, y_onehot,  hidden=256, max_iter=200, batch_size=256) -> keras.Model:
    """
    softmax 回帰モデルの学習
    X = 86次元、約400万件、競走馬の名前を4文字幅で窓解析し距離重みをつけた特徴量
    y = 86クラス、次に来る文字のID（one-hotエンコード）
    例: イクノディクタス： イクノデ -> ィ となる、ィ のindexが Y の値
    目的: 競走馬の名前の文字列生成モデルを作成する
      ランダムにそれっぽい競走馬を生成する事が目的
      正解を求める機械学習ではなく、確率テーブルを作成するイメージ
    
    この目的に対しては単層パーセプトロン（softmax回帰モデル）で十分と考える
    多層パーセプトロンを使うと過学習しやすいため、あえて単純なモデルを使用する
    """
    num_features = x.shape[1]
    num_classes  = y_onehot.shape[1]
    logger.info(f"Training softmax regression model: num_features={num_features}, num_classes={num_classes}")

    inp   = layers.Input(shape=(num_features,))
    dense = layers.Dense(num_classes, activation="softmax")
    # dense = output  なので output 層は作らない

    model = keras.Sequential()
    model.add(inp)
    model.add(dense)

    model.compile(optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"])

    early_stopping = callbacks.EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True,
    )

    model.fit(
        x,
        y_onehot,
        epochs=max_iter,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=2,
        shuffle=True,
    )

    return model

def train_softmax_class_weighted_2(x, y_onehot,  hidden=256, max_iter=200, batch_size=256) -> keras.Model:
    # def train_softmax_class_weighted に１層隠れ層を追加したバージョン
    num_features = x.shape[1]
    num_classes  = y_onehot.shape[1]
    logger.info(f"Training softmax regression model with hidden layer: num_features={num_features}, num_classes={num_classes}")
    inp   = layers.Input(shape=(num_features,))
    h1    = layers.Dense(hidden, activation="relu")(inp)
    dense = layers.Dense(num_classes, activation="softmax")(h1)
    model = keras.Model(inputs=inp, outputs=dense)
    model.compile(optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"])
    early_stopping = callbacks.EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True,
    )
    model.fit(
        x,
        y_onehot,
        epochs=max_iter,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=2,
        shuffle=True,
    )
    return model

def evaluate(model, x, y_onehot, k_list=(1,5)):
    # 予測確率（shape: [n_samples, n_classes]）
    P = model.predict(x)
    n_classes = P.shape[1]

    # one-hot → クラスID（shape: [n_samples]）
    y_true = y_onehot.argmax(axis=1)

    # クラス順の整合性（MLPClassifier等は model.classes_ の順に確率が並ぶ）
    labels = getattr(model, "classes_", np.arange(n_classes))

    # ロス
    val_log_loss = log_loss(y_true, P, labels=labels)

    # top-k（kはクラス数を超えないように）
    results = {}
    for k in k_list:
        kk = min(k, n_classes)
        results[f"top{kk}_acc"] = top_k_accuracy_score(y_true, P, k=kk, labels=labels)

    return val_log_loss, results.get("top1_acc", None), results.get("top5_acc", None)


def load_characters_info() -> futurize.CharacterInfo:
    return ioutils.load_dataclass_from_json(os.path.join(futurize.OUT_DIR, "futurize_characters_summary.json"), futurize.CharacterInfo)


def model_train():
    elogger = logutils.ElapsedLogger("model_train", logger)
    try:
        characters_info = load_characters_info()
        logger.info(f"Loaded characters info: {characters_info}")
        ids, x = load_ids_and_feature()
        logger.info(f"Loaded features: {x.shape}")
        _, y_int = load_ids_and_y()

        # debug
        # ids = ids[:100]
        # x = x[:100]
        # y_int = y_int[:100]

        logger.info(f"Loaded labels: {y_int.shape}")
        y_onehot = np.zeros((len(y_int), characters_info.char_count), dtype=np.int32)
        for i in range(len(y_int)):
            y_onehot[i, y_int[i]] = 1

        # debug
        # indexes = np.where(Y == 1)[0]
        # print("indexes:", indexes)
        # sys.exit()

        logger.info(f"Total: {len(y_int)} samples")
        assert len(x) == len(y_int)
        logger.info(f"X.shape: {x.shape}")
        logger.info(f"Y.shape: {y_int.shape}")

        # train
        logger.info(f"start training logistic regression model...")
        # model = train_softmax_class_weighted(x, y_onehot, hidden=512, max_iter=100, batch_size=2048)
        model = train_softmax_class_weighted_2(x, y_onehot, hidden=512, max_iter=100, batch_size=2048)
        logger.info(f"training completed. elapsed: {elogger.elapsed_seconds():.2f} seconds")
        
        # val_log_loss, val_top1_acc, val_top5_acc = evaluate(model, x, y_onehot)
        # logger.info(f"Logistic Regression: log_loss={val_log_loss:.4f}, top1_acc={val_top1_acc:.4f}, top5_acc={val_top5_acc:.4f}")

        # save model
        # logger.info("Saving model pickle")
        # model_file = os.path.join(OUT_DIR, "model.pickle")
        # ioutils.save_pickle(model_file, model)

        model.build((None, x.shape[1]))
        # _ = model(np.zeros((1, x.shape[1]), dtype=np.float32))

        logger.info("Saving model keras")
        model_keras_file = os.path.join(OUT_DIR, "model.keras")
        model.save(model_keras_file)

        logger.info("Saving model 'saved model'")
        export_dir = os.path.join(OUT_DIR, "saved_model")
        os.makedirs(export_dir, exist_ok=True)
        model.export(export_dir, "tf_saved_model") # # ['tf_saved_model', 'onnx', 'openvino']

    except Exception as e:
        logger.error(f"Error in model_train: {e}")
        logger.error(traceback.format_exc())
        raise e
    finally:
        elogger.finish()

def tf_gpu_status():
    phys = tensorflow.config.list_physical_devices("GPU")
    logical = tensorflow.config.list_logical_devices("GPU")
    xla_gpu = tensorflow.config.experimental.list_physical_devices("XLA_GPU")
    built_with_cuda = tensorflow.test.is_built_with_cuda()  # CUDAサポートでビルドされているか
    return {
        "physical_gpus": [d.name for d in phys],
        "logical_gpus": [d.name for d in logical],
        "xla_gpus": [d.name for d in xla_gpu],
        "built_with_cuda": built_with_cuda,
        "tf_version": tensorflow.__version__,
    }

def test1():
    print(tf_gpu_status())
    characters_info = ioutils.load_dataclass_from_json(os.path.join(futurize.OUT_DIR, "futurize_characters_summary.json"), futurize.CharacterInfo)
    _, y = load_ids_and_y()
    y_onehot = np.zeros((len(y), characters_info.char_count), dtype=np.int32)
    for i in range(len(y)):
        y_onehot[i, y[i]] = 1

def main():
    # test1()
    # tf_gpu_status()
    model_train()

if __name__ == "__main__":
    main()

# export PYTHONPATH=/workspaces/pj0005_horse_name
# python model_trainer/model_trainer.py
"""
        tensorflowjs_converter \
        --input_format=tf_saved_model \
        --saved_model_tags=serve \
        model_trainer/model_trainer_out/saved_model \
        model_trainer/model_trainer_out/saved_model_out
"""
