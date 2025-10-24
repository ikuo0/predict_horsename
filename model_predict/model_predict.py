import os
import random
import sys
import traceback

import numpy as np
import tensorflow
from tensorflow import keras

from futurize import futurize
from model_trainer import model_trainer
from scaling_feature import scaling_feature
from utils import ioutils, logutils, utils


def load_characters_info() -> futurize.CharacterInfo:
    return ioutils.load_dataclass_from_json(os.path.join(futurize.OUT_DIR, "futurize_characters_summary.json"), futurize.CharacterInfo)

def power_smooth(prob: np.ndarray, power: float) -> np.ndarray:
    prob = np.power(prob, power)
    prob /= prob.sum()
    return prob

def weighted_choice_index_from_probs(probs: np.ndarray) -> int:
    total = np.sum(probs)
    # rand_val = random.uniform(0, total)
    rand_val = np.random.uniform(0, total)
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if rand_val < cumulative:
            return i
    return len(probs) - 1  # 万一のため最後のインデックスを返す


def probabilistic_choice_character(char_list: list[str], probas: np.ndarray, top_n: int = 5) -> str:
    sort_indexes = np.argsort(probas)[::-1]  # 降順にソートしたインデックス
    top_indexes = sort_indexes[:top_n]
    top_probs = probas[top_indexes]
    # smooth_top_probs = power_smooth(top_probs, power=2.0)
    # smooth_top_probs = np.abs(smooth_top_probs)
    choice_index = weighted_choice_index_from_probs(top_probs)
    return char_list[top_indexes[choice_index]]

def predict_proba(model: keras.Model, X, batch_size=8192) -> np.ndarray:
    # logits -> softmax で確率に
    logits = model.predict(X, batch_size=batch_size, verbose=0)
    probs = tensorflow.nn.softmax(logits, axis=-1).numpy()
    return probs

def load_model(model_path: str):
    model = keras.models.load_model(model_path)
    return model


def model_predict(model, input_horse_name_predict: str):
    """
    input_horse_name_predict：
      競走馬の名前の最初の３～４文字を受け取り続きを推定する
      最大１２文字で終了、又は EOS が推定で出現したら終了
      次のように推定を進める
      イクノ → イクノ「デ」 と推定
      イクノデ → イクノデ「ィ」 と推定
      クノディ → クノディ「ク」 と推定
      省略
      イクノディクタス として完了
      推定文字と結合、末尾４文字を次の推定材料として使用～　を繰り返す
    """
    # 文字情報読み込み
    characters_info = load_characters_info()
    # print(f"文字情報を読み込みました: {characters_info}")

    # 各種パラメータ
    name_max_size = 12
    window_size = futurize.WINDOW_SIZE
    bos = futurize.BOS
    eos = futurize.EOS
    pre_pads = [bos] * window_size # ["BOS_PAD", "BOS_PAD", "BOS_PAD", "BOS_PAD", ...]
    start_char_list = pre_pads + list(input_horse_name_predict)

    while True:
        name_parts = start_char_list[-window_size:] # 直近の window_size 文字を取得
        feat = futurize.create_feature1(characters_info, name_parts)
        feat = np.array(feat, dtype=np.float32)
        feat = scaling_feature.l2_normalize(feat)
        feat = feat.reshape(1, -1)
        preds = predict_proba(model, feat)
        probs = preds[0]
        smooth_probs = power_smooth(probs, power=2.0)
        next_char = probabilistic_choice_character(characters_info.chars, smooth_probs, top_n=5)
        # print(f"next_char: {next_char}")
        if next_char == eos:
            break
        start_char_list.append(next_char)
        if len(start_char_list) >= name_max_size + window_size:
            break

    horse_name = "".join(start_char_list[window_size:])
    print(f"予測された競走馬の名前: {horse_name}")

def main():
    model_path = os.path.join(model_trainer.OUT_DIR, "model.keras")
    model = load_model(model_path)
    model_predict(model, input_horse_name_predict="イ")
    model_predict(model, input_horse_name_predict="イク")
    model_predict(model, input_horse_name_predict="イクノ")
    model_predict(model, input_horse_name_predict="イクノデ")
    model_predict(model, input_horse_name_predict="トウカイ")
    model_predict(model, input_horse_name_predict="メジロマ")
    model_predict(model, input_horse_name_predict="サクラバ")
    for i in range(20):
        model_predict(model, input_horse_name_predict="メイショウ")
    for i in range(20):
        model_predict(model, input_horse_name_predict="イクノデ")
    for i in range(20):
        model_predict(model, input_horse_name_predict="トウカイ")

if __name__ == "__main__":
    main()

# export PYTHONPATH=/workspaces/pj0005_horse_name
# python model_predict/model_predict.py
