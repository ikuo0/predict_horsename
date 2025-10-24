
import json
import os
import sqlite3
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import sqlite_vec

from scaling_feature import scaling_feature
from character_summary import character_summary
from enum_horse_name import enum_horse_name
from futurize import futurize
from character_summary import character_summary
from utils import ioutils, logutils, timeutils

HORSE_NAMES_DOC = """
イクノアクティブ
イクノアメンボ
イクノイッシン
イクノエイトナイン
イクノエース
イクノオリンピア
イクノオンザルース
イクノガキタゾ
イクノギャルソンヌ
イクノグレイス
イクノコウトクテン
イクノゴールド
イクノサファイヤ
イクノシャトル
イクノシンプウ
イクノシード
イクノジャケット
イクノスイシン
イクノスカイ
イクノスズカ
イクノタキシード
イクノダイリキ
イクノダンス
イクノチャン
イクノディクタス
イクノトップクイン
イクノナデシコ
イクノナンプウ
イクノノココロ
イクノハレスガタ
イクノパートナー
イクノヒコボシ
イクノファイト
イクノフラッシュ
イクノブライト
イクノブランド
イクノブロート
イクノホープアイ
イクノポイント
イクノマゴムスメ
イクノマッハ
イクノミライ
イクノミラクル
イクノムード
イクノメモライズ
イクノランボー
イクノリバー
イクノリージェント
イクノレガート
イクノレーヴ
イクノローナ
イクノローマン
"""

HORSE_NAMES = [name for name in HORSE_NAMES_DOC.split("\n") if name]

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


def get_all_vector_id_from_horse_name(horse_name: str) -> list[int]:
    # labels テーブルから horse_name に対応する id を取得する
    con = scaling_feature.connect()
    cur = con.cursor()
    cur.execute("SELECT vector_id FROM labels WHERE horse_name = ?", (horse_name,))
    result = cur.fetchall()
    con.close()
    return [r[0] for r in result]

def get_y_from_vector_id_list(vector_id: list[int]) -> list[int]:
    # y_values テーブルから vector_id に対応する y を取得する
    con = scaling_feature.connect()
    cur = con.cursor()
    y_list = []
    for vid in vector_id:
        cur.execute("SELECT y FROM y_values WHERE vector_id = ?", (vid,))
        result = cur.fetchone()
        if result:
            y_list.append(int(result[0]))
    con.close()
    assert len(y_list) == len(vector_id)
    return y_list

def get_x_from_vector_id_list(vector_id: list[int]) -> list[np.ndarray]:
    # x_vec テーブルから vector_id に対応する x を取得する
    con = scaling_feature.connect()
    cur = con.cursor()
    x_list = []
    for vid in vector_id:
        cur.execute("SELECT x FROM x_vec WHERE id = ?", (vid,))
        result = cur.fetchone()
        if result:
            v = np.frombuffer(result[0], dtype=np.float32)
            x_list.append(v)
    con.close()
    assert len(x_list) == len(vector_id)
    return x_list

def vector_to_str(info: futurize.CharacterInfo, vector: np.ndarray) -> str:
    indexes = np.where(vector > 0)[0]
    widxs = np.argsort(-vector[indexes])
    indexes = indexes[widxs]
    result = ""
    for i in indexes:
        char = info.chars[i]
        result += char
    return result
        
    

def main():
    characters_info = ioutils.load_dataclass_from_json(os.path.join(futurize.OUT_DIR, "futurize_characters_summary.json"), futurize.CharacterInfo)
    target_vector_id_list = get_all_vector_id_from_horse_name("イクノディクタス")
    x = get_x_from_vector_id_list(target_vector_id_list)
    y = get_y_from_vector_id_list(target_vector_id_list)
    print(f"ids: {target_vector_id_list}")

    for i, ix in enumerate(x):
        v = ix
        s = vector_to_str(characters_info, v)
        yidx = y[i]
        print(f"feature[{i}]: {s}, next char: {characters_info.chars[yidx]}")


if __name__ == "__main__":
    main()

# export PYTHONPATH=/workspaces/pj0005_horse_name
# python scaling_feature/data_confirm.py
