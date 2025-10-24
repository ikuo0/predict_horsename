import json
import os
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import sqlite_vec

from character_summary import character_summary
from enum_horse_name import enum_horse_name
from utils import utils, ioutils, logutils, timeutils

IDENTITY = utils.source_path_identity(__file__)
OUT_DIR = utils.setup_out_dir(__file__)
logger = logutils.get_logger(OUT_DIR)


HORSE_NAME_FILE_PATH = os.path.join(enum_horse_name.OUT_DIR, "enum_horse_name_horse_names.json")
CHARACTER_SUMMARY_PATH = os.path.join(character_summary.OUT_DIR, "character_summary_characters_summary.json")
DB_FILE = os.path.join(OUT_DIR, "futurize.db")

BOS = "BOS_PAD"
EOS = "EOS_PAD"

WINDOW_SIZE = 4

@dataclass
class DBInfo:
    db_file: str
    dimension: int

class FeatureDB:
    def __init__(self, info: DBInfo):
        self.info = info

    def create_tables(self, con: sqlite3.Connection):
        cur = con.cursor()

        # ラベル：vector_id がPK、horse_idは“複数の特徴の元”を表すID（同じhorse_idに複数vector_idが紐づく）
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS labels(
            vector_id   INTEGER PRIMARY KEY
                        REFERENCES vector_meta(vector_id) ON DELETE CASCADE,
            horse_id    INTEGER NOT NULL,
            horse_name  TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_labels_horse_name
            ON labels(horse_name);
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_labels_horse ON labels(horse_id);")
        # Y：vector_id がPK（1特徴=1Y）
        cur.execute("""
        CREATE TABLE IF NOT EXISTS y_values(
            vector_id INTEGER PRIMARY KEY
                    REFERENCES vector_meta(vector_id) ON DELETE CASCADE,
            y REAL NOT NULL
        )
        """)
        # X：ベクトル本体（vec0 仮想表）… 主キーは1列のみ（vector_id）
        cur.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS x_vec
        USING vec0(
            id INTEGER PRIMARY KEY,               -- = vector_id
            x  float[{self.info.dimension}] distance_metric=cosine
        )
        """)
        con.commit()

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.info.db_file)
        con.enable_load_extension(True)
        sqlite_vec.load(con)
        return con

    def insert_feature(self, con: sqlite3.Connection, vector_id: int, horse_id: int, horse_name: str, x: list[float], y: float, commit=True):
        cur = con.cursor()
        x_array = np.array(x, dtype=np.float32)
        # vector_meta に空レコードを挿入して vector_id を発番
        # labels にレコードを挿入
        cur.execute("INSERT INTO labels (vector_id, horse_id, horse_name) VALUES (?, ?, ?) ON CONFLICT(vector_id) DO NOTHING;", (vector_id, horse_id, horse_name))
        # y_values にレコードを挿入
        cur.execute("INSERT INTO y_values (vector_id, y) VALUES (?, ?) ON CONFLICT(vector_id) DO NOTHING;", (vector_id, y))
        # x_vec にレコードを挿入
        try:
            cur.execute("INSERT OR IGNORE INTO x_vec (id, x) VALUES (?, ?);", (vector_id, x_array))
        except sqlite3.IntegrityError as e:
            logger.error(f"IntegrityError inserting x_vec for vector_id {vector_id}: {e}")
            pass
        except sqlite3.Error as e:
            logger.error(f"Error inserting x_vec for vector_id {vector_id}: {e}")
            pass
        # 上記raiseは仮想テーブルへの上書きやIGNOREで起こる、エラー発生＝２度目以降の実行であることが多いので一旦無視
        # なんか面倒になったらDB削除してやり直してください
        if commit:
            con.commit()

@dataclass
class CharacterInfo:
    chars: list[str]
    char_count: int

    def index(self, char: str) -> int:
        return self.chars.index(char)
    # 見つからない場合はエラーで良い、データが間違っているということなのでデータ作成からやり直す

def create_empty_vector(info: CharacterInfo) -> list[float]:
    return [0] * info.char_count

def create_character_info(filename: str) -> CharacterInfo:
    summary = ioutils.load_dataclass_from_json(filename, character_summary.CharactersSummary)
    characters = [BOS, EOS]
    for c in summary.characters_list:
        characters.append(c)
    info = CharacterInfo(characters, len(characters))
    return info

def create_feature1(info: CharacterInfo, name_part: list[str]) -> list[float]:
    size = len(name_part)
    vector = create_empty_vector(info)
    for i, c in enumerate(name_part):
        index = info.index(c)
        # weight = 1 / (i + 1) # 距離重み、近いほど大きい
        weight = 1 / (size - i) # 遠いほど大きい
        vector[index] += weight
    return vector

def create_feature(info: CharacterInfo, name: str) -> Tuple[int, list[list[float]], list[float]]:
    name_array = [BOS] * WINDOW_SIZE + list(name) + [EOS] * (WINDOW_SIZE - 1)
    size = WINDOW_SIZE + len(name)
    xdata = []
    ydata = []

    for i in range(size - 3):
        tail = i + WINDOW_SIZE
        x = name_array[i:tail]
        y = info.index(name_array[tail])
        feat = create_feature1(info, x)
        xdata.append(feat)
        ydata.append(y)
    data_size = len(xdata)
    return data_size, xdata, ydata

def test1():
    name = "イクノディクタス"
    info = create_character_info(CHARACTER_SUMMARY_PATH)
    data_size, xdata, ydata = create_feature(info, name)
    print(f"name: {name}")
    for ix in range(len(xdata)):
        x = xdata[ix]
        y = ydata[ix]
        print(f"x[{ix}]: {x}")
        print(f"y[{ix}]: {y} ({info.chars[y]})")

def create_horse_id_list(horse_names: list[str]) -> list[int]:
    id_list = []
    for i, name in enumerate(horse_names):
        id_list.append(i)
    return id_list

def execute_all():
    info = create_character_info(CHARACTER_SUMMARY_PATH)
    ioutils.save_dataclass_to_json(os.path.join(OUT_DIR, "futurize_characters_summary.json"), info)
    horse_names = enum_horse_name.load_horse_names(HORSE_NAME_FILE_PATH).names
    # # debug, 最初の10件だけ
    # horse_names = horse_names[:10]

    horse_id_list = create_horse_id_list(horse_names)
    os.makedirs(OUT_DIR, exist_ok=True)

    db_info = DBInfo(
        db_file=DB_FILE,
        dimension=info.char_count
    )
    feature_db = FeatureDB(db_info)
    conn = feature_db.connect()
    feature_db.create_tables(conn)
    con = feature_db.connect()

    commit_span = 1000
    vector_id = 0
    con.execute("BEGIN;")
    for horse_id, horse_name in zip(horse_id_list, horse_names):
        print(f"{horse_id}: {horse_name}")
        data_size, xdata_list, ydata_list = create_feature(info, horse_name)
        for i in range(data_size):
            feature_db.insert_feature(
                con,
                vector_id=vector_id,
                horse_id=horse_id,
                horse_name=horse_name,
                x=xdata_list[i],
                y=ydata_list[i],
                commit=False
            )
            vector_id += 1
        if horse_id % commit_span == 0:
            con.commit()
    con.commit()
    con.close()


def main():
    elogger = logutils.ElapsedLogger("futurize", logger)
    try:
        # test1()
        # sys.exit()
        execute_all()
    except Exception as e:
        logger.error(f"Error in futurize: {e}")
    finally:
        elogger.finish()


if __name__ == "__main__":
    main()

# export PYTHONPATH=/workspaces/pj0005_horse_name
# python futurize/futurize.py
