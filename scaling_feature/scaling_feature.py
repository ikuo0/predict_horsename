"""
重み付き文字インデックスの特徴は、０値が多いため全体平均と使ったスケーリングはしない
L2正規化とする
"""

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

from character_summary import character_summary
from enum_horse_name import enum_horse_name
from futurize import futurize
from utils import utils, ioutils, logutils, timeutils

IDENTITY = utils.source_path_identity(__file__)
OUT_DIR = utils.setup_out_dir(__file__)
logger = logutils.get_logger(OUT_DIR)


DB_FILE = futurize.DB_FILE


def connect():
    con = sqlite3.connect(DB_FILE)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    return con

def get_dimension(con: sqlite3.Connection) -> int:
    # ベクターテーブル x_vec の x フィールドのベクターデータの次元を取得
    cur = con.cursor()
    cur.execute("""
        SELECT vec_length(x) FROM x_vec LIMIT 1
    """)
    result = cur.fetchone()
    if result:
        return result[0]
    raise ValueError("Vector dimension not found in x_vec table.")

# L2正規化する
def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / max(norm, 1e-10)

def get_features(con: sqlite3.Connection) -> Tuple[np.ndarray, np.ndarray]:
    # テーブル x_vec(id, x) を全てスキャンして id, x(86次元) の配列を作成する
    # L2正規化も行う
    ids = []
    features = []
    cur = con.cursor()
    cur.execute("SELECT id, x FROM x_vec")
    rows = cur.fetchall()
    for row in rows:
        ids.append(row[0])
        v = np.frombuffer(row[1], dtype=np.float32)
        x = l2_normalize(v)  # L2正規化
        features.append(x)
    return np.array(ids), np.array(features)


def main():
    elogger = logutils.ElapsedLogger("scaling_feature", logger)
    try:
        logger.info("Starting feature scaling process...")
        ioutils.ensure_dir(OUT_DIR)
        con = connect()
        dimension = get_dimension(con)
        logger.info(f"Vector dimension: {dimension}")
        # id, feature を取得、 npz で保存
        ids, features = get_features(con)
        out_file = os.path.join(OUT_DIR, "scaled_features.npz")
        np.savez_compressed(out_file, ids=ids, features=features)
        logger.info(f"Scaled features saved to {out_file}")
    except Exception as e:
        logger.error(f"Error in scaling features: {e}")
        logger.error(traceback.format_exc())
    finally:
        elogger.finish()

if __name__ == "__main__":
    main()

# export PYTHONPATH=/workspaces/pj0005_horse_name
# python scaling_feature/scaling_feature.py
