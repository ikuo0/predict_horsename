import inspect
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import __main__

from utils import utils

MY_DIR = Path(__file__).resolve().parent

OUT_DIR_ROOT = os.path.join(MY_DIR, "out")

def get_execute_identity() -> str:
    """
    実行中のスクリプトのファイルベース名を返す。
    """
    path = getattr(__main__, "__file__", None) or sys.argv[0] or ""
    if not path:
        return "__console__"
    return Path(path).resolve().stem

def get_identity(caller__file__: str) -> str:
    """
    呼び出し元のファイルベース名を返す。
    """
    if caller__file__:
        return Path(caller__file__).resolve().stem
    return "__console__"

class Application:
    def __init__(self, caller__file__: str):
        self.execute_identity = get_execute_identity()
        self.identity = get_identity(caller__file__)
        self.out_dir = os.path.join(OUT_DIR_ROOT, f"{self.identity}_out")
        os.makedirs(self.out_dir, exist_ok=True)
