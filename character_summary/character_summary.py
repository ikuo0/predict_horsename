import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from enum_horse_name import enum_horse_name
from parse import parse
from utils import utils, ioutils, logutils, timeutils

IDENTITY = utils.source_path_identity(__file__)
OUT_DIR = utils.setup_out_dir(__file__)
logger = logutils.get_logger(OUT_DIR)

FILE_PREFIX = Path(__file__).stem

_KATAKANA_ONLY = re.compile(r'^[ァ-ヶー]+$')

ENUM_HORSE_NAME_DIR = enum_horse_name.OUT_DIR


############################################################
############################################################
# 競走馬リストから登場文字列の集計を行う
############################################################
@dataclass
class CharactersSummary:
    characters_list: list[str]
    characters_index: list[int]
    characters_dict: dict[str, int]
    character_type_count: int
    total_character_count: int

def save_characters_summary(out_file: str, summary: CharactersSummary):
    ioutils.save_dataclass_to_json(out_file, summary)
    print(f"Saved characters summary to {out_file}")

def load_characters_summary(file_name: str) -> CharactersSummary:
    summary = ioutils.load_dataclass_from_json(file_name, CharactersSummary)
    return summary

def enum_all_characters(out_file: str, source_file: str):
    horse_names = enum_horse_name.load_horse_names(source_file)
    characters = {}
    for name in horse_names.names:
        for char in name.strip():
            if char not in characters:
                characters[char] = 0
            characters[char] += 1

    # 集計データ作成
    characters_list = characters.keys()
    sorted_characters_list = sorted(characters_list)
    total_count = sum(characters.values())

    characters_list = sorted_characters_list
    characters_index = list(range(len(characters_list)))
    characters_dict = {char: count for char, count in zip(characters_list, characters_index)}
    character_type_count = len(characters_list)
    total_character_count = total_count

    summary = CharactersSummary(
        characters_list=characters_list,
        characters_index=characters_index,
        characters_dict=characters_dict,
        character_type_count=character_type_count,
        total_character_count=total_character_count
    )
    save_characters_summary(out_file, summary)
    print(f"Saved all characters to {out_file}")


############################################################
############################################################
# 競走馬リストから登場文字列の集計を行う
############################################################
class Executer:
    def __init__(self, horse_name_file: str, out_dir: str):
        self.horse_name_file = horse_name_file
        self.out_dir = out_dir

    def main(self):
        try:
            # 競走馬名から登場文字の集計と保存
            elogger = logutils.ElapsedLogger("Enum Horse Names", logger)
            os.makedirs(OUT_DIR, exist_ok=True)
            characters_summary_file = os.path.join(OUT_DIR, f"{FILE_PREFIX}_characters_summary.json")
            enum_all_characters(characters_summary_file, self.horse_name_file)
            result_file = os.path.join(OUT_DIR, f"{FILE_PREFIX}_result.json")
            result_data = {
                "start_time": timeutils.to_str(elogger.meas.start_time),
                "elapsed_time": timeutils.elapsed(elogger.meas.start_time)[1].total_seconds(),
                "horse_name_file": self.horse_name_file,
                "characters_summary_file": characters_summary_file,
            }
            ioutils.save_json(result_file, result_data)
        except Exception as e:
            logger.error(f"Error in Executer.main: {e}")
        finally:
            elogger.finish()

def main():
    horse_name_file = os.path.join(ENUM_HORSE_NAME_DIR, "enum_horse_name_horse_names.json")
    executer = Executer(horse_name_file, OUT_DIR)
    executer.main()

if __name__ == "__main__":
    main()

# export PYTHONPATH=/workspaces/pj0005_horse_name
# python character_summary/character_summary.py
