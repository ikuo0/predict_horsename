import glob
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from app import Application
from parse import parse
from utils import ioutils, logutils, timeutils, utils

app = Application(__file__)
logger = logutils.get_logger(app.out_dir)


FILE_PREFIX = Path(__file__).stem

_KATAKANA_ONLY = re.compile(r'^[ァ-ヶー]+$')

PARSE_DIR = "/workspaces/pj0005_horse_name/out/parse_out"


############################################################
############################################################
# JSONから競走馬の名前列挙
############################################################
@dataclass
class HorseNames:
    names: list[str]


def save_horse_names(out_file: str, names: list[str]):
    horse_names = HorseNames(
        names=names
    )
    ioutils.save_dataclass_to_json(out_file, horse_names)
    print(f"Saved {len(names)} names to {out_file}")


def load_horse_names(file_name: str) -> HorseNames:
    horse_names = ioutils.load_dataclass_from_json(file_name, HorseNames)
    return horse_names


def enum_horse_data_files(data_dir: str) -> list[str]:
    data_dir = os.path.abspath(data_dir)
    file_paths = glob.glob(os.path.join(data_dir, "parsed_page_*.json"))
    file_paths.sort()
    return file_paths


def is_katakana_name(name: str) -> bool:
    return bool(_KATAKANA_ONLY.match(name))


def get_names_from_json_file(file_name: str) -> list[str]:
    names = []
    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)
        for record in data:
            name = record.get("name")
            if name and is_katakana_name(name):
                names.append(name)
    return names

# マルチスレッド実行用の実行関数
def execute_get_names_from_json_file(file_name: str) -> list[str]:
    try:
        names = get_names_from_json_file(file_name)
        return names
    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        return []

def extract_names(out_file: str, file_paths: list[str]):
    all_names = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(execute_get_names_from_json_file, fp): fp for fp in file_paths}
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                names = future.result()
                all_names.extend(names)
                print(f"Extracted {len(names)} names from {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    unique_names = sorted(set(all_names))
    print(f"Total unique names: {len(unique_names)}")
    # Save to a text file
    save_horse_names(out_file, unique_names)
    print(f"Saved names to {out_file}")


############################################################
############################################################
# 競走馬リストから登場文字列の集計を行う
############################################################
class Executer:
    def __init__(self, data_dir: str, out_dir: str):
        self.data_dir = data_dir
        self.out_dir = out_dir

    def main(self):
        try:
            elogger = logutils.ElapsedLogger("Enum Horse Names", logger)
            os.makedirs(self.out_dir, exist_ok=True)
            files = enum_horse_data_files(PARSE_DIR)
            # files = files[:10] # debug
            # print(files)
            print(f"Found {len(files)} files to process.")
            # 競走馬名の列挙と保存
            horse_name_file = os.path.join(OUT_DIR, f"{FILE_PREFIX}_horse_names.json")
            extract_names(horse_name_file, files)
            result_file = os.path.join(OUT_DIR, f"{FILE_PREFIX}_result.json")
            result_data = {
                "start_time": timeutils.to_str(elogger.meas.start_time),
                "elapsed_time": timeutils.elapsed(elogger.meas.start_time)[1].total_seconds(),
                "total_files": len(files),
                "horse_name_file": horse_name_file,
            }
            ioutils.save_json(result_file, result_data)
        except Exception as e:
            logger.error(f"Error in Executer.main: {e}")
        finally:
            elogger.finish()

def main():
    executer = Executer(PARSE_DIR, OUT_DIR)
    executer.main()

if __name__ == "__main__":
    main()

# export PYTHONPATH=/workspaces/pj0005_horse_name
# python enum_horse_name/enum_horse_name.py
