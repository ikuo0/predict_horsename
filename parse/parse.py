# import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import json
import os
import sys

from bs4 import BeautifulSoup

DOC_DIR = "/workspaces/pj0005_horse_name/scraping/download_data"
OUT_DIR = "/workspaces/pj0005_horse_name/parse/parse_data"

def euc_to_utf8(bin_text: bytes) -> str:
    """Convert EUC-JP encoded bytes to UTF-8 string."""
    return bin_text.decode("euc_jp", errors="replace").encode("utf-8").decode("utf-8")

def get_anchor_text_or_text(elm) -> str:
    """Get the text from an anchor tag if present, otherwise get the text of the element."""
    a_tag = elm.find("a")
    if a_tag:
        return a_tag.get_text(strip=True)
    return elm.get_text(strip=True)

def parse_html(file_path: str) -> list[dict]:
    """Parse the HTML file and extract horse names and URLs."""
    with open(file_path, "rb") as f:
        bin_text = f.read()
    
    utf8_text = euc_to_utf8(bin_text)
    soup = BeautifulSoup(utf8_text, "html.parser")
    
    # 一番採捕のテーブルを取得
    first_table = soup.find("table")

    # ヘッダを除き、2行目以降全てのデータを取得
    rows = first_table.find_all("tr")[1:]
    # print(rows)
    # 列構成
    # チェックボックス, 馬名, 性, 生年, 厩舎, 父, 母, 母父, 馬主, 生産者, 総賞金(万円)
    # チェックボックスだけ id 属性を取得、他の列はテキストを取得して行列を作成
    data = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 11:
            continue
        horse_id = cols[0].find("input")["id"]
        horse_name = get_anchor_text_or_text(cols[1])
        horse_gender = get_anchor_text_or_text(cols[2])
        horse_birth_year = get_anchor_text_or_text(cols[3])
        links = "" # get_anchor_text_or_text(cols[4]) # 全てリンクなので空にする
        horse_trainer = get_anchor_text_or_text(cols[5])
        horse_father = get_anchor_text_or_text(cols[6])
        horse_mother = get_anchor_text_or_text(cols[7])
        horse_mother_father = get_anchor_text_or_text(cols[8])
        horse_owner = get_anchor_text_or_text(cols[9])
        horse_breeder = get_anchor_text_or_text(cols[10])
        horse_total_prize = get_anchor_text_or_text(cols[11])
        data.append({
            "id": horse_id,
            "name": horse_name,
            "gender": horse_gender,
            "birth_year": horse_birth_year,
            "trainer": horse_trainer,
            "father": horse_father,
            "mother": horse_mother,
            "mother_father": horse_mother_father,
            "owner": horse_owner,
            "breeder": horse_breeder,
            "total_prize": horse_total_prize,
        })
    return data

# マルチスレッドで実行する場合のラッパー
def parse_execute(file_name: str) -> int:
    try:
        data = parse_html(file_name)
        basename = os.path.basename(file_name)
        out_file = os.path.join(OUT_DIR, f"parsed_{basename}.json")
        # 既にファイルが存在していたら処理を終了する
        if os.path.exists(out_file):
            print(f"File {out_file} already exists. Skipping.")
            return 0
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return len(data)
    except Exception as e:
        print(f"Error parsing {file_name}: {e}")
        return 0

def parse_all():
    file_paths = glob.glob(os.path.join(DOC_DIR, "page_*.html"))
    print(f"Found {len(file_paths)} files to parse.")
    # print(file_paths[:10])
    # file_paths = file_paths[:10] # debug

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(parse_execute, fp): fp for fp in file_paths}
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                result = future.result()
                print(f"Parsed {file_path}: {result} records")
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")

def parse_specific(file_path: str):
    result = parse_execute(file_path)
    print(f"Parsed {file_path}: {result} records")

def main():
    parse_all()
    # parse_specific("scraping/download_data/page_00072.html")


if __name__ == "__main__":
    main()

# export PYTHONPATH=/workspaces/pj0005_horse_name
# python parse/parse.py
