import numpy as np
import sys
import os
import futurize
# from .futurize import CharacterInfo, create_characters, CHARACTERS_FILE_PATH

SOURCE_DIR = "futurize/futurize_data"

def confirm_feature(characters: np.ndarray, xdata_file: str, ydata_file: str):
    xdata = np.load(xdata_file, allow_pickle=True)
    ydata = np.load(ydata_file, allow_pickle=True)
    size = xdata.shape[0]
    for i in range(size):
        xdata1 = xdata[i]
        ydata1 = ydata[i]
        hot_indexes = np.where(xdata1 > 0)[0]
        # print(f"len(hot_indexes): {len(hot_indexes)}", hot_indexes)
        xvalues = xdata1[hot_indexes]
        sorted_indexes = hot_indexes[np.argsort(-xvalues)]  # 降順
        confirm_string = "".join(characters[sorted_indexes])
        confirm_char = characters[ydata1]
        print(f"{i:03d}: {confirm_string} -> {confirm_char}")
    print(f"Total: {size} entries")
    # xdata1 = xdata[0]
    # print(xdata1)
    # sys.exit()
    # hot_indexes = np.where(xdata1 > 0)
    # xvalues = xdata1[hot_indexes]
    # sorted_indexes = np.argsort(-xvalues)  # 降順
    # confirm_string = "".join(characters[sorted_indexes])
    # print(confirm_string)


def main():
    xdata_file = os.path.join(SOURCE_DIR, "feat_171239_xdata_ソロナメント.npy")
    ydata_file = os.path.join(SOURCE_DIR, "feat_171239_ydata_ソロナメント.npy")
    characters = futurize.create_characters(futurize.CHARACTERS_FILE_PATH)
    info = futurize.CharacterInfo(characters)
    characters = np.array(info.chars)
    confirm_feature(characters, xdata_file, ydata_file)


if __name__ == "__main__":
    main()

# export PYTHONPATH=/workspaces/pj0005_horse_name
# python futurize/data_confirm.py

