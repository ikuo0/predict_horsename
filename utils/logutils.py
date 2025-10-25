import logging
import sys
from pathlib import Path

from utils import timeutils

"""
out_dir: 出力ディレクトリ

description:
    out_dir/log.txt に全てのレベルのログを出力する
    out_dir/error.txt に warning 以上のログを出力する（※warning 以上は log.txt にも error.txt にも両方に出力される）
    標準出力にも全てのレベルのログが出力される
"""
def get_logger(out_dir: str) -> logging.Logger:
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)

    # 設定済みであればなにも処理をせず終了
    if logger.hasHandlers():
        return logger

    # フォーマッタの設定
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # ファイルハンドラ（log.txt） - 全てのレベルのログを出力
    file_handler_all = logging.FileHandler(out_dir_path / 'log.txt', mode='a', encoding='utf-8')
    file_handler_all.setLevel(logging.DEBUG)
    file_handler_all.setFormatter(formatter)
    logger.addHandler(file_handler_all)

    # ファイルハンドラ（error.txt） - warning 以上のログを出力
    file_handler_error = logging.FileHandler(out_dir_path / 'error.txt', mode='a', encoding='utf-8')
    file_handler_error.setLevel(logging.WARNING)
    file_handler_error.setFormatter(formatter)
    logger.addHandler(file_handler_error)

    # コンソールハンドラ - 全てのレベルのログを標準出力に出力
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


class ElapsedLogger:
    def __init__(self, process_name: str, logger: logging.Logger):
        self.meas = timeutils.TimeMeasurer()
        self.process_name = process_name
        self.logger = logger
        logger.info(f"Start {self.process_name} at {timeutils.to_str(self.meas.start_time)}")

    def finish(self):
        end_time, elapsed_time = self.meas.finish()
        self.logger.info(f"Finish {self.process_name} at {timeutils.to_str(end_time)}")
        self.logger.info(f"Elapsed time for {self.process_name}: {elapsed_time}")

    def elapsed_seconds(self) -> float:
        end_time, elapsed_time = timeutils.elapsed(self.meas.start_time)
        return elapsed_time.total_seconds()
