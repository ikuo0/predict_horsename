import sys

from app import Application
from utils import function_report, logutils

app = Application(__file__)
logger = logutils.get_logger(app.out_dir)
# TF_LIGHT_START = True

def measure_tensorflow_import_time(tf_light_start: bool):
    logger.info(f"Measuring TensorFlow import time with light_start={tf_light_start}")
    with function_report.FunctionReport(logger):
        if tf_light_start:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # GPUを完全に不可視化（CUDA初期化をスキップ）
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # ログ抑制（時間短縮は小さいが静かになる）
            os.environ["TF_NUM_INTRAOP_THREADS"] = "1"  # スレッドプール小さく（初期化時間短縮に効くことがある）
            os.environ["TF_NUM_INTEROP_THREADS"] = "1"
        logger.info("Starting TensorFlow import...")
        import tensorflow as tf
        logger.info("TensorFlow import completed.")

def measure_keras_import_time():
    logger.info("Measuring Keras import time")
    with function_report.FunctionReport(logger):
        logger.info("Starting Keras import...")
        import keras
        logger.info("Keras import completed.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tensorflow_import_test.py [tensorflow|tensorflow_light|keras|tensorflow_cpu] [tf_light_start (true/false, optional, default=true for tensorflow_light)]")
        sys.exit(1)
    test_type = sys.argv[1].lower()

    if test_type == "tensorflow":
        measure_tensorflow_import_time(tf_light_start=False)
    elif test_type == "tensorflow_light":
        measure_tensorflow_import_time(tf_light_start=True)
    elif test_type == "keras":
        measure_keras_import_time()
    else:
        print("Invalid test type.")
        sys.exit(1)

# export PYTHONPATH=/workspaces/pj0005_horse_name
# python tensorflow_test/tensorflow_import_test.py tensorflow
# python tensorflow_test/tensorflow_import_test.py tensorflow_light
# python tensorflow_test/tensorflow_import_test.py keras
# python tensorflow_test/tensorflow_import_test.py tensorflow_cpu
