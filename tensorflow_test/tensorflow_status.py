import os
# os.environ.pop("TF_XLA_FLAGS", None)
# os.environ.pop("CUDA_VISIBLE", None)

import sys
import numpy as np
import tensorflow
from tensorflow import keras as tf_keras
import keras
from app import Application
from utils import logutils
from utils import function_report

app = Application(__file__)
logger = logutils.get_logger(app.out_dir)

def list_physical_devices(device_name: str):
    devices = tensorflow.config.list_physical_devices(device_name)
    return devices

def list_logical_devices():
    devices = tensorflow.config.list_logical_devices()
    return devices

def simple_tensorflow_operation():
    a = tensorflow.constant([[1, 2], [3, 4]])
    b = tensorflow.constant([[5, 6], [7, 8]])
    c = tensorflow.matmul(a, b)
    return c.numpy()

def generate_tensorflow_infos():
    infos = {
        "tensorflow_version": tensorflow.__version__,
        "tf_keras_version": tf_keras.__version__,
        "keras_version": keras.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version,
        "current_working_directory": os.getcwd(),
        "is_built_with_cuda": tensorflow.test.is_built_with_cuda(),
        "GPUS": list_physical_devices("GPU"),
        "XLA_GPUS": list_physical_devices("XLA_GPU"),
        "logical_devices": list_logical_devices(),
        "simple_tensorflow_operation_result": simple_tensorflow_operation(),
    }
    return infos

def main():
    with function_report.FunctionReport(logger):
        tensorflow_infos = generate_tensorflow_infos()
        for key, value in tensorflow_infos.items():
            logger.info(f"{key}: {value}")


if __name__ == "__main__":
    main()

# export PYTHONPATH=/workspaces/pj0005_horse_name
# python tensorflow_test/tensorflow_status.py
