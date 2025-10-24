import os
# os.environ.pop("TF_XLA_FLAGS", None)
# os.environ.pop("CUDA_VISIBLE", None)

import sys
import numpy as np
import tensorflow
from tensorflow import keras

def list_physical_devices(device_name: str):
    devices = tensorflow.config.list_physical_devices(device_name)
    return devices

def list_logical_devices():
    devices = tensorflow.config.list_logical_devices()
    return devices

def main():
    print("TensorFlow version:", tensorflow.__version__)
    # print("Keras version:", keras.__version__)
    print("NumPy version:", np.__version__)
    print("Python version:", sys.version)
    print("Current working directory:", os.getcwd())
    print(f"is_built_with_cuda: {tensorflow.test.is_built_with_cuda()}")

    # Check if GPU is available
    gpus = list_physical_devices("GPU")
    # gpus = list_physical_devices("XLA_GPU")
    print(f"gpus: {gpus}")
    if gpus:
        print(f"GPUs available: {len(gpus)}")
        for gpu in gpus:
            print(f" - {gpu}")
    else:
        print("No GPU available")

    # List logical devices
    logical_gpus = list_logical_devices()
    print(f"logical_gpus: {logical_gpus}")

    # Simple TensorFlow operation to verify functionality
    a = tensorflow.constant([[1, 2], [3, 4]])
    b = tensorflow.constant([[5, 6], [7, 8]])
    c = tensorflow.matmul(a, b)
    print("Result of TensorFlow matmul operation:\n", c.numpy())

if __name__ == "__main__":
    main()

# export PYTHONPATH=/workspaces/pj0006_tensorflow_test
# python tensorflow_status.py
