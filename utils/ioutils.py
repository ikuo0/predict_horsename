import json
import os
import pickle
from dataclasses import asdict, dataclass
from typing import Any, TypeVar

T = TypeVar('T')

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def save_dataclass_to_json(filepath, instance: Any):
    with open(filepath, 'w') as f:
        json.dump(asdict(instance), f, indent=4, ensure_ascii=False)

def load_dataclass_from_json(filepath, cls: type[T]) -> T:
    with open(filepath, 'r') as f:
        data = json.load(f)
    return cls(**data)

def save_pickle(file, obj):
    with open(file, 'wb') as file:
        pickle.dump(obj, file)

def load_pickle(file):
    with open(file, 'rb') as file:
        return pickle.load(file)

