import os
import __main__


def source_path_identity(full_path: str) -> str:
    filename = os.path.basename(full_path)
    base_name = os.path.splitext(filename)[0]
    return base_name

def setup_out_dir(current_file: str) -> str:
    abs_path = os.path.abspath(current_file)
    base_dir_name = os.path.splitext(abs_path)[0]
    out_dir = f"{base_dir_name}_out"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir
