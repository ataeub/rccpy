import sys
import os
# from pathlib import Path


def set_ccpy_paths(cc_path):
    sys.path.insert(0, os.path.join(
        cc_path, "CloudCompare"))
    sys.path.insert(0, os.path.join(
        cc_path, "doc/PythonAPI_test"))
    sys.path.insert(0, os.path.join(cc_path, "ccViewer"))
    sys.path.insert(0, os.path.join(
        cc_path, "CloudCompare/plugins"))


def get_sys_paths():
    return sys.path


def reset_paths(original_path):
    sys.path = original_path.copy()
