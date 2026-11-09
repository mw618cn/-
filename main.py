"""仓库入口：直接运行本文件，参数在simple_version/参数数据.xlsx。"""
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    folder = Path(__file__).resolve().parent / "simple_version"
    sys.path.insert(0, str(folder))
    runpy.run_path(str(folder / "main.py"), run_name="__main__")
