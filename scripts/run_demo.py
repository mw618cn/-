#!/usr/bin/env python3
"""从源码目录启动完整实验，不要求先安装本项目为Python包。

执行逻辑：定位本脚本所在目录→定位项目的src→加入模块搜索路径→调用
正式CLI。参数解析、模型建立、求解、校核和出图全部复用正式入口；这里
不复制一套不同的算例逻辑。可先运行 ``python scripts/run_demo.py --help``。
"""
from __future__ import annotations

from pathlib import Path
import sys


def main() -> None:
    """加入当前代码包src路径并把命令行参数原样交给正式CLI。

    使用脚本绝对路径定位项目，意味着从其他工作目录调用此文件也能
    正确导入代码。输入配置和输出目录的相对路径仍按调用者的当前目录
    解析，便于用户明确控制实验文件保存位置。
    """
    # 01 __file__是本文件的位置，parents[1]即代码包根目录。
    project_root = Path(__file__).resolve().parents[1]
    source_directory = project_root / "src"

    # 02 让Python优先找到当前交付包的代码，避免误用机器上另一旧版本。
    if str(source_directory) not in sys.path:
        sys.path.insert(0, str(source_directory))

    # 03 依赖包仍须通过requirements安装；调用正式入口，保持单一实现。
    from tdm_restore.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
