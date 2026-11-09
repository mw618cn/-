#!/usr/bin/env python3
"""把逐时段CSV转换为模型可直接读取的JSON曲线配置。

默认读取24小时、15分钟、96行。模板列包括时间、3个微网PV、6节点
关键/一般负荷以及电价。这里处理数据与单位，不调用任何优化求解器。
输出可直接传给 ``python scripts/run_demo.py --config 输出.json``。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys


NODES = ("T0", "T1", "D0", "M1", "M2", "M3")
MICROGRIDS = ("MG1", "MG2", "MG3")


def read_profiles(path: Path, step_minutes: float, horizon_hours: float) -> dict:
    """读取并严格校验CSV，返回profiles字典，所有值保持输入的物理单位。

    PV及负荷单位MW，价格/损失率单位千元/MWh；time_hour为区间起点。
    每行必须对应一个完整调度区间，不接受跳行、重复或错序时刻。
    """
    # 01 检查时间参数，明确N行是N个区间，不是含末端的N+1状态点。
    dt = step_minutes / 60
    if dt <= 0 or horizon_hours <= 0:
        raise ValueError("步长与时域必须为正")
    count = horizon_hours / dt
    if not math.isclose(count, round(count), abs_tol=1e-9):
        raise ValueError("时域必须能被步长整除")
    n = int(round(count))

    # 02 三个PV列+六节点两类负荷是必须输入；价格与损失率可选。
    pv_columns = [f"pv_{name}" for name in MICROGRIDS]
    critical_columns = [f"critical_{name}" for name in NODES]
    general_columns = [f"general_{name}" for name in NODES]
    required = ["time_hour", *pv_columns, *critical_columns, *general_columns]
    optional = ["grid_price", "critical_voll", "general_voll"]
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        missing = [key for key in required if key not in columns]
        if missing:
            raise ValueError("CSV缺少列：" + ", ".join(missing))
        records = list(reader)
    if len(records) != n:
        raise ValueError(f"需要{n}个区间的数据行，实际{len(records)}行；请勿加入24h末端状态行")

    def number(record: dict, column: str, row_number: int) -> float:
        """数值校验附CSV实际行号，避免空值被静默当作零。"""
        raw = record.get(column, "")
        if raw is None or not str(raw).strip():
            raise ValueError(f"CSV第{row_number}行{column}为空")
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"CSV第{row_number}行{column}不是数值：{raw}") from exc
        if not math.isfinite(value):
            raise ValueError(f"CSV第{row_number}行{column}必须为有限数")
        if value < 0:
            raise ValueError(f"CSV第{row_number}行{column}不得为负；输入单位为MW或千元/MWh")
        return value

    # 03 保证时间从0开始、顺序正确，所有给定曲线都有完整有效数据。
    data = {column: [] for column in required + [c for c in optional if c in columns]}
    for t, record in enumerate(records):
        row_number = t + 2  # CSV首行为表头。
        for column in data:
            data[column].append(number(record, column, row_number))
        if not math.isclose(data["time_hour"][-1], t * dt, abs_tol=1e-8):
            raise ValueError(f"CSV第{row_number}行time_hour应为{t*dt:g}，实际{data['time_hour'][-1]:g}")

    # 04 CSV是“时段×设备”，模型要求“设备×时段”，这里显式转置。
    profiles = {
        "pv_mw": [data[c] for c in pv_columns],
        "load_critical_mw": [data[c] for c in critical_columns],
        "load_general_mw": [data[c] for c in general_columns],
    }
    for column in optional:
        if column in data:
            profiles[column] = data[column]
    return profiles


def main(argv=None) -> int:
    """解析导入参数，复用模型配置校验后写入JSON，不运行MILP。"""
    parser = argparse.ArgumentParser(description="96时段CSV曲线→可直接运行的JSON配置")
    parser.add_argument("input_csv", type=Path, help="UTF-8 CSV，参照configs/profile_template_96.csv")
    parser.add_argument("--output", type=Path, required=True, help="输出JSON配置")
    parser.add_argument("--base-config", type=Path, help="可选：合并已有设备/场景配置")
    parser.add_argument("--step-minutes", type=float, default=15, help="连续调度步长，默认15分钟")
    parser.add_argument("--horizon-hours", type=float, default=24, help="时域长度，默认24小时")
    args = parser.parse_args(argv)

    # 05 和主入口使用同一套配置/曲线校验，防止导入成功但模型读不到数据。
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from tdm_restore.config import load_config, build_profiles
    try:
        profiles = read_profiles(args.input_csv, args.step_minutes, args.horizon_hours)
        cfg = load_config(args.base_config, {
            "horizon_hours": args.horizon_hours,
            "step_minutes": args.step_minutes,
            "profiles": profiles,
        })
        built = build_profiles(cfg)
        # 如果旧版本config忽略profiles，这里会提醒版本错误而不输出假成功。
        import numpy as np
        for key, expected in profiles.items():
            if not np.allclose(np.asarray(built[key]), np.asarray(expected), rtol=0, atol=1e-12):
                raise ValueError(f"配置模块未正确接入{key}覆盖，请使用完整交付代码版本")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))

    # 06 导出完整解析配置，设备参数、情景与曲线同文件归档，便于复现。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cfg, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"已导入{cfg['n_steps']}时段曲线：{args.output.resolve()}（未运行求解器）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
