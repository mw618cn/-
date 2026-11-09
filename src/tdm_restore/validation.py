"""解的数值校核与物理复核。

本模块不把“求解器返回了一个数”直接等同于可实施方案。第一层检查全部
线性约束、变量边界及整数性；第二层独立检查储能、受保护负荷、启动供电
和信息时序。该复核是离散时间线性网络模型内的可行性检查，不替代交流潮流、
保护配合、频率暂态及现场黑启动试验。
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _positive_max(values: Any) -> float:
    """返回正违约量的最大值；空约束集合的违约量定义为零。"""
    arr = np.asarray(values, dtype=float)
    return float(max(0.0, np.max(arr))) if arr.size else 0.0


def validate_solution(model: Any, result: dict, cfg: dict, tolerance: float = 2e-5) -> dict:
    """逐项校核一个 MILP/LP incumbent，返回可直接写入 JSON 的审计记录。

    参数 ``result`` 必须是 ``linear.solve`` 的原始输出，而非出图用的汇总。
    ``status`` 允许为时间上限：有合法 incumbent 仍可校核其可行性，但绝不能
    据此宣称已经证明最优。检查中的残差单位随被检查等式而变化，通常为 MW、
    MVAr、MWh 或标幺值，故这里只用容差辨认可行性，不跨物理量比较大小。
    """
    checks: list[dict] = []

    def add(name: str, value: float, note: str = "") -> None:
        """集中生成校核记录，确保每一个失败都能追溯到具体检查。"""
        checks.append({"name": name, "max_violation": float(value),
                       "passed": bool(np.isfinite(value) and value <= tolerance),
                       "note": note})

    raw = result.get("x")
    if raw is None:
        return {"passed": False, "has_incumbent": False,
                "tolerance": tolerance, "checks": [],
                "reason": "求解器未提供可行候选解，不能进行物理校核或生成恢复结论。"}
    x = np.asarray(raw, dtype=float)
    if x.shape != np.asarray(model.c).shape or not np.isfinite(x).all():
        return {"passed": False, "has_incumbent": True,
                "tolerance": tolerance, "checks": [],
                "reason": "候选解长度不匹配或含非有限数值。"}

    # 第一层：覆盖整个模型，防止只检查少量绘图变量而遗漏耦合约束。
    activity = np.asarray(model.A @ x).ravel()
    lower = np.asarray(model.row_lb)
    upper = np.asarray(model.row_ub)
    add("全部约束下界", _positive_max(lower - activity))
    add("全部约束上界", _positive_max(activity - upper))
    add("全部变量下界", _positive_max(np.asarray(model.lb) - x))
    add("全部变量上界", _positive_max(x - np.asarray(model.ub)))
    integer = np.asarray(model.integrality) != 0
    add("离散决策整数性", _positive_max(np.abs(x[integer] - np.rint(x[integer]))))
    expected_obj = float(np.asarray(model.c) @ x + model.objective_constant)
    reported_obj = result.get("objective")
    if reported_obj is not None and np.isfinite(reported_obj):
        obj_tol = tolerance * max(1.0, abs(expected_obj))
        delta = abs(expected_obj - float(reported_obj))
        checks.append({"name": "目标函数与候选解一致", "max_violation": delta,
                       "passed": delta <= obj_tol,
                       "note": f"本项使用相对尺度容差 {obj_tol:.3g}；含停电损失常数项。"})

    ids = model.var_indices
    meta = model.metadata
    profiles = meta.get("profiles", {})
    n = int(cfg.get("n_steps", len(profiles.get("time_hours", []))))
    dt = float(profiles.get("dt_hours", cfg.get("step_minutes", 15) / 60))

    def val(key: str) -> np.ndarray:
        """按变量索引取回形状，避免依赖 CSV 的舍入值进行复核。"""
        return x[np.asarray(ids[key])]

    # 第二层 A：每个微网的跨期能量守恒，不能由单时段可行域代替。
    microgrids = cfg.get("microgrids", [])
    if all(k in ids for k in ("energy", "battery_charge", "battery_discharge")):
        energy, charge, discharge = val("energy"), val("battery_charge"), val("battery_discharge")
        energy_err = []
        for m, mg in enumerate(microgrids):
            ec = mg.get("eta_charge", mg.get("charge_efficiency"))
            ed = mg.get("eta_discharge", mg.get("discharge_efficiency"))
            if ec is not None and ed is not None:
                energy_err.append(energy[:, m, 1:] - energy[:, m, :-1]
                                  - dt * (float(ec) * charge[:, m] - discharge[:, m] / float(ed)))
        if energy_err:
            add("储能逐时能量守恒_MWh", _positive_max(np.abs(energy_err)))
        add("充放电互斥_MW", _positive_max(np.minimum(charge, discharge)),
            "只有充、放电功率同时为正才记违约；模式变量另由整数性和全部约束检查。")
        for m, mg in enumerate(microgrids):
            capacity = mg["energy_mwh"]
            add(f"MG{m+1}初始能量_MWh",
                _positive_max(np.abs(energy[:, m, 0] - capacity * mg["soc_initial"])))
            add(f"MG{m+1}末端储备_MWh",
                _positive_max(capacity * mg["soc_terminal"] - energy[:, m, -1]))

    # 第二层 B：微网关键负荷在每个场景、每个时段均被硬保护。
    critical = profiles.get("load_critical_mw")
    if critical is not None and "served_critical" in ids:
        critical = np.asarray(critical, dtype=float)
        protected = cfg.get("load", {}).get("protected_nodes", [3, 4, 5])
        add("微网受保护关键负荷_MW", _positive_max(np.abs(val("served_critical")[:, protected] - critical[None, protected])))

    # 第二层 C：启动序列只发生一次，完整辅助功率由同一节点平衡供应。
    if "target_start" in ids:
        starts = val("target_start")
        add("目标机组最多启动一次", _positive_max(starts.sum(axis=1) - 1))
        if meta.get("branch") == "start":
            add("当前启动分支约束", _positive_max(np.abs(starts[:, 0] - 1)))
        if meta.get("branch") == "wait":
            add("当前等待分支约束", _positive_max(np.abs(starts[:, 0])))
        online_coeff = meta.get("online_coeff")
        cap_coeff = meta.get("generation_cap_coeff")
        if cap_coeff is not None and "target_power" in ids:
            cap = starts @ np.asarray(cap_coeff).T
            add("机组启动完成前禁止发电及启动后出力上界_MW",
                _positive_max(val("target_power") - cap))
        if online_coeff is not None:
            online = starts @ np.asarray(online_coeff).T
            add("目标机组在线状态不超过一", _positive_max(online - 1))
            if all(k in ids for k in ("target_power", "target_q")) and "s_mva" in cfg.get("generator", {}):
                apparent = np.hypot(val("target_power"), val("target_q"))
                add("目标机组视在功率额定值_MVA", _positive_max(apparent - cfg["generator"]["s_mva"] * online))

    # 第二层 D：只有已修复线路可合闸；恢复开关只增不减。
    event_map = meta.get("switch_index_for_t")
    schedules = meta.get("schedules", [])
    for site in ("A", "B"):
        key = f"switch_{site}"
        if key not in ids:
            continue
        switches = val(key)
        add(f"线路{site}恢复开关单调性", _positive_max(-np.diff(switches, axis=-1)))
        if event_map is not None and schedules:
            hourly_switch = switches[:, np.asarray(event_map, dtype=int)]
            available = np.asarray([s["availability"][site] for s in schedules], dtype=float)
            add(f"线路{site}未修复时不得合闸", _positive_max(hourly_switch - available))

    # 第二层 E：从物理注入重新拼装节点功率守恒，不使用原 A 矩阵行。
    required = ("flow_p", "flow_q", "served_critical", "served_general", "grid_p",
                "grid_q", "target_power", "target_q", "target_start", "dg_power",
                "pv_used", "battery_discharge", "battery_charge", "inverter_q")
    if all(k in ids for k in required):
        fp, fq = val("flow_p"), val("flow_q")
        served = val("served_critical") + val("served_general")
        active_residual = -served.copy()
        pf = cfg["load"]["power_factor"]
        reactive_residual = -served * np.tan(np.arccos(pf))
        aux = val("target_start") @ np.asarray(meta["target_aux_coeff"]).T
        aux_tan = np.tan(np.arccos(cfg["generator"]["aux_power_factor"]))
        active_residual[:, 0] += val("grid_p") + val("target_power") - aux
        reactive_residual[:, 0] += val("grid_q") + val("target_q") - aux * aux_tan
        for m in range(3):
            active_residual[:, m+3] += (val("dg_power")[:, m] + val("pv_used")[:, m]
                                        + val("battery_discharge")[:, m] - val("battery_charge")[:, m])
            reactive_residual[:, m+3] += val("inverter_q")[:, m]
        for edge, line in enumerate(cfg["edges"]):
            fr, to = line["from"], line["to"]
            active_residual[:, fr] -= fp[:, edge]
            active_residual[:, to] += fp[:, edge]
            reactive_residual[:, fr] -= fq[:, edge]
            reactive_residual[:, to] += fq[:, edge]
        add("独立节点有功平衡_MW", _positive_max(np.abs(active_residual)))
        add("独立节点无功平衡_MVAr", _positive_max(np.abs(reactive_residual)))
        if event_map is not None and all(k in ids for k in ("switch_A", "switch_B")):
            za = val("switch_A")[:, np.asarray(event_map, dtype=int)]
            zb = val("switch_B")[:, np.asarray(event_map, dtype=int)]
            statuses = np.ones_like(fp)
            statuses[:, 1], statuses[:, 2], statuses[:, 5] = 1-zb, zb, za
            ratings = np.array([line["s_mva"] for line in cfg["edges"]])
            add("支路圆形视在功率上界_MVA",
                _positive_max(np.hypot(fp, fq) - statuses * ratings[None, :, None]))

    # 第二层 F：相同观测历史不能提前采取不同的调度决策。
    # 全部 NA 行已由第一层核查。这里独立按真实巡检时序重建等价类，交叉检查
    # 主要连续控制；不把“任务跨时段锁定”错误当作信息非预期性。
    information = meta.get("information", "causal")
    if information != "perfect_information" and schedules:
        keys = ["battery_charge", "battery_discharge", "pv_used", "dg_power",
                "served_critical", "served_general", "inverter_q", "target_power",
                "target_q", "grid_p", "grid_q"]
        na_err = 0.0
        for t in range(n):
            groups: dict[str, list[int]] = {}
            for s, schedule in enumerate(schedules):
                history = schedule["observation_history_by_step"][t]
                groups.setdefault(repr(history), []).append(s)
            for group in groups.values():
                if len(group) < 2:
                    continue
                for key in keys:
                    if key not in ids or np.asarray(ids[key]).shape[-1] != n:
                        continue
                    values = val(key)[group, ..., t]
                    na_err = max(na_err, _positive_max(np.abs(values - values[0])))
        add("真实观测历史条件下非预期性", na_err,
            "只比较当前时刻仍无法由完成巡检结果区分的场景。")
        # 模式/合闸/启动指令的决定时刻各自不同；必须按其块起点判定信息。
        discrete_err = 0.0
        for key, slots_key in (("battery_mode", "battery_mode_slots"),
                               ("switch_A", "event_slots"),
                               ("switch_B", "event_slots"),
                               ("target_start", "start_slots")):
            if key not in ids or slots_key not in meta:
                continue
            decisions = val(key)
            for j, t in enumerate(meta[slots_key]):
                groups: dict[str, list[int]] = {}
                for s, schedule in enumerate(schedules):
                    history = schedule["observation_history_by_step"][int(t)]
                    groups.setdefault(repr(history), []).append(s)
                for group in groups.values():
                    if len(group) > 1:
                        values = decisions[group, ..., j]
                        discrete_err = max(discrete_err,
                                           _positive_max(np.abs(values-values[0])))
        add("离散指令决定时刻的非预期性", discrete_err,
            "特别按小时模式块起点检查电池模式，块内稍后获得的信息不能追溯改变该决定。")

    return {"passed": bool(all(c["passed"] for c in checks)),
            "has_incumbent": True, "tolerance": tolerance,
            "checks": checks, "n_variables": int(x.size),
            "n_constraints": int(activity.size),
            "objective_recomputed": expected_obj,
            "interpretation": "通过表示本离散线性恢复模型内可行；不表示交流/频率暂态已验证，也不表示已证明最优。"}
