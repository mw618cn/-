"""结果分析与计算流程：取解、可行性校核、能力域、D值和策略比较。

main.py调用run_experiment。该函数循环调用抢修模型和恢复模型，再输出
24h结果、CSV和图件。三个创新点对应的计算都在这里调用，共用同一模型。
"""
from __future__ import annotations
from data import build_profiles
from repair_model import enumerate_policies
from power_model import build_model, solve, freeze_discrete, solve_lp
from plots import plot_all, _set_style, _save, _stairs

# ==================== 一、提取调度结果与韧性指标 ====================
import numpy as np


def extract_solution(model, result, cfg):
    """提取可行解；无incumbent时只返回状态，绝不编造供电曲线。"""
    md=model.metadata;profiles=md.get("profiles",build_profiles(cfg))
    scenarios=md["scenarios"];probs=np.array([s["probability"] for s in scenarios]);n=cfg["n_steps"]
    dt=cfg["step_minutes"]/60;idx=model.var_indices
    solver={k:v for k,v in result.items() if k!="x"}
    policy=md["policy"]
    out={"policy_id":policy["id"],"policy_label":policy["label"],"branch":md.get("branch","free"),
         "information":md.get("information","causal"),"horizon_hours":cfg["horizon_hours"],"dt_hours":dt,
         "time_hours":np.asarray(profiles["time_hours"]).tolist(),"scenario_names":[s["name"] for s in scenarios],
         "scenario_ids":[s["id"] for s in scenarios],"probabilities":probs.tolist(),"solver":solver,
         "schedules":md["schedules"],"summary":{},"series":{},"cost_components":{},"diagnostics":{},
         "parameter_digest":{"alpha":cfg["alpha"],"soc_initial":[m["soc_initial"] for m in cfg["microgrids"]],
          "capacity_mwh":[m["energy_mwh"] for m in cfg["microgrids"]],"candidate_policies":8,
          "network_model":"lossless LinDistFlow + inscribed apparent power polygon",
          "protected_nodes":cfg["load"]["protected_nodes"],"start_slots":list(map(int,md["start_slots"])),
          "start_hours":(np.asarray(md["start_slots"])*dt).tolist(),"handover_threshold_mw":cfg["generator"]["handover_threshold_mw"]}}
    if result.get("x") is None:return out
    x=np.asarray(result["x"])
    def val(name):return x[idx[name]]
    pc=val("battery_charge");pd=val("battery_discharge");energy=val("energy");pv=val("pv_used");dg=val("dg_power")
    sc=val("served_critical");sg=val("served_general");fp=val("flow_p");fq=val("flow_q")
    target=val("target_power");starts=val("target_start");grid=val("grid_p")
    aux=starts@np.asarray(md["target_aux_coeff"]).T;online=starts@np.asarray(md["online_coeff"]).T
    event_map=np.asarray(md["switch_index_for_t"],dtype=int)
    a=val("switch_A")[:,event_map];b=val("switch_B")[:,event_map]
    dc=np.asarray(profiles["load_critical_mw"]);dgen=np.asarray(profiles["load_general_mw"])
    uc=np.maximum(0,dc[None]-sc);ug=np.maximum(0,dgen[None]-sg)
    capacities=np.array([m["energy_mwh"] for m in cfg["microgrids"]])
    def expected(arr):return float(np.dot(probs,np.asarray(arr)))
    eens_s=(uc+ug).sum(axis=(1,2))*dt
    weighted_eens=(uc*cfg["load"]["critical_voll"]/cfg["load"]["general_voll"]+ug).sum(axis=(1,2))*dt
    restored=(sc+sg).sum(axis=(1,2))*dt;demand_energy=float((dc+dgen).sum()*dt)
    lost_cost=((uc*np.asarray(profiles["critical_voll"])[None,None,:]+ug*np.asarray(profiles["general_voll"])[None,None,:]).sum(axis=(1,2))*dt)
    dg_cost=(dg*np.array([m["dg_cost"] for m in cfg["microgrids"]])[None,:,None]).sum(axis=(1,2))*dt
    throughput=((pc+pd)*np.array([m["throughput_cost"] for m in cfg["microgrids"]])[None,:,None]).sum(axis=(1,2))*dt
    startcost=starts.sum(axis=1)*cfg["generator"]["start_cost"]
    fuel=target.sum(axis=1)*dt*cfg["generator"]["fuel_cost"]
    gridcost=(grid*np.asarray(profiles["grid_price"])[None,:]).sum(axis=1)*dt
    crewcost=np.array([max(t["end_hour"] for t in sched["tasks"])*cfg["crew"]["hourly_cost"] for sched in md["schedules"]])
    costs={"outage_cost":expected(lost_cost),"microgrid_fuel_cost":expected(dg_cost),"battery_throughput_cost":expected(throughput),
           "generator_start_cost":expected(startcost),"generator_fuel_cost":expected(fuel),"grid_import_cost":expected(gridcost),"crew_cost":expected(crewcost)}
    # 阈值是“首个可用净出力”统计定义，不替代机组动态启动成功校核。
    threshold=cfg["generator"]["handover_threshold_mw"];handover=[];start_hours=[]
    for s in range(len(probs)):
        hit=np.flatnonzero(target[s]>=threshold-1e-6)
        handover.append(float(hit[0]*dt) if len(hit) else None)
        chosen=np.flatnonzero(starts[s]>0.5)
        start_hours.append(float(md["start_slots"][chosen[0]]*dt) if len(chosen) else None)
    hp=sum(probs[s] for s,h in enumerate(handover) if h is not None)
    out["summary"]={"expected_cost":float(result["objective"]),"expected_eens_mwh":expected(eens_s),"expected_weighted_eens_mwh":expected(weighted_eens),
       "expected_restored_energy_mwh":expected(restored),"total_demand_energy_mwh":demand_energy,
       "resilience_area_ratio":expected(restored)/demand_energy,"start_probability":expected(starts.sum(axis=1)),
       "handover_probability":float(hp),"expected_handover_hour":sum(probs[s]*h for s,h in enumerate(handover) if h is not None)/hp if hp else None,
       "scenario_eens_mwh":eens_s.tolist(),"scenario_handover_hour":handover,"scenario_start_hour":start_hours,
       "protected_critical_served_fraction":float(sc[:,cfg["load"]["protected_nodes"],:].sum()/np.broadcast_to(dc[None,cfg["load"]["protected_nodes"],:],sc[:,cfg["load"]["protected_nodes"],:].shape).sum())}
    series={"mg_charge_mw":pc,"mg_discharge_mw":pd,"mg_energy_mwh":energy,"mg_soc":energy/capacities[None,:,None],
      "mg_pv_used_mw":pv,"mg_dg_mw":dg,"mg_export_mw":-fp[:,[3,4,5],:],"upward_p_mw":-(fp[:,1,:]+fp[:,2,:]),
      "upward_q_mvar":-(fq[:,1,:]+fq[:,2,:]),"target_power_mw":target,"target_aux_mw":aux,"target_online":online,
      "target_start":starts,"grid_power_mw":grid,"served_critical_mw":sc,"served_general_mw":sg,
      "demand_critical_mw":dc,"demand_general_mw":dgen,"voltage_pu":np.sqrt(np.maximum(val("voltage_sq"),0)),
      "flow_p_mw":fp,"flow_q_mvar":fq,"switch_A":a,"switch_B":b,"mg_inverter_q_mvar":val("inverter_q")}
    out["series"]={k:v.tolist() for k,v in series.items()};out["cost_components"]=costs
    out["diagnostics"]={"reconstructed_cost":sum(costs.values()),"cost_reconstruction_error":sum(costs.values())-float(result["objective"]),
          "minimum_soc":float(np.min(series["mg_soc"])),"maximum_soc":float(np.max(series["mg_soc"])),
          "minimum_voltage_pu":float(np.min(series["voltage_pu"])),"maximum_voltage_pu":float(np.max(series["voltage_pu"])),
          "maxsimultaneous_charge_discharge_mw":float(np.max(np.minimum(pc,pd)))}
    return out

# ==================== 二、功率/SOC/信息因果性校核 ====================
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

# ==================== 三、多时段条件能力域与整条轨迹分解 ====================
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, QhullError



def _jsonable(value: Any) -> Any:
    """将 Numpy 数值转成标准 JSON 类型，避免复现实验时依赖 Numpy 解码。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def interface_trajectories(model: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """提取全部场景、全部时段的向上接口功率，单位 MW/Mvar。

    主模型边 1、2 正向均为 T1→D0，所以主配接口向上功率取其负和。
    不能用微网总发电替代接口净输出，因本地负荷及网络交换会改变它。
    """
    fp = np.asarray(model.var_indices["flow_p"])
    fq = np.asarray(model.var_indices["flow_q"])
    return -(x[fp[:, 1, :]] + x[fp[:, 2, :]]), -(x[fq[:, 1, :]] + x[fq[:, 2, :]])


def feasibility_residuals(model: Any, x: np.ndarray) -> dict[str, float]:
    """独立校核线性约束、边界及离散固定值，给出最大绝对违反量。

    主模型采用带物理单位的行，因此此处最大值只作数值残差诊断；
    物理 SOC、功率平衡等分项指标由 validation.py 进一步报告。
    """
    x = np.asarray(x, dtype=float)
    ax = np.asarray(model.A @ x).ravel()
    row_lower = np.maximum(np.asarray(model.row_lb) - ax, 0.0)
    row_upper = np.maximum(ax - np.asarray(model.row_ub), 0.0)
    bound_lower = np.maximum(np.asarray(model.lb) - x, 0.0)
    bound_upper = np.maximum(x - np.asarray(model.ub), 0.0)
    integer = np.asarray(model.integrality) != 0
    return {
        "row_max_violation": float(max(row_lower.max(initial=0), row_upper.max(initial=0))),
        "bound_max_violation": float(max(bound_lower.max(initial=0), bound_upper.max(initial=0))),
        "integer_max_violation": float(np.abs(x[integer] - np.rint(x[integer])).max(initial=0)),
    }


def _extract_witness(model: Any, x: np.ndarray) -> dict[str, Any]:
    """将一个高维可行点还原为可解释的 24 h 资源指令和 SOC 见证。"""
    x = np.asarray(x, dtype=float)
    indices = model.var_indices
    upward_p, upward_q = interface_trajectories(model, x)
    result: dict[str, Any] = {
        "upward_p_mw": upward_p,
        "upward_q_mvar": upward_q,
        "physical_objective": float(np.dot(np.asarray(model.c), x) + model.objective_constant),
        "feasibility": feasibility_residuals(model, x),
    }
    names = {
        "battery_charge": "mg_charge_mw", "battery_discharge": "mg_discharge_mw",
        "energy": "mg_energy_mwh", "pv_used": "mg_pv_used_mw",
        "dg_power": "mg_dg_mw", "inverter_q": "mg_inverter_q_mvar",
        "target_power": "target_power_mw", "served_critical": "served_critical_mw",
        "served_general": "served_general_mw",
    }
    for source, target in names.items():
        if source in indices:
            result[target] = x[np.asarray(indices[source])]
    # 微网边按 D0→M1/M2/M3 排列，负向即微网向配网净输出。
    fp = np.asarray(indices["flow_p"])
    fq = np.asarray(indices["flow_q"])
    result["mg_export_mw"] = -x[fp[:, [3, 4, 5], :]]
    result["mg_export_q_mvar"] = -x[fq[:, [3, 4, 5], :]]
    return _jsonable(result)


def decompose_trajectory(
    frozen_model: Any,
    upward_p_mw: np.ndarray,
    upward_q_mvar: np.ndarray | None = None,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """固定整条接口轨迹后，求满足物理约束的资源分解及最低运行成本。

    参数 ``frozen_model`` 必须已通过 freeze_discrete 固定全部整数变量。
    输入形状均为 (场景数, 时段数)，包含全部 96 个时段，而非只给图上
    的二维点。若轨迹不可行，返回失败状态，不将近似拟合当成可行分解。
    无功轨迹为 None 时仅固定有功，此时报告的是有功轨迹的可延拓性。

    这里通过集中式 LP 给出分解证书。它可以作为未来分布式 ADMM 的
    基准；整数恢复问题本身仍由主 MILP 求解，未对其声称 ADMM 收敛。
    """
    if np.any(np.asarray(frozen_model.integrality) != 0):
        raise ValueError("轨迹分解要求先固定离散决策并将模型松弛为连续 LP。")
    p = np.asarray(upward_p_mw, dtype=float)
    fp = np.asarray(frozen_model.var_indices["flow_p"])
    fq = np.asarray(frozen_model.var_indices["flow_q"])
    if p.shape != (fp.shape[0], fp.shape[2]):
        raise ValueError(f"有功轨迹应为 {(fp.shape[0], fp.shape[2])}，收到 {p.shape}。")
    q = None if upward_q_mvar is None else np.asarray(upward_q_mvar, dtype=float)
    if q is not None and q.shape != p.shape:
        raise ValueError("无功轨迹必须与有功轨迹具有相同形状。")

    # 每条等式对整条场景树的一个时刻固定接口净功率，保留其余约束。
    equalities: list[tuple[dict[int, float], float]] = []
    for s in range(p.shape[0]):
        for t in range(p.shape[1]):
            equalities.append(({int(fp[s, 1, t]): -1., int(fp[s, 2, t]): -1.}, float(p[s, t])))
            if q is not None:
                equalities.append(({int(fq[s, 1, t]): -1., int(fq[s, 2, t]): -1.}, float(q[s, t])))
    solved = solve_lp(frozen_model, extra_equalities=equalities)
    if solved.get("x") is None:
        return {"success": False, "message": str(solved.get("message", "分解 LP 未得到可行解")),
                "status": solved.get("status"), "fixes_reactive": q is not None}
    xx = np.asarray(solved["x"], dtype=float)
    pp, qq = interface_trajectories(frozen_model, xx)
    error_p = float(np.abs(pp - p).max(initial=0))
    error_q = None if q is None else float(np.abs(qq - q).max(initial=0))
    witness = _extract_witness(frozen_model, xx)
    max_violation = max(witness["feasibility"]["row_max_violation"],
                        witness["feasibility"]["bound_max_violation"], error_p,
                        0.0 if error_q is None else error_q)
    return {
        "success": bool(max_violation <= tolerance), "status": solved.get("status"),
        "message": str(solved.get("message", "")), "fixes_reactive": q is not None,
        "p_tracking_max_mw": error_p, "q_tracking_max_mvar": error_q,
        "certificate_tolerance": tolerance, "witness": witness,
        "solve_seconds": float(solved.get("solve_seconds", 0)),
    }


def _projection_steps(model: Any, x: np.ndarray, cfg: dict[str, Any]) -> list[int]:
    """默认每小时投影一次，覆盖全24h；每次求解仍保留全部96时段。

    interval_hours=0.25可展示全部96个截面。projection_hours仅用于显式
    指定的诊断时刻，不能把展示截面数误当成模型时段数。
    """
    n = np.asarray(model.var_indices["flow_p"]).shape[2]
    dt = float(cfg.get("step_minutes", 15)) / 60
    setting = cfg.get("capability", {})
    if "projection_hours" in setting:
        hours = np.asarray(setting["projection_hours"], dtype=float)
        if not np.isfinite(hours).all() or np.any(hours < 0) or np.any(hours > n*dt):
            raise ValueError("投影展示时刻必须位于研究时域内。")
        return sorted({min(n-1, int(round(h/dt))) for h in hours})
    interval = float(setting.get("interval_hours", 1.0))
    if not np.isfinite(interval) or interval<=0:
        raise ValueError("能力域截面间隔必须为有限正数。")
    stride = int(round(interval/dt))
    if stride < 1 or not np.isclose(stride*dt, interval):
        raise ValueError("能力域截面间隔须为调度步长的正整数倍。")
    return list(range(0, n, stride))




def compute_capability(
    model: Any,
    x: np.ndarray,
    cfg: dict[str, Any],
    outdir: str | Path | None = None,
    directions: int = 8,
    progress=None,
) -> dict[str, Any]:
    """生成条件 P–Q 截面、24 h 可行见证和完整轨迹分解结果。

    步骤 1 固定 MILP 整数解；步骤 2 对某场景某时刻沿等角度方向求
    支持函数；步骤 3 保存每个方向的全时域可行见证；步骤 4 选择一个
    支持方向的完整轨迹，用原运行成本重新解 LP 得到可执行资源分解。

    ``directions`` 只控制二维显示精度，不改变 24 h 时域长度。每个
    可行点均有原高维见证，所以其凸包安全包含在条件连续域内；若
    某 LP 未成功则不将该方向纳入已认证边界。
    """
    if int(directions) < 4:
        raise ValueError("至少需要 4 个投影方向。")
    x = np.asarray(x, dtype=float)
    conditioned = freeze_discrete(model, x)
    fp = np.asarray(conditioned.var_indices["flow_p"])
    fq = np.asarray(conditioned.var_indices["flow_q"])
    n = fp.shape[2]
    dt = float(cfg.get("step_minutes", 15)) / 60
    setting = cfg.get("capability", {})
    scenario = int(setting.get("scenario_index", 0))
    if scenario < 0 or scenario >= fp.shape[0]:
        raise ValueError("capability.scenario_index 超出场景范围。")
    basep, baseq = interface_trajectories(conditioned, x)
    slices: list[dict[str, Any]] = []
    witnesses: list[dict[str, Any]] = []
    selected_trajectory: tuple[np.ndarray, np.ndarray] | None = None
    all_x: dict[str, np.ndarray] = {"incumbent": x}

    steps = _projection_steps(model, x, cfg)
    for step_number, t in enumerate(steps, 1):
        points: list[list[float]] = []
        halfspaces: list[list[float]] = []
        records: list[dict[str, Any]] = []
        for k, angle in enumerate(np.linspace(0, 2 * math.pi, int(directions), endpoint=False)):
            cp, cq = float(math.cos(angle)), float(math.sin(angle))
            objective = np.zeros_like(np.asarray(conditioned.c), dtype=float)
            # solve_lp 最小化，-[cp*P_up+cq*Q_up] = cp*(P_L0+P_B)+...。
            for edge in (1, 2):
                objective[int(fp[scenario, edge, t])] += cp
                objective[int(fq[scenario, edge, t])] += cq
            solved = solve_lp(conditioned, objective=objective)
            if solved.get("x") is None:
                records.append({"direction": k, "angle_rad": float(angle), "success": False,
                                "message": str(solved.get("message", ""))})
                continue
            xx = np.asarray(solved["x"], dtype=float)
            residual = feasibility_residuals(conditioned, xx)
            good = max(residual["row_max_violation"], residual["bound_max_violation"]) <= 1e-6
            if not good:
                records.append({"direction": k, "angle_rad": float(angle), "success": False,
                                "message": "LP 解的独立残差超过 1e-6", "feasibility": residual})
                continue
            p, q = interface_trajectories(conditioned, xx)
            point = [float(p[scenario, t]), float(q[scenario, t])]
            h = cp * point[0] + cq * point[1]
            # 可行点本身足以进入内近似；外半空间还要求支持函数最优化成功。
            # 若将未收敛 LP 的可行目标当成最大值，会错误切掉真实可行域。
            support_certified = bool(solved.get("success", solved.get("status") == 0))
            witness_id = f"t{t:03d}_d{k:02d}"
            points.append(point)
            if support_certified:
                halfspaces.append([cp, cq, h])
            all_x[witness_id] = xx
            record_witness = {"id": witness_id, "time_step": t, "hour": t*dt,
                              "direction": k, "point_pq": point, "feasibility": residual}
            if not setting.get("compact_witnesses", False) or outdir is None:
                record_witness.update(_extract_witness(conditioned, xx))
            witnesses.append(record_witness)
            records.append({"direction": k, "angle_rad": float(angle), "success": True,
                            "point_pq": point, "support_value": h, "witness_id": witness_id,
                            "support_certified": support_certified,
                            "feasibility": residual, "solve_seconds": float(solved.get("solve_seconds", 0))})
            if selected_trajectory is None:
                selected_trajectory = (p, q)
        arr = np.asarray(points, dtype=float).reshape((-1, 2))
        # 重复极点不影响证书；只有满二维时才可画有面积的凸多边形。
        vertices: list[int] = []
        if len(arr) >= 3:
            try:
                vertices = ConvexHull(arr).vertices.tolist()
            except QhullError:
                vertices = []
        slices.append({"time_step": t, "hour": t * dt, "scenario_index": scenario,
                       "points_pq": points, "inner_hull_vertices": vertices,
                       "support_halfspaces": halfspaces, "directions": records,
                       "reference_pq": [float(basep[scenario, t]), float(baseq[scenario, t])]})
        if progress is not None:
            progress(f"  [24h投影] {step_number}/{len(steps)}：t={t*dt:g}h；{len(points)}个可行方向点", flush=True)

    reference = decompose_trajectory(conditioned, basep, baseq)
    alternative = None if selected_trajectory is None else decompose_trajectory(conditioned, *selected_trajectory)
    result: dict[str, Any] = {
        "scope": "fixed_discrete_full_horizon_system_conditioned_projection",
        "scope_zh": "固定离散方案、保留全时域与系统约束的条件接口投影",
        "horizon_hours": n * dt, "n_steps": n, "dt_hours": dt,
        "time_hours": (np.arange(n) * dt).tolist(), "scenario_index": scenario,
        "fixed_integer_count": int(np.count_nonzero(np.asarray(model.integrality))),
        "directions_requested": int(directions), "slices": slices,
        "scenario_name": model.metadata.get("scenarios", [{}])[scenario].get("name", str(scenario)),
        "projection_count": len(slices), "reference_upward_p_mw": basep.tolist(),
        "reference_upward_q_mvar": baseq.tolist(),
        "witness_storage": "high_dimensional_witnesses.npz; key=id in witness records",
        "witnesses": witnesses, "reference_decomposition": reference,
        "alternative_decomposition": alternative,
        "interpretation": {
            "inner": "已认证高维可行点投影的凸包；仅对当前固定离散方案成立。",
            "outer": "成功求得的支持半空间交集；有限方向不能声称精确恢复完整投影域。",
            "temporal": "每次 LP 均保留全部时段；不允许任意拼接各二维截面中的点。",
            "decomposition": "全部场景、全部时段 P/Q 轨迹固定后，最小化原成本取得资源分解。",
        },
    }
    if outdir is not None:
        directory = Path(outdir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "capability.json").write_text(json.dumps(_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
        np.savez_compressed(directory / "high_dimensional_witnesses.npz", **all_x)
    return _jsonable(result)

# ==================== 四、策略循环与当前启动/等待的D值 ====================
import csv
import json
from pathlib import Path
import time
import numpy as np


def jsonable(obj):
    """规范JSON：NumPy转列表、非有限数转null，禁止输出伪造的有限界。"""
    if isinstance(obj,dict):return {str(k):jsonable(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [jsonable(v) for v in obj]
    if isinstance(obj,np.ndarray):return jsonable(obj.tolist())
    if isinstance(obj,(np.integer,)):return int(obj)
    if isinstance(obj,(float,np.floating)):return float(obj) if np.isfinite(obj) else None
    if isinstance(obj,np.bool_):return bool(obj)
    return obj


def save_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(jsonable(data),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def _aggregate(records):
    """从所有候选的界得到菜单最优值区间：L=min Lp，U=min Up。"""
    feasible=[r for r in records if r["result"].get("x") is not None]
    best=min(feasible,key=lambda r:r["result"]["objective"]) if feasible else None
    lower=min((r["result"]["lower_bound"] if r["result"]["status"]!=2 else np.inf for r in records),default=-np.inf)
    upper=best["result"]["objective"] if best else np.inf
    return {"objective":upper,"lower_bound":lower,"absolute_bound_gap":upper-lower if np.isfinite(upper-lower) else None,
            "policy_id":best["policy"]["id"] if best else None,"best":best,
            "certified_infeasible":all(r["result"]["status"]==2 for r in records),"candidate_count":len(records)}


def _public_aggregate(agg):
    return {k:v for k,v in agg.items() if k!="best"}


def run_experiment(cfg, output, policy_ids=None, branches=("start","wait"), capability=False,
                   baselines=False, progress=print):
    """运行全部24h实验；baseline=True另算同菜单禁止向上反送的嵌套基线。"""
    start_time=time.perf_counter();out=Path(output);out.mkdir(parents=True,exist_ok=True)
    checkpoints=out/"checkpoints";checkpoints.mkdir(exist_ok=True)
    save_json(out/"config_resolved.json",cfg)
    policies=enumerate_policies(cfg)
    if policy_ids and policy_ids!="all":
        wanted=set(policy_ids.split(",") if isinstance(policy_ids,str) else policy_ids)
        policies=[p for p in policies if p["id"] in wanted]
        if len(policies)!=len(wanted):raise ValueError("未知策略ID；可用A_11,A_10,A_01,A_00,B_11,B_10,B_01,B_00")
    records={"free":[],"start":[],"wait":[]};screening=[]
    def calculate(policy,branch,coordination=True,information="causal"):
        progress(f"[求解] {policy['id']} / {branch} / {'协同' if coordination else '禁止向上反送'}",flush=True)
        model=build_model(cfg,policy,branch=branch,coordination=coordination,information=information)
        result=solve(model,cfg["solver"])
        validation=validate_solution(model,result,cfg)
        if result.get("x") is not None and not validation.get("passed",validation.get("valid",True)):
            raise RuntimeError(f"解的物理校核失败: {policy['id']} {branch}: {validation}")
        name=f"{policy['id']}_{branch}_{'coord' if coordination else 'no_upward'}"
        if result.get("x") is not None:
            np.savez_compressed(checkpoints/f"{name}.npz",x=result["x"])
        summary={k:v for k,v in result.items() if k!="x"}
        save_json(checkpoints/f"{name}.json",{"policy":policy,"branch":branch,"solver":summary,"validation":validation})
        screening.append({"policy_id":policy["id"],"branch":branch,"coordination":coordination,"information":information,
            **summary,"validation_passed":validation.get("passed",validation.get("valid")),"reused_free":False})
        progress(f"  状态={result['status']}；J上界={result['objective']:.4f}；下界={result['lower_bound']:.4f}；{result['solve_seconds']:.1f}s",flush=True)
        return {"policy":policy,"result":result,"validation":validation}
    for policy in policies:
        free=calculate(policy,"free");records["free"].append(free)
        for branch in branches:
            # 同动作的自由模型incumbent可复用；自由模型下界也是条件子集的有效下界。
            x=free["result"].get("x");reuse=False
            if x is not None:
                model=build_model(cfg,policy)
                current=x[model.var_indices["target_start"][:,0]]
                reuse=bool(np.all(current>0.5) if branch=="start" else np.all(current<0.5))
            if reuse:
                result=dict(free["result"]);result["message"]="复用满足本分支的自由模型可行解及有效下界；"+result["message"]
                record={"policy":policy,"result":result,"validation":free["validation"]}
                screening.append({"policy_id":policy["id"],"branch":branch,"coordination":True,"information":"causal",
                   **{k:v for k,v in result.items() if k!="x"},"validation_passed":True,"reused_free":True})
            else:record=calculate(policy,branch)
            records[branch].append(record)
    aggregates={name:_aggregate(rows) for name,rows in records.items() if rows}
    comparison={name:_public_aggregate(a) for name,a in aggregates.items()}
    if "start" in aggregates and "wait" in aggregates:
        s=aggregates["start"];w=aggregates["wait"]
        if np.isfinite(s["objective"]) and np.isfinite(w["objective"]):
            dl=w["lower_bound"]-s["objective"];du=w["objective"]-s["lower_bound"]
            comparison.update({"D_estimate":w["objective"]-s["objective"],"D_lower":dl,"D_upper":du,
                "D_units":"千元", "decision":"当前启动占优" if dl>1e-6 else "当前等待占优" if du<-1e-6 else "界区间跨零，尚不能判定",
                "definition":"D=J_wait*-J_start*；正值支持当前启动；wait仅禁止t=0启动，未来仍可启动"})
        else:comparison.update({"D_estimate":None,"D_lower":None,"D_upper":None,"decision":"至少一分支尚无可行解；检查求解状态，不计算有限D"})
    freebest=aggregates["free"]["best"]
    if freebest is None:
        save_json(out/"branch_comparison.json",comparison)
        raise RuntimeError("所有候选均无可行incumbent；请检查状态、增加限时或修正数据")
    bestmodel=build_model(cfg,freebest["policy"])
    solution=extract_solution(bestmodel,freebest["result"],cfg)
    save_json(out/"solution.json",solution);save_json(out/"validation.json",freebest["validation"])
    for name in ("start","wait"):
        if name in aggregates and aggregates[name]["best"]:
            r=aggregates[name]["best"];m=build_model(cfg,r["policy"],branch=name)
            save_json(out/f"solution_{name}.json",extract_solution(m,r["result"],cfg))
    # 固定抢修策略基线直接使用同模型已求结果，避免不必要重复求解。
    fixed=next((r for r in records["free"] if r["policy"]["id"]=="A_11"),records["free"][0])
    comparison["fixed_crew"]={"policy_id":fixed["policy"]["id"],"objective":fixed["result"]["objective"],"lower_bound":fixed["result"]["lower_bound"],
       "meaning":"首巡A且先修已巡地点；同一物理/信息模型。若不在菜单中则取首个被请求策略。"}
    if baselines:
        nocoord=[calculate(p,"free",coordination=False) for p in policies]
        a=_aggregate(nocoord);comparison["no_upward_coordination"]=_public_aggregate(a)
        if a["best"]:
            r=a["best"];m=build_model(cfg,r["policy"],coordination=False)
            save_json(out/"solution_no_upward.json",extract_solution(m,r["result"],cfg))
            comparison["coordination_gain_cost"]={"estimate":a["objective"]-aggregates["free"]["objective"],
                "lower":a["lower_bound"]-aggregates["free"]["objective"],"upper":a["objective"]-aggregates["free"]["lower_bound"],
                "note":"同菜单禁反送是协同可行集的子集；已优化的可选协同最优增益理论上非负，有限界不确定时不得声称负增益。"}
    cap=None
    if capability:
        progress("[能力域] 固定离散方案，保留全部24h约束投影与轨迹分解",flush=True)
        cap=compute_capability(bestmodel,freebest["result"]["x"],cfg,out/"capability",directions=cfg["experiment"]["capability_directions"],progress=progress)
    surface=None
    if cfg.get("experiment",{}).get("compute_cost_surface",False):
        progress("[条件代价] 扫描启动需求倍率与统一初始SOC；固定离散方案，逐点重解24h LP",flush=True)
        surface=compute_conditional_cost_surface(bestmodel,freebest["result"]["x"],cfg,out/"cost_surface",progress=progress)
    save_json(out/"branch_comparison.json",comparison)
    if screening:
        with (out/"policy_screening.csv").open("w",encoding="utf-8-sig",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(screening[0]));writer.writeheader();writer.writerows(screening)
    export_series(solution,out)
    plot_all(solution,out/"figures",comparison=comparison,capability=cap,cost_surface=surface)
    if (out/"solution_no_upward.json").exists():
        plot_baseline_comparison(solution,json.loads((out/"solution_no_upward.json").read_text()),out/"figures")
    elapsed=time.perf_counter()-start_time
    save_json(out/"run_manifest.json",{"elapsed_seconds":elapsed,"n_steps":cfg["n_steps"],"horizon_hours":cfg["horizon_hours"],
       "scenario_count":len(cfg["scenarios"]),"candidate_count":len(policies),"milp_solves":sum(not r["reused_free"] for r in screening),
       "best_policy":solution["policy_id"],"objective_upper":solution["summary"]["expected_cost"],"objective_lower":aggregates["free"]["lower_bound"],
       "note":"研究原型；自设参数；有限策略菜单；不代表实际电网/暂态黑启动认证"})
    return solution,comparison


def export_series(solution,out):
    """以长表导出场景/时段/PQ/恢复供电与各微网SOC，便于Excel改图复核。"""
    out=Path(out);ss=solution["series"];time=solution["time_hours"]
    fields=["scenario","time_hour","upward_p_mw","upward_q_mvar","target_power_mw","target_aux_mw","grid_power_mw","served_total_mw","switch_A","switch_B"]
    with (out/"interface_restoration_24h.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for s,name in enumerate(solution["scenario_names"]):
            for t,h in enumerate(time):
                row={"scenario":name,"time_hour":h}
                for key in fields[2:]:
                    row[key]=float(np.asarray(ss["served_critical_mw"])[s,:,t].sum()+np.asarray(ss["served_general_mw"])[s,:,t].sum()) if key=="served_total_mw" else ss[key][s][t]
                w.writerow(row)
    with (out/"microgrid_dispatch_24h.csv").open("w",encoding="utf-8-sig",newline="") as f:
        fields=["scenario","microgrid","time_hour","pv_mw","dg_mw","charge_mw","discharge_mw","export_mw","soc_start","soc_end"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for s,name in enumerate(solution["scenario_names"]):
            for m in range(3):
                for t,h in enumerate(time):
                    w.writerow(dict(zip(fields,[name,f"MG{m+1}",h,ss["mg_pv_used_mw"][s][m][t],ss["mg_dg_mw"][s][m][t],ss["mg_charge_mw"][s][m][t],ss["mg_discharge_mw"][s][m][t],ss["mg_export_mw"][s][m][t],ss["mg_soc"][s][m][t],ss["mg_soc"][s][m][t+1]])))


def plot_baseline_comparison(solution,baseline,out):
    """绘制同需求/场景的协同与禁反送韧性结果，直接回答协同体现在哪里。"""
    import matplotlib.pyplot as plt
    _set_style()
    t=np.asarray(solution["time_hours"]);dt=solution["dt_hours"];prob=np.asarray(solution["probabilities"])
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={"width_ratios":[1.5,1]})
    for sol,label,color in ((solution,"可选向上协同","#1767a5"),(baseline,"禁止向上反送","#ef963e")):
        served=np.asarray(sol["series"]["served_critical_mw"])+np.asarray(sol["series"]["served_general_mw"])
        power=np.einsum("s,snt->t",prob,served)
        _stairs(axes[0],t,power,dt,label=label,color=color,linewidth=2)
    demand=(np.asarray(solution["series"]["demand_critical_mw"])+np.asarray(solution["series"]["demand_general_mw"])).sum(axis=0)
    axes[0].plot(t,demand,ls="--",color="#6d8090",label="同一总需求")
    axes[0].set(xlim=(0,24),xlabel="灾后时间 / h",ylabel="期望恢复供电 / MW");axes[0].legend(fontsize=9);axes[0].grid(alpha=.3)
    vals=[solution["summary"]["expected_eens_mwh"],baseline["summary"]["expected_eens_mwh"]]
    bars=axes[1].bar(["可选协同","禁反送基线"],vals,color=["#1767a5","#ef963e"],width=.6)
    for bar,v in zip(bars,vals):axes[1].text(bar.get_x()+bar.get_width()/2,v+.08,f"{v:.3f}",ha="center")
    axes[1].set(ylabel="期望失供电量 EENS / MWh",ylim=(0,max(vals)*1.18+.2));axes[1].grid(axis="y",alpha=.3)
    fig.suptitle("主配微协同的韧性收益：同场景、同资源、同需求",fontsize=15,fontweight="bold")
    fig.text(.5,.015,"自设参数计算结果；协同可自由选择不反送，基线仅增加禁止向上反送约束。",ha="center",fontsize=9,color="#6d8090")
    fig.tight_layout(rect=(0,.04,1,.93));_save(fig,Path(out),"09_协同与禁反送基线_韧性对比")


# ==================== 五、初始条件与24h条件最优代价 ====================
def compute_conditional_cost_surface(model, x, cfg, outdir=None, progress=print):
    """逐点重解24h LP，展示启动需求倍率/统一初始SOC与条件最小费用。

    固定参考方案的启动、开关、充放模式和抢修策略。每个网格点只改变
    启动辅助功率轮廓的倍率和三个微网的共同初始SOC；全部96时段、
    四情景、末端储备、网络和信息约束均重新建立。灰色仅代表这个固定
    方案的LP被证明不可行，不代表所有恢复方案都不可行。

    返回状态区分：optimal、infeasible、unknown、invalid_parameter。
    代价含停电损失常数项；未收敛或无可行解的点不会伪造为代价曲面。
    """
    import copy
    from data import validate_config
    from openpyxl import Workbook
    settings = cfg.get("experiment", {})
    kappas = np.asarray(settings.get("surface_startup_multipliers", [.4,.6,.8,1.,1.2,1.4,1.6]), dtype=float)
    socs = np.asarray(settings.get("surface_soc_levels", [.2,.3,.4,.5,.6,.7,.8,.9]), dtype=float)
    if (len(kappas)<2 or len(socs)<2 or not np.isfinite(kappas).all() or not np.isfinite(socs).all()
            or np.any(kappas<=0) or np.any((socs<0)|(socs>1)) or np.any(np.diff(kappas)<=0) or np.any(np.diff(socs)<=0)):
        raise ValueError("条件代价扫描需要递增且至少2个倍率与SOC；倍率须正、SOC位于[0,1]。")
    x=np.asarray(x,dtype=float);policy=model.metadata["policy"]
    base_cfg=copy.deepcopy(cfg)
    # 包结构接口的cfg可能只存日曲线参数；此时使用已建参考模型的实际曲线。
    # 不是重新生成一套需求，确保各扫描点与参考方案采用同一物理场景。
    if "profiles" not in base_cfg:
        base_cfg["profiles"]={key:np.asarray(model.metadata["profiles"][key]).tolist()
            for key in ("pv_mw","load_critical_mw","load_general_mw","grid_price")}
    costs=np.full((len(socs),len(kappas)),np.nan)
    statuses=np.full(costs.shape,"unknown",dtype=object)
    records=[];witnesses={};profile=np.asarray(cfg["generator"]["startup_profile_mw"],dtype=float)
    for i,soc in enumerate(socs):
        for j,kappa in enumerate(kappas):
            point={"soc_initial":float(soc),"startup_multiplier":float(kappa),"status":"unknown",
                   "objective":None,"lower_bound":None,"solve_seconds":0.0}
            trial=copy.deepcopy(base_cfg)
            for mg in trial["microgrids"]:mg["soc_initial"]=float(soc)
            trial["generator"]["startup_profile_mw"]=(profile*kappa).tolist()
            try:
                validate_config(trial)
            except ValueError as exc:
                point.update(status="invalid_parameter",message=str(exc))
                statuses[i,j]=point["status"];records.append(point);continue
            trial_model=build_model(trial,policy,branch=model.metadata.get("branch","free"),
                information=model.metadata.get("information","causal"),
                coordination=model.metadata.get("coordination",True),
                fixed_switches=model.metadata.get("fixed_switches",False))
            if set(trial_model.var_indices)!=set(model.var_indices) or any(not np.array_equal(trial_model.var_indices[k],model.var_indices[k]) for k in model.var_indices):
                raise RuntimeError("扫描参数改变了动作/变量网格，不能套用原离散恢复方案。")
            fixed=freeze_discrete(trial_model,x)
            solved=solve_lp(fixed)
            point.update(solver_status=solved["status"],message=solved["message"],solve_seconds=solved["solve_seconds"])
            if solved.get("success") and solved.get("x") is not None:
                check=validate_solution(fixed,solved,trial,tolerance=1e-6)
                if check["passed"]:
                    point.update(status="optimal",objective=float(solved["objective"]),
                                 lower_bound=float(solved["lower_bound"]),validation_passed=True)
                    costs[i,j]=point["objective"]
                    key=f"soc{i:02d}_task{j:02d}";point["witness_id"]=key;witnesses[key]=solved["x"]
                else:point.update(message="LP候选没有通过物理校核",validation_passed=False)
            elif solved.get("status")==2:
                point["status"]="infeasible"
            statuses[i,j]=point["status"];records.append(point)
        if progress is not None:
            progress(f"  [条件代价] {i+1}/{len(socs)}：SOC={soc:g}，本行{int(np.isfinite(costs[i]).sum())}/{len(kappas)}个可行最优点",flush=True)
    reference=None
    candidates=[r for r in records if r["status"]=="optimal"]
    if candidates:
        mean_soc=float(np.mean([m["soc_initial"] for m in cfg["microgrids"]]))
        reference=min(candidates,key=lambda r:(r["startup_multiplier"]-1)**2+(r["soc_initial"]-mean_soc)**2)
    result={"scope":"fixed_discrete_full_horizon_conditional_cost",
        "scope_zh":"固定离散恢复方案和抢修策略，完整24h随机LP条件最小费用",
        "horizon_hours":cfg["horizon_hours"],"n_steps":cfg["n_steps"],"scenario_count":len(cfg["scenarios"]),"policy_id":policy["id"],
        "startup_selected":bool(np.any(x[model.var_indices["target_start"]]>.5)),
        "startup_multipliers":kappas.tolist(),"soc_levels":socs.tolist(),
        "objective_grid":costs.tolist(),"status_grid":statuses.tolist(),"records":records,
        "reference_point":reference,"cost_units":"千元（含停电损失常数）",
        "infeasible_meaning":"仅限该固定离散方案下的LP，不是整个混合整数恢复问题不可行",
        "grid_note":"每个网格点均重新建立完整96时段模型并求解；网格间连面仅用于显示"}
    result=jsonable(result)
    if outdir is not None:
        directory=Path(outdir);directory.mkdir(parents=True,exist_ok=True)
        save_json(directory/"conditional_cost_surface.json",result)
        np.savez_compressed(directory/"grid_witnesses.npz",**witnesses)
        columns=["soc_initial","startup_multiplier","status","objective","lower_bound","solve_seconds"]
        with (directory/"conditional_cost_grid.csv").open("w",encoding="utf-8-sig",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
            writer.writerows({k:r.get(k) for k in columns} for r in records)
        book=Workbook();ws=book.active;ws.title="条件最小费用"
        ws.append(["统一SOC/启动倍率",*kappas.tolist()])
        for i,soc in enumerate(socs):ws.append([float(soc),*[None if not np.isfinite(c) else float(c) for c in costs[i]]])
        ws=book.create_sheet("求解状态");ws.append(columns)
        for r in records:ws.append([r.get(k) for k in columns])
        book.save(directory/"条件代价网格.xlsx")
    return result
