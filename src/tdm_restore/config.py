"""集中参数与24小时曲线：所有算例数值均为可替换的演示假设。

顺序：读取JSON→合并参数→检查时间网格/单位/概率→生成物理负荷和PV曲线。
不在求解器内部偷偷修改资源规模，也不预设灾后某个时刻主网自动恢复。
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np


def default_config() -> dict:
    """返回独立的默认参数；功率MW、能量MWh、时间h、费用千元。"""
    return {
        "case_name": "24h_demo", "horizon_hours": 24.0, "step_minutes": 15,
        "base_mva": 10.0, "alpha": 0.0, "grid_reference_mw": 3.0,
        "grid_price": 0.65, "voltage_min": 0.95, "voltage_max": 1.05,
        "polygon_sides": 8, "battery_mode_minutes": 60,
        "nodes": ["T0", "T1", "D0", "M1", "M2", "M3"],
        "edges": [
            {"name":"T0T1","from":0,"to":1,"s_mva":4.0,"r_pu":0.012,"x_pu":0.018},
            {"name":"L0","from":1,"to":2,"s_mva":0.9,"r_pu":0.022,"x_pu":0.024},
            {"name":"B","from":1,"to":2,"s_mva":2.1,"r_pu":0.012,"x_pu":0.015},
            {"name":"D0M1","from":2,"to":3,"s_mva":1.6,"r_pu":0.018,"x_pu":0.020},
            {"name":"D0M2","from":2,"to":4,"s_mva":1.5,"r_pu":0.020,"x_pu":0.023},
            {"name":"A","from":2,"to":5,"s_mva":1.3,"r_pu":0.022,"x_pu":0.025}],
        "microgrids": [
            {"name":"MG1","energy_mwh":2.6,"soc_initial":0.70,"soc_min":0.15,"soc_max":0.95,"soc_terminal":0.30,"charge_mw":0.70,"discharge_mw":0.90,"eta_charge":0.95,"eta_discharge":0.95,"pv_peak_mw":0.85,"dg_max_mw":0.42,"inverter_mva":1.30,"dg_cost":1.10,"throughput_cost":0.04},
            {"name":"MG2","energy_mwh":2.2,"soc_initial":0.66,"soc_min":0.15,"soc_max":0.95,"soc_terminal":0.30,"charge_mw":0.65,"discharge_mw":0.80,"eta_charge":0.95,"eta_discharge":0.95,"pv_peak_mw":0.70,"dg_max_mw":0.36,"inverter_mva":1.15,"dg_cost":1.15,"throughput_cost":0.04},
            {"name":"MG3","energy_mwh":1.8,"soc_initial":0.72,"soc_min":0.15,"soc_max":0.95,"soc_terminal":0.30,"charge_mw":0.50,"discharge_mw":0.65,"eta_charge":0.95,"eta_discharge":0.95,"pv_peak_mw":0.60,"dg_max_mw":0.30,"inverter_mva":1.00,"dg_cost":1.20,"throughput_cost":0.04}],
        "load": {"critical_base_mw":[0.0,0.30,0.50,0.18,0.16,0.13],"general_base_mw":[0.0,0.55,1.25,0.25,0.22,0.32],"power_factor":0.97,"critical_voll":35.0,"general_voll":9.0,"elapsed_cost_growth":0.025,"protected_nodes":[3,4,5]},
        "generator": {"max_mw":2.4,"min_mw":0.25,"ramp_mw_per_hour":1.8,"q_max_mvar":1.2,"s_mva":2.7,"startup_profile_mw":[0.45,0.75,0.55,0.25],"startup_profile_step_minutes":15,"aux_power_factor":0.95,"start_interval_minutes":60,"last_start_hour":8.0,"start_cost":5.0,"fuel_cost":0.75,"handover_threshold_mw":0.60},
        "crew": {"travel_hours":{"depot":{"A":0.50,"B":0.75},"A":{"B":0.50},"B":{"A":0.50}},"inspection_hours":{"A":0.50,"B":0.50},"repair_hours":{"A":{"light":1.0,"heavy":4.0},"B":{"light":1.5,"heavy":5.0}},"hourly_cost":0.12},
        "scenarios": [{"id":"LL","name":"A轻-B轻","probability":0.36,"damage":{"A":"light","B":"light"}},{"id":"LH","name":"A轻-B重","probability":0.24,"damage":{"A":"light","B":"heavy"}},{"id":"HL","name":"A重-B轻","probability":0.24,"damage":{"A":"heavy","B":"light"}},{"id":"HH","name":"A重-B重","probability":0.16,"damage":{"A":"heavy","B":"heavy"}}],
        "solver": {"time_limit":25.0,"mip_rel_gap":0.005,"presolve":True,"disp":False},
        "experiment": {"policies":"all","branches":["start","wait"],"capability_directions":8}
    }


def _merge(base: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(path=None, overrides=None) -> dict:
    """读取覆盖配置并检查物理单位与时间离散；列表采用整体替换。"""
    cfg = default_config()
    if path is not None:
        with Path(path).open(encoding="utf-8") as f:
            _merge(cfg, json.load(f))
    if overrides:
        _merge(cfg, overrides)
    dt = float(cfg["step_minutes"]) / 60
    if dt <= 0:
        raise ValueError("时间步长必须为正")
    n = float(cfg["horizon_hours"]) / dt
    if dt <= 0 or not np.isclose(n, round(n)) or n < 1:
        raise ValueError("时域必须为正且能被时间步长整除")
    cfg["n_steps"] = int(round(n))
    if len(cfg["nodes"]) != 6 or len(cfg["edges"]) != 6 or len(cfg["microgrids"]) != 3:
        raise ValueError("本原型固定6节点、6支路、3微网；扩展规模须同时修改网络与节点映射")
    if not 0 <= cfg["alpha"] <= 1:
        raise ValueError("alpha是剩余外部支撑比例，取值[0,1]；0表示无外部注入")
    probs = [s["probability"] for s in cfg["scenarios"]]
    if any(p <= 0 for p in probs) or not np.isclose(sum(probs), 1.0):
        raise ValueError("场景概率须为正且总和为1")
    for mg in cfg["microgrids"]:
        if not (0 <= mg["soc_min"] <= mg["soc_initial"] <= mg["soc_max"] <= 1):
            raise ValueError("初始SOC或SOC上下界不合法")
        if not mg["soc_min"] <= mg["soc_terminal"] <= mg["soc_max"]:
            raise ValueError("末端SOC储备不合法")
        if any(mg[k] <= 0 for k in ("energy_mwh","charge_mw","discharge_mw","eta_charge","eta_discharge")):
            raise ValueError("储能容量/功率/效率必须为正")
        if mg["eta_charge"] > 1 or mg["eta_discharge"] > 1:
            raise ValueError("充放电效率不得大于1")
    for minutes in (cfg["battery_mode_minutes"], cfg["generator"]["start_interval_minutes"]):
        if minutes <= 0 or not np.isclose(minutes / cfg["step_minutes"], round(minutes / cfg["step_minutes"])):
            raise ValueError("模式与启动决策周期须为调度步长的整数倍")
    for site in ("A","B"):
        vals = [cfg["crew"]["inspection_hours"][site], *cfg["crew"]["repair_hours"][site].values()]
        for val in vals:
            if val < 0 or not np.isclose(val/dt, round(val/dt)):
                raise ValueError("巡检/维修时间必须非负且与调度网格对齐")
    aux_minutes = cfg["generator"]["startup_profile_step_minutes"]
    if aux_minutes <= 0 or not np.isclose(aux_minutes/cfg["step_minutes"], round(aux_minutes/cfg["step_minutes"])):
        raise ValueError("辅助功率分段时间须为调度步长整数倍；禁止粗网格跳过启动峰值")
    if not cfg["generator"]["startup_profile_mw"] or any(p < 0 for p in cfg["generator"]["startup_profile_mw"]):
        raise ValueError("启动辅助功率序列须非空且非负")
    if cfg["polygon_sides"] < 4:
        raise ValueError("容量内接多边形至少4边")
    for pf in (cfg["load"]["power_factor"],cfg["generator"]["aux_power_factor"]):
        if not 0 < pf <= 1:
            raise ValueError("负荷和启动辅助负荷功率因数须在(0,1]")
    if len({s["id"] for s in cfg["scenarios"]}) != len(cfg["scenarios"]):
        raise ValueError("场景ID必须唯一")
    for origin, destinations in cfg["crew"]["travel_hours"].items():
        for val in destinations.values():
            if val < 0 or not np.isclose(val/dt, round(val/dt)):
                raise ValueError("行程时间须非负且与调度网格对齐")
    return cfg


def build_profiles(cfg: dict) -> dict:
    """生成24h中点负荷/PV曲线；负荷为MW，费用为千元/MWh。"""
    n = cfg["n_steps"]; dt = cfg["step_minutes"] / 60
    time = np.arange(n) * dt
    midpoint = time + dt / 2
    # 灾害发生时刻视为当天00:00；负荷日曲线只反映需求，不意味着自动恢复。
    shape = 0.85 + 0.13*np.exp(-((midpoint-9)/3)**2) + 0.30*np.exp(-((midpoint-19)/3.5)**2)
    critical = np.asarray(cfg["load"]["critical_base_mw"])[:,None] * (0.96+0.04*shape)
    general = np.asarray(cfg["load"]["general_base_mw"])[:,None] * shape
    solar = np.maximum(0, np.sin(np.pi*(midpoint-6)/12))
    solar[(midpoint<6)|(midpoint>18)] = 0
    pv = np.array([m["pv_peak_mw"] for m in cfg["microgrids"]])[:,None] * solar
    elapsed = 1 + cfg["load"]["elapsed_cost_growth"] * midpoint
    result = {"time_hours":time,"dt_hours":dt,"pv_mw":pv,"load_critical_mw":critical,
            "load_general_mw":general,"critical_voll":cfg["load"]["critical_voll"]*elapsed,
            "general_voll":cfg["load"]["general_voll"]*elapsed,
            "grid_price":np.full(n,cfg["grid_price"])}
    # 真实曲线可直接写入JSON profiles；仅覆盖明确支持的字段，不改变时间轴。
    expected_shapes={"pv_mw":(3,n),"load_critical_mw":(6,n),"load_general_mw":(6,n),
                     "grid_price":(n,),"critical_voll":(n,),"general_voll":(n,)}
    for key, values in cfg.get("profiles",{}).items():
        if key not in expected_shapes:
            raise ValueError(f"未知曲线字段 {key}；请检查拼写")
        data=np.asarray(values,dtype=float)
        if data.shape!=expected_shapes[key] or not np.isfinite(data).all() or (data<0).any():
            raise ValueError(f"曲线{key}应为{expected_shapes[key]}非负有限数组，当前为{data.shape}")
        result[key]=data
    return result
