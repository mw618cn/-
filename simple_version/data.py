"""参数输入和结果表格：所有算例数值均从“参数数据.xlsx”读取。

本文件负责三件事：读取各工作表→检查单位/概率/时间/SOC→生成96时段
曲线。没有另一个JSON参数文件，也不会在缺少参数时悄悄采用隐藏默认值。
用户只改Excel的数值区域。行2是程序字段名，行3是中文列说明，数据从行4开始。
"""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
from openpyxl import load_workbook, Workbook
from openpyxl.styles import Font, PatternFill, Alignment

NODES = ["T0", "T1", "D0", "M1", "M2", "M3"]
MICROGRIDS = ["MG1", "MG2", "MG3"]
EDGE_NAMES = ["T0T1", "L0", "B", "D0M1", "D0M2", "A"]
MG_FIELDS = ["energy_mwh", "soc_initial", "soc_min", "soc_max", "soc_terminal",
             "charge_mw", "discharge_mw", "eta_charge", "eta_discharge", "pv_peak_mw",
             "dg_max_mw", "inverter_mva", "dg_cost", "throughput_cost"]
SCALAR_FIELDS = ["case_name", "horizon_hours", "step_minutes", "base_mva", "alpha",
    "grid_reference_mw", "grid_price", "voltage_min", "voltage_max", "polygon_sides",
    "battery_mode_minutes", "load.power_factor", "load.critical_voll", "load.general_voll",
    "load.elapsed_cost_growth", "generator.max_mw", "generator.min_mw",
    "generator.ramp_mw_per_hour", "generator.q_max_mvar", "generator.s_mva",
    "generator.startup_profile_step_minutes", "generator.aux_power_factor",
    "generator.start_interval_minutes", "generator.last_start_hour", "generator.start_cost",
    "generator.fuel_cost", "generator.handover_threshold_mw", "crew.hourly_cost",
    "solver.time_limit", "solver.mip_rel_gap", "solver.presolve", "solver.disp"]
INTEGER_FIELDS = {"step_minutes", "polygon_sides", "battery_mode_minutes",
                  "generator.startup_profile_step_minutes", "generator.start_interval_minutes"}
RUN_FIELDS = {"policy_ids", "compare_start_wait", "compute_capability", "compute_baseline",
              "run_rolling", "realized_scenario", "rolling_hours", "capability_directions",
              "curve_mode", "output_dir"}
# 新增绘图开关为可选项，旧版Excel仍能读取；只改变显示/附加分析设置。
OPTIONAL_RUN_SETTINGS = {
    "capability_interval_hours": 1.0,
    "compute_cost_surface": True,
    "surface_startup_multipliers": "0.4,0.6,0.8,1.0,1.2,1.4,1.6",
    "surface_soc_levels": "0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
}


def _number(value, label):
    """Excel单元格必须为有限数值；空白/未计算公式/文字会给出明确位置。"""
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label}必须填写数值；Excel公式请先计算并保存") from None
    if not math.isfinite(result):
        raise ValueError(f"{label}不允许NaN或无穷值")
    return result


def _boolean(value, label):
    """支持Excel的TRUE/FALSE以及中文“是/否”、0/1。"""
    text = str(value).strip().lower()
    if text in {"true", "1", "是", "yes"}: return True
    if text in {"false", "0", "否", "no"}: return False
    raise ValueError(f"{label}请填写“是”或“否”")


def _rows(book, sheet):
    """按行2字段读取行4以后的数据；检测重复表头，保留数值原精度。"""
    if sheet not in book.sheetnames:
        raise ValueError(f"参数文件缺少工作表：{sheet}")
    ws = book[sheet]
    headers = [c.value for c in ws[2]]
    if not headers or any(h is None for h in headers) or len(set(headers)) != len(headers):
        raise ValueError(f"{sheet}的第2行字段名缺失或重复，请保留表头")
    return [{str(h): row[i] for i, h in enumerate(headers)}
            for row in ws.iter_rows(min_row=4, values_only=True) if row and row[0] is not None]


def _keyed(book, sheet):
    """读取按parameter标识的纵向参数表，拒绝重复参数名。"""
    result = {}
    for row in _rows(book, sheet):
        key = str(row["parameter"]).strip()
        if key in result: raise ValueError(f"{sheet}参数重复：{key}")
        result[key] = row
    return result


def _assign(cfg, path, value):
    """把generator.max_mw等表格字段写入相应模型参数。"""
    keys = path.split("."); item = cfg
    for key in keys[:-1]: item = item.setdefault(key, {})
    item[keys[-1]] = value


def load_parameters(filename):
    """读一份Excel，返回模型cfg及运行开关settings。

    曲线方式=系数：96行表按节点基值/PV容量换算成MW，修改装机或负荷基值
    会立即生效；曲线方式=实际功率：直接把96行表视为MW实测/预测曲线。
    网络结构保持本原型的6节点、3微网和A/B两条修复线路。
    """
    path = Path(filename)
    if not path.is_file(): raise FileNotFoundError(f"没有找到参数文件：{path}")
    book = load_workbook(path, data_only=True)
    try:
        # 01 读取基础物理/费用/求解参数。无数值默认值，缺项即提示。
        table = _keyed(book, "基础参数")
        missing = set(SCALAR_FIELDS) - set(table)
        unknown = set(table) - set(SCALAR_FIELDS)
        if missing or unknown:
            raise ValueError(f"基础参数缺失{sorted(missing)}；未知项{sorted(unknown)}")
        cfg = {}
        for key in SCALAR_FIELDS:
            value = table[key]["value"]
            if key == "case_name": value = str(value)
            elif key in {"solver.presolve", "solver.disp"}: value = _boolean(value, key)
            else:
                value = _number(value, key)
                if key in INTEGER_FIELDS:
                    if not value.is_integer(): raise ValueError(f"{key}应为整数")
                    value = int(value)
            _assign(cfg, key, value)

        # 02 读取三微网：每行一个参数，三列分别为MG1/MG2/MG3。
        mg_table = _keyed(book, "微网参数")
        if set(mg_table) != set(MG_FIELDS): raise ValueError("微网参数行缺失或新增了未知字段")
        cfg["microgrids"] = [{"name": name, **{k: _number(mg_table[k][name], f"{name}.{k}")
                                                  for k in MG_FIELDS}} for name in MICROGRIDS]
        nodes = _rows(book, "节点负荷")
        if [r["node"] for r in nodes] != NODES: raise ValueError(f"节点负荷应按{NODES}排列")
        cfg["nodes"] = NODES.copy()
        cfg["load"]["critical_base_mw"] = [_number(r["critical_base_mw"], "关键负荷") for r in nodes]
        cfg["load"]["general_base_mw"] = [_number(r["general_base_mw"], "一般负荷") for r in nodes]
        cfg["load"]["protected_nodes"] = [i for i, r in enumerate(nodes) if _boolean(r["protected"], "关键负荷硬保护")]
        if set(cfg["load"]["protected_nodes"]) - {3,4,5}:
            raise ValueError("当前模型的硬保护用于M1/M2/M3；主配网关键负荷由高失供损失优先保障")
        edges = _rows(book, "线路参数")
        if [r["name"] for r in edges] != EDGE_NAMES: raise ValueError(f"线路应按{EDGE_NAMES}排列")
        cfg["edges"] = []
        expected_pairs = [(0,1),(1,2),(1,2),(2,3),(2,4),(2,5)]
        for row, pair in zip(edges, expected_pairs):
            if (row["from_node"], row["to_node"]) != (NODES[pair[0]], NODES[pair[1]]):
                raise ValueError("本原型固定6节点拓扑；可以改容量/阻抗，扩展接线需同步改power_model.py")
            values = {k: _number(row[k], f"{row['name']}.{k}") for k in ("s_mva", "r_pu", "x_pu")}
            if values["s_mva"] <= 0 or min(values["r_pu"], values["x_pu"]) < 0: raise ValueError("线路容量须正、阻抗非负")
            cfg["edges"].append({"name": row["name"], "from": pair[0], "to": pair[1], **values})

        # 03 巡检、修复和路程共同决定每个情景的信息揭示与通道可用时刻。
        crews = _rows(book, "巡检抢修")
        if [r["site"] for r in crews] != ["A", "B"]: raise ValueError("巡检抢修表需要A、B两行")
        cfg["crew"].update({"inspection_hours": {}, "repair_hours": {}, "travel_hours": {}})
        for row in crews:
            site = row["site"]
            cfg["crew"]["inspection_hours"][site] = _number(row["inspection_hour"], "巡检时间")
            cfg["crew"]["repair_hours"][site] = {"light": _number(row["light_hour"], "轻伤修复"), "heavy": _number(row["heavy_hour"], "重伤修复")}
        for row in _rows(book, "行程时间"):
            origin, target = str(row["origin"]), str(row["destination"])
            mapping = cfg["crew"]["travel_hours"].setdefault(origin, {})
            if target in mapping: raise ValueError("行程时间表存在重复路段")
            mapping[target] = _number(row["travel_hour"], "行程时间")
        for origin, target in [("depot","A"),("depot","B"),("A","B"),("B","A")]:
            if target not in cfg["crew"]["travel_hours"].get(origin, {}): raise ValueError(f"缺少行程{origin}→{target}")
        damage_names = {"light":"light", "heavy":"heavy", "轻":"light", "重":"heavy"}
        cfg["scenarios"] = []
        for row in _rows(book, "损伤场景"):
            try: damage = {site: damage_names[str(row["damage_"+site]).strip()] for site in ("A", "B")}
            except KeyError: raise ValueError("损伤类型应为light/heavy或轻/重") from None
            cfg["scenarios"].append({"id": str(row["id"]), "name": str(row["name"]),
                "probability": _number(row["probability"], "场景概率"), "damage": damage})
        startup = _rows(book, "启动曲线")
        ds = cfg["generator"]["startup_profile_step_minutes"] / 60
        if not startup or any(not np.isclose(_number(r["time_hour"], "启动阶段时刻"), i*ds) for i,r in enumerate(startup)):
            raise ValueError("启动曲线时刻应从0开始，按辅助功率分段时间等间隔排列")
        cfg["generator"]["startup_profile_mw"] = [_number(r["aux_power_mw"], "启动辅助功率") for r in startup]
        validate_config(cfg)
        if cfg["base_mva"] <= 0: raise ValueError("标幺容量基准base_mva必须为正")
        if not 0 < cfg["voltage_min"] <= 1 <= cfg["voltage_max"]: raise ValueError("电压上下界须为正并包含参考值1pu")
        if cfg["generator"]["min_mw"] < 0 or cfg["generator"]["max_mw"] < cfg["generator"]["min_mw"]: raise ValueError("机组有功上下界不合法")
        if cfg["generator"]["ramp_mw_per_hour"] <= 0 or cfg["generator"]["s_mva"] <= 0: raise ValueError("机组爬坡与额定容量须为正")
        if cfg["load"]["critical_voll"] <= 0 or cfg["load"]["general_voll"] <= 0: raise ValueError("停电损失系数须为正")
        if any(v < 0 for v in cfg["load"]["critical_base_mw"] + cfg["load"]["general_base_mw"]): raise ValueError("负荷不可为负")
        if not cfg["scenarios"]: raise ValueError("损伤场景不能为空")
        if cfg["solver"]["time_limit"] <= 0 or cfg["solver"]["mip_rel_gap"] < 0: raise ValueError("求解限时须正、相对间隙非负")

        # 04 运行开关与96点数据。后面的所有模型都使用同一个cfg。
        switches = _keyed(book, "运行设置")
        if not RUN_FIELDS <= set(switches) or set(switches) - RUN_FIELDS - set(OPTIONAL_RUN_SETTINGS):
            raise ValueError("运行设置参数缺失或有未知项")
        settings = {**OPTIONAL_RUN_SETTINGS, **{key: row["value"] for key,row in switches.items()}}
        for key in ("compare_start_wait", "compute_capability", "compute_baseline", "run_rolling", "compute_cost_surface"):
            settings[key] = _boolean(settings[key], key)
        settings["policy_ids"] = str(settings["policy_ids"]).strip()
        choices = {f"{s}_{i}{j}" for s in ("A", "B") for i in (0,1) for j in (0,1)}
        if settings["policy_ids"] != "all" and not set(settings["policy_ids"].split(",")) <= choices:
            raise ValueError("抢修策略填all或A_11、B_11等ID，可用英文逗号选多个")
        directions = _number(settings["capability_directions"], "能力域查询方向数")
        if directions < 4 or not directions.is_integer(): raise ValueError("能力域方向数应为至少4的整数")
        interval = _number(settings["capability_interval_hours"], "全天能力截面间隔")
        dt = cfg["step_minutes"]/60
        if interval<=0 or not np.isclose(interval/dt,round(interval/dt)):
            raise ValueError("全天能力截面间隔须为调度步长的正整数倍")
        cfg["capability"] = {"interval_hours":interval,"compact_witnesses":True}
        settings["capability_interval_hours"] = interval
        cfg["experiment"] = {"capability_directions":int(directions),"compute_cost_surface":settings["compute_cost_surface"]}
        for key in ("surface_startup_multipliers","surface_soc_levels"):
            values = [_number(v,key) for v in str(settings[key]).replace("，",",").split(",") if v.strip()]
            if len(values)<2 or any(b<=a for a,b in zip(values,values[1:])):
                raise ValueError(f"{key}应填写至少2个递增数值，用英文逗号分隔")
            if key=="surface_startup_multipliers" and min(values)<=0:raise ValueError("启动需求倍率须为正")
            if key=="surface_soc_levels" and (min(values)<0 or max(values)>1):raise ValueError("SOC扫描值须在[0,1]内")
            cfg["experiment"][key] = values
        settings["realized_scenario"] = str(settings["realized_scenario"])
        if settings["realized_scenario"] not in {s["id"] for s in cfg["scenarios"]}: raise ValueError("滚动真实场景ID不在损伤场景表中")
        settings["rolling_hours"] = tuple(float(x) for x in str(settings["rolling_hours"]).split(",") if x.strip())
        if settings["run_rolling"]:
            hours = settings["rolling_hours"];dt=cfg["step_minutes"]/60
            if not hours or list(hours) != sorted(set(hours)) or any(h<=0 or h>=cfg["horizon_hours"] or not np.isclose(h/dt,round(h/dt)) for h in hours):
                raise ValueError("滚动时刻应递增、唯一、在时域内部且与调度步长对齐")
        mode = str(settings["curve_mode"]).strip()
        if mode not in {"系数", "实际功率"}: raise ValueError("曲线方式请填写“系数”或“实际功率”")
        curve = _rows(book, "24小时曲线");n=cfg["n_steps"];dt=cfg["step_minutes"]/60
        if len(curve) != n: raise ValueError(f"24小时曲线应有{n}行数据，目前{len(curve)}行")
        times = np.array([_number(r["time_hour"], "曲线时刻") for r in curve])
        if not np.allclose(times, np.arange(n)*dt, atol=1e-9,rtol=0): raise ValueError("曲线时刻必须与调度网格逐一对应")
        def series(keys): return np.array([[_number(r[key], key) for r in curve] for key in keys])
        pv = series(["pv_"+m for m in MICROGRIDS]);dc=series(["critical_"+m for m in NODES]);dg=series(["general_"+m for m in NODES])
        if min(pv.min(),dc.min(),dg.min()) < 0: raise ValueError("曲线不允许负值")
        if mode == "系数":
            if pv.max() > 1+1e-9: raise ValueError("光伏系数不应大于1；若输入MW请将曲线方式改为实际功率")
            pv *= np.array([m["pv_peak_mw"] for m in cfg["microgrids"]])[:,None]
            dc *= np.asarray(cfg["load"]["critical_base_mw"])[:,None]
            dg *= np.asarray(cfg["load"]["general_base_mw"])[:,None]
        price = series(["grid_price"])[0]
        if mode == "系数": price *= cfg["grid_price"]
        if price.min() < 0: raise ValueError("本算例购电价格须非负")
        cfg["profiles"] = {"pv_mw": pv.tolist(), "load_critical_mw": dc.tolist(), "load_general_mw": dg.tolist(), "grid_price": price.tolist()}
        return cfg, settings
    finally:
        book.close()


def build_profiles(cfg):
    """把Excel读取的完整时序转为模型数组，费用按灾后经历时间增加。"""
    n=cfg["n_steps"];dt=cfg["step_minutes"]/60;time=np.arange(n)*dt
    elapsed=1+cfg["load"]["elapsed_cost_growth"]*(time+dt/2)
    result={"time_hours":time,"dt_hours":dt,
            "critical_voll":cfg["load"]["critical_voll"]*elapsed,
            "general_voll":cfg["load"]["general_voll"]*elapsed}
    for key,shape in {"pv_mw":(3,n),"load_critical_mw":(6,n),"load_general_mw":(6,n),"grid_price":(n,)}.items():
        values=np.asarray(cfg["profiles"][key],dtype=float)
        if values.shape != shape or not np.isfinite(values).all() or (values<0).any(): raise ValueError(f"时序字段{key}维度或数值不合法")
        result[key]=values
    return result


def write_result_workbook(solution, comparison, filename):
    """把关键指标、D区间和24h功率/SOC另存Excel，方便直接查看结果。"""
    book=Workbook();ws=book.active;ws.title="核心指标"
    labels={"expected_cost":"可行费用上界/千元","expected_eens_mwh":"期望失供电量/MWh",
            "expected_restored_energy_mwh":"期望恢复电量/MWh","resilience_area_ratio":"供电恢复面积比",
            "start_probability":"情景加权启动比例","expected_handover_hour":"形成可用出力时刻/h"}
    ws.append(["指标","数值"])
    for key,label in labels.items():ws.append([label,solution["summary"].get(key)])
    ws.append(["所选抢修策略",solution["policy_id"]])
    ws.append(["最优费用下界/千元",comparison.get("free",{}).get("lower_bound")])
    ws.append(["费用上/下界差/千元",comparison.get("free",{}).get("absolute_bound_gap")])
    ws.append(["本次比较策略数",comparison.get("free",{}).get("candidate_count")])
    ws=book.create_sheet("启动等待D值");ws.append(["条目","数值"])
    for key,label in [("D_estimate","两可行方案费用差/千元"),("D_lower","D下界/千元"),("D_upper","D上界/千元"),("decision","当前行动判定")]:
        ws.append([label,comparison.get(key)])
    ws.append(["判据","D正支持启动，D负支持等待；区间跨零尚不能判定"])
    ws=book.create_sheet("24小时功率");ws.append(["场景","灾后时间/h","配网向上净送电/MW","目标机组出力/MW","启动辅助功率/MW","恢复供电/MW"])
    ss=solution["series"];served=np.asarray(ss["served_critical_mw"])+np.asarray(ss["served_general_mw"])
    for s,name in enumerate(solution["scenario_names"]):
        for t,h in enumerate(solution["time_hours"]):
            ws.append([name,h,ss["upward_p_mw"][s][t],ss["target_power_mw"][s][t],ss["target_aux_mw"][s][t],float(served[s,:,t].sum())])
    ws=book.create_sheet("微网SOC");ws.append(["场景","状态时刻/h","MG1_SOC","MG2_SOC","MG3_SOC"])
    for s,name in enumerate(solution["scenario_names"]):
        for t in range(len(solution["time_hours"])+1):
            ws.append([name,t*solution["dt_hours"],*[ss["mg_soc"][s][m][t] for m in range(3)]])
    for ws in book:
        ws.freeze_panes="B2";ws.auto_filter.ref=ws.dimensions
        for row in ws:
            for cell in row:
                cell.font=Font(name="微软雅黑",size=10);cell.alignment=Alignment(vertical="center")
                if cell.row==1:
                    cell.fill=PatternFill("solid",fgColor="17334A");cell.font=Font(name="微软雅黑",color="FFFFFF",bold=True)
        for col in ws.columns:ws.column_dimensions[col[0].column_letter].width=24
    book.save(filename)
def validate_config(cfg: dict) -> dict:
    """检查Excel参数的网格、概率、容量和SOC，避免错误数据进入模型。"""
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
