"""24 小时条件接口域的投影、可行见证和完整轨迹分解。

本模块对应论文“能力域”环节，计算对象有严格边界：先固定主模型
MILP 的抢修策略及全部离散决策，然后投影其剩余连续可行集。投影
保留全部时段 SOC、末端储能、网络和条件非预期性约束。因此它是
当前离散方案下、含系统约束的条件接口域，并非全部离散方案的并集，
也不是与主网状态无关的微网固有能力域。

图上的二维截面用于解释。可行点的凸包是该条件凸域的内近似；
支持半空间则构成外近似。任意拼接不同截面的点不保证全时域可行。
本模块另外保存整条 24 h 接口轨迹并重新解分解 LP，避免这种误用。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .linear import freeze_discrete, solve_lp


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
    """选择展示时刻，内部优化仍保留完整 24 h，而不是改为三时段模型。"""
    n = np.asarray(model.var_indices["flow_p"]).shape[2]
    dt = float(cfg.get("step_minutes", 15)) / 60
    setting = cfg.get("capability", {})
    if "projection_hours" in setting:
        hours = list(setting["projection_hours"])
    else:
        # 优先展示实际启动阶段，再展示中午与晚间的可行截面。
        first_hour = 2.0
        if "target_start" in model.var_indices:
            starts = x[np.asarray(model.var_indices["target_start"])]
            selected = np.argwhere(starts > .5)
            if len(selected):
                slots = model.metadata.get("start_slots", [])
                if len(slots):
                    first_hour = float(slots[int(selected[0, 1])]) * dt
        hours = [first_hour, min(12., (n - 1) * dt), min(20., (n - 1) * dt)]
    return sorted({min(n - 1, max(0, int(round(float(h) / dt)))) for h in hours})


def compute_capability(
    model: Any,
    x: np.ndarray,
    cfg: dict[str, Any],
    outdir: str | Path | None = None,
    directions: int = 8,
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

    for t in _projection_steps(model, x, cfg):
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
            witnesses.append({"id": witness_id, "time_step": t, "hour": t * dt,
                              "direction": k, "point_pq": point, **_extract_witness(conditioned, xx)})
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

    reference = decompose_trajectory(conditioned, basep, baseq)
    alternative = None if selected_trajectory is None else decompose_trajectory(conditioned, *selected_trajectory)
    result: dict[str, Any] = {
        "scope": "fixed_discrete_full_horizon_system_conditioned_projection",
        "scope_zh": "固定离散方案、保留全时域与系统约束的条件接口投影",
        "horizon_hours": n * dt, "n_steps": n, "dt_hours": dt,
        "time_hours": (np.arange(n) * dt).tolist(), "scenario_index": scenario,
        "fixed_integer_count": int(np.count_nonzero(np.asarray(model.integrality))),
        "directions_requested": int(directions), "slices": slices,
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
