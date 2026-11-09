"""实际求解 D(alpha, SOC) 扫描，并以菜单目标界判断启动/等待边界。

每个参数点重新建立同一 24 h 模型，比较“当前启动”和“当前等待、
未来仍可启动”。D 的估计值只是两个可行解费用之差；只有菜单最优
值上下界推出的整个 D 区间具有同号时，才给出已认证的行动偏好。
参数扫描可能耗时，因此本模块导入时不进行任何求解。
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import load_config
from .crew import enumerate_policies
from .experiments import run_experiment, save_json


_LABELS = {
    "start_certified": "区间支持当前启动",
    "wait_certified": "区间支持当前等待",
    "tie_certified": "容差内无差异",
    "undetermined": "D区间跨零，未判定",
    "bound_unknown": "有效界不足，未判定",
    "start_infeasible": "启动分支已证不可行",
    "wait_infeasible": "等待分支已证不可行",
    "both_infeasible": "两分支均已证不可行",
    "no_incumbent": "至少一分支无可行解，未判定",
    "missing_comparison": "缺少条件分支比较",
}
_COLORS = {
    "start_certified": "#1767a5", "wait_certified": "#d87924",
    "tie_certified": "#2f8b78", "undetermined": "#8f9dac",
    "bound_unknown": "#c1c9d0", "start_infeasible": "#cf7c73",
    "wait_infeasible": "#5e86b3", "both_infeasible": "#8d5665",
    "no_incumbent": "#b4a7bc", "missing_comparison": "#b8bec5",
}


def _finite(value: Any) -> float | None:
    """JSON 的 null、非数值和无穷界均保持为未知，不替换成零。"""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _parameter_fingerprint(cfg: dict[str, Any]) -> str:
    """排除扫描参数与纯求解设置后，检查不同算例能否共用一张参数相图。"""
    normalized = copy.deepcopy(cfg)
    for key in ("alpha", "solver", "experiment", "capability", "case_name"):
        normalized.pop(key, None)
    for mg in normalized.get("microgrids", []):
        mg.pop("soc_initial", None)
    encoded = json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _menu_ids(directory: Path) -> list[str]:
    """从实际求解记录读取菜单，防止把两个不同策略集合悄悄混成同一边界。"""
    file = directory / "policy_screening.csv"
    if not file.exists():
        return []
    with file.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    return sorted({row["policy_id"] for row in records
                   if row.get("branch") == "free" and row.get("coordination", "True").lower() == "true"})


def _derive_decision(comparison: dict[str, Any], tolerance: float = 1e-6) -> dict[str, Any]:
    """由菜单 U/L 推出 D 区间；分支不可行与限时未知分别处理。

    J_start∈[L_s,U_s]、J_wait∈[L_w,U_w] 时，
    D∈[L_w-U_s,U_w-L_s]。只用可行解之差判断符号是不成立的。
    """
    start = comparison.get("start", {})
    wait = comparison.get("wait", {})
    us, ls = _finite(start.get("objective")), _finite(start.get("lower_bound"))
    uw, lw = _finite(wait.get("objective")), _finite(wait.get("lower_bound"))
    si, wi = bool(start.get("certified_infeasible", False)), bool(wait.get("certified_infeasible", False))
    output: dict[str, Any] = {
        "start_upper": us, "start_lower": ls, "wait_upper": uw, "wait_lower": lw,
        "start_certified_infeasible": si, "wait_certified_infeasible": wi,
        "start_candidate_count": start.get("candidate_count"),
        "wait_candidate_count": wait.get("candidate_count"),
        "start_policy_id": start.get("policy_id"), "wait_policy_id": wait.get("policy_id"),
        "D_estimate": None, "D_lower": None, "D_upper": None,
    }
    if not start or not wait:
        status = "missing_comparison"
    elif si and wi:
        status = "both_infeasible"
    elif si:
        # 若等待分支还没得到可行解，不能声称“等待可行且占优”。
        status = "start_infeasible" if uw is not None else "no_incumbent"
    elif wi:
        status = "wait_infeasible" if us is not None else "no_incumbent"
    elif us is None or uw is None:
        status = "no_incumbent"
    else:
        estimate = uw - us
        lower = None if lw is None else lw - us
        upper = None if ls is None else uw - ls
        output.update(D_estimate=estimate, D_lower=lower, D_upper=upper)
        if lower is not None and lower > tolerance:
            status = "start_certified"
        elif upper is not None and upper < -tolerance:
            status = "wait_certified"
        elif lower is None or upper is None:
            status = "bound_unknown"
        elif abs(lower) <= tolerance and abs(upper) <= tolerance:
            status = "tie_certified"
        else:
            status = "undetermined"
    output.update(decision_code=status, decision_label=_LABELS[status])
    return output


def _read_case(path: str | Path) -> dict[str, Any]:
    """读取一个已计算算例的配置与分支界，不重求解、不填补缺失结果。"""
    directory = Path(path)
    if directory.is_file():
        directory = directory.parent
    cfgfile = directory / "config_resolved.json"
    compfile = directory / "branch_comparison.json"
    if not cfgfile.exists():
        raise FileNotFoundError(f"结果目录缺少 config_resolved.json：{directory}")
    cfg = json.loads(cfgfile.read_text(encoding="utf-8"))
    comparison = json.loads(compfile.read_text(encoding="utf-8")) if compfile.exists() else {}
    socs = [float(mg["soc_initial"]) for mg in cfg["microgrids"]]
    capacities = np.asarray([mg["energy_mwh"] for mg in cfg["microgrids"]], dtype=float)
    weighted = float(np.average(socs, weights=capacities))
    return {
        "case_directory": str(directory.resolve()), "case_name": cfg.get("case_name", directory.name),
        "alpha": float(cfg["alpha"]), "soc_axis": weighted, "soc_initial_values": socs,
        "soc_axis_kind": "common_initial_soc" if np.ptp(socs) < 1e-10 else "capacity_weighted_initial_soc",
        "horizon_hours": cfg["horizon_hours"], "step_minutes": cfg["step_minutes"],
        "per_milp_time_limit_seconds": cfg.get("solver", {}).get("time_limit"),
        "parameter_fingerprint": _parameter_fingerprint(cfg), "policy_ids": _menu_ids(directory),
        **_derive_decision(comparison),
    }


def _format_bound(value: float | None) -> str:
    """图上未知界写问号，避免读者把缺失值理解为零。"""
    return "?" if value is None else f"{value:.2g}" if 0 < abs(value) < .01 else f"{value:.2f}"


def _plot_summary(cases: list[dict[str, Any]], outdir: Path, phase_allowed: bool) -> list[str]:
    """绘制 D 区间棒及可选二维分类图，颜色始终由区间而非点估计决定。"""
    from .plotting import _set_style, _save
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    _set_style()
    outdir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(cases, key=lambda row: (row["soc_axis"], row["alpha"], row["case_name"]))
    paths: list[str] = []
    if not ordered:
        return paths
    height = max(4.8, .72 * len(ordered) + 2.2)
    fig, ax = plt.subplots(figsize=(12.6, height))
    finite_values = [v for row in ordered for key in ("D_lower", "D_upper", "D_estimate")
                     if (v := row.get(key)) is not None]
    extent = max([abs(v) for v in finite_values] + [1.])
    xmin, xmax = -1.15 * extent, 1.15 * extent
    for i, row in enumerate(ordered):
        color = _COLORS[row["decision_code"]]
        lo, hi, est = row["D_lower"], row["D_upper"], row["D_estimate"]
        if lo is not None and hi is not None:
            ax.hlines(i, lo, hi, color=color, lw=4, alpha=.85)
            ax.plot([lo, hi], [i, i], "|", color=color, markersize=14, markeredgewidth=1.6)
        elif est is not None:
            left, right = xmin if lo is None else lo, xmax if hi is None else hi
            ax.hlines(i, left, right, color=color, lw=2, ls=":", alpha=.8)
            if lo is None:
                ax.plot(xmin, i, "<", color=color)
            if hi is None:
                ax.plot(xmax, i, ">", color=color)
        if est is not None:
            ax.scatter(est, i, s=55, facecolor="white", edgecolor=color, linewidth=1.8, zorder=4)
        annotation = row["decision_label"]
        if est is not None:
            annotation += f"  [{_format_bound(lo)}, {_format_bound(hi)}]"
        ax.text(1.025, i, annotation, transform=ax.get_yaxis_transform(), va="center", fontsize=9,
                color=color, in_layout=False)
    labels = [f"α={r['alpha']:.2f}  |  SOC={100*r['soc_axis']:.1f}%" for r in ordered]
    ax.set_yticks(np.arange(len(ordered)), labels)
    ax.axvline(0., color="#263f56", lw=1.2, ls="--", zorder=0)
    ax.set(xlim=(xmin, xmax), ylim=(-.7, len(ordered)-.3),
           xlabel="D = J等待* − J当前启动* / 千元")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("启动还是等待：用最优值界判定 D 的符号", fontsize=16, fontweight="bold", x=.50)
    fig.text(.5, .025, "线段为菜单最优 D 的有效界；空心点仅为可行费用差估计；区间跨零不认定阈值。\n"
             "SOC为容量加权初始值（扫描时三微网取同值）；结果基于自设参数与明确的候选菜单。",
             ha="center", fontsize=9, color="#61768b")
    # 给右侧文字固定留白；tight_layout若再次计算框外文字，会把主轴挤窄。
    fig.subplots_adjust(left=.18, right=.66, bottom=.23, top=.84)
    name = "11_D边界_区间判据"
    _save(fig, outdir, name)
    paths.extend(str(outdir / f"{name}.{suffix}") for suffix in ("png", "svg", "pdf"))

    # 只有唯一且完整的二维网格、其余物理参数和菜单相同，才形成参数相图。
    alphas = sorted({r["alpha"] for r in ordered})
    socs = sorted({r["soc_axis"] for r in ordered})
    pairs = {(r["alpha"], r["soc_axis"]) for r in ordered}
    if phase_allowed and len(alphas) > 1 and len(socs) > 1 and len(pairs) == len(ordered) == len(alphas) * len(socs):
        keys = list(_LABELS)
        code_index = {key: i for i, key in enumerate(keys)}
        lookup = {(r["alpha"], r["soc_axis"]): r for r in ordered}
        matrix = np.array([[code_index[lookup[(a, s)]["decision_code"]] for a in alphas] for s in socs])
        fig, ax = plt.subplots(figsize=(max(8, len(alphas)*1.6), max(5.8, len(socs)*1.1)))
        ax.imshow(matrix, origin="lower", aspect="auto", vmin=-.5, vmax=len(keys)-.5,
                  cmap=ListedColormap([_COLORS[k] for k in keys]))
        ax.set_xticks(np.arange(len(alphas)), [f"{a:.2f}" for a in alphas])
        ax.set_yticks(np.arange(len(socs)), [f"{100*s:.0f}%" for s in socs])
        ax.set(xlabel="剩余外部主网支撑比例 α", ylabel="容量加权初始 SOC")
        for j, s in enumerate(socs):
            for i, a in enumerate(alphas):
                row = lookup[(a,s)]
                text = f"[{_format_bound(row['D_lower'])}, {_format_bound(row['D_upper'])}]"
                if row["D_estimate"] is None:
                    text = _LABELS[row["decision_code"]].replace("，", "\n")
                ax.text(i,j,text,ha="center",va="center",fontsize=9,color="white" if row["decision_code"] in {"start_certified","wait_certified","both_infeasible"} else "#162b40")
        used = {row["decision_code"] for row in ordered}
        ax.legend(handles=[Patch(color=_COLORS[k], label=_LABELS[k]) for k in keys if k in used],
                  loc="upper center", bbox_to_anchor=(.5,-.13), ncol=min(3,len(used)), fontsize=9)
        fig.suptitle("D(α, SOC) 参数扫描：按有效区间分类", fontsize=16, fontweight="bold")
        fig.text(.5,.018,"每格为实际重求解结果；不在采样点之间插值宣称临界曲线。括号内单位为千元。",ha="center",fontsize=9,color="#61768b")
        fig.tight_layout(rect=(0,.10,1,.94))
        name="12_D参数扫描_区间分类"
        _save(fig,outdir,name)
        paths.extend(str(outdir / f"{name}.{suffix}") for suffix in ("png","svg","pdf"))
    return paths


def summarize_cases(paths: Iterable[str | Path], outdir: str | Path) -> dict[str, Any]:
    """汇总已有算例并作图，不重求解；无法读取的目录单独列为错误。

    不同菜单或其他物理参数不同时，仍展示各自 D 区间，但不把它们连成
    同一条相变边界。该函数适合先计算少量关键工况，随后追加结果。
    """
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    cases, errors = [], []
    for path in paths:
        try:
            cases.append(_read_case(path))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    fingerprints = sorted({row["parameter_fingerprint"] for row in cases})
    menus = sorted({tuple(row["policy_ids"]) for row in cases})
    menu_known_and_equal = len(menus) == 1 and bool(menus[0])
    # 相同加权SOC可对应不同资源分布。二维标量SOC相图只对三微网同SOC
    # 扫描成立；异质SOC已有工况仍可画区间棒并保留完整SOC向量。
    common_soc_scan = bool(cases) and all(row["soc_axis_kind"] == "common_initial_soc" for row in cases)
    phase_allowed = len(fingerprints) <= 1 and menu_known_and_equal and common_soc_scan
    result = {
        "definition": "D=J_wait*-J_start*；wait只禁止当前启动，未来可启动",
        "interval_rule": "D∈[L_wait-U_start,U_wait-L_start]；估计值不替代有效界",
        "cases": cases, "errors": errors, "same_other_parameters": len(fingerprints) <= 1,
        "same_policy_menu": menu_known_and_equal, "phase_map_applicable": phase_allowed,
        "common_soc_scan": common_soc_scan,
        "parameter_fingerprints": fingerprints, "menus": menus,
    }
    if cases:
        # CSV 用字符串保留SOC向量和菜单，便于在Excel核对不同微网的初始状态。
        with (output / "sensitivity_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
            writer.writeheader()
            for row in cases:
                writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                                 for key, value in row.items()})
    result["figures"] = _plot_summary(cases, output / "figures", phase_allowed)
    save_json(output / "sensitivity_summary.json", result)
    return result


def run_sensitivity(
    cfg: dict[str, Any],
    output: str | Path,
    alphas: Iterable[float] | None = None,
    socs: Iterable[float] | None = None,
    pairs: Iterable[tuple[float, float]] | None = None,
    policy_ids: str | Iterable[str] = "all",
    time_limit: float | None = None,
    reuse_existing: bool = True,
    progress=print,
) -> dict[str, Any]:
    """实际重求解给定 alpha/SOC 参数点，再汇总条件分支与最优值界。

    参数 ``pairs`` 若提供则直接指定 (alpha, SOC) 点，否则使用两个序列
    的笛卡尔积。默认点为 alpha=[0,.5,1]、SOC=[.7]。每个点的三个
    微网 SOC 同值，容量、负荷、启动参数、信息和候选菜单均保持不变。
    原始 cfg 不被原地修改。限时是每个 MILP 的时限，不是整次扫描预算。
    """
    if pairs is None:
        aa = list(alphas) if alphas is not None else [0.,.5,1.]
        ss = list(socs) if socs is not None else [.7]
        points = [(float(a),float(s)) for s in ss for a in aa]
    else:
        points = [(float(a),float(s)) for a,s in pairs]
    if not points:
        raise ValueError("参数扫描至少需要一个(alpha,SOC)点。")
    if len(set(points)) != len(points):
        raise ValueError("参数扫描点重复；请每个(alpha,SOC)只指定一次。")
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("每MILP时限必须为正。")
    if isinstance(policy_ids, str):
        policy_ids = "all" if policy_ids == "all" else [value.strip() for value in policy_ids.split(",") if value.strip()]
    else:
        policy_ids = list(policy_ids)
    allowed = {policy["id"] for policy in enumerate_policies(cfg)}
    if policy_ids != "all" and (not policy_ids or set(policy_ids) - allowed):
        raise ValueError(f"策略菜单为空或包含未知ID；可选{','.join(sorted(allowed))}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    directories, failures = [], []

    def emit(message, **_kwargs):
        progress(message)

    for i,(alpha,soc) in enumerate(points):
        point_cfg = copy.deepcopy(cfg)
        point_cfg["alpha"] = alpha
        point_cfg["case_name"] = f"sensitivity_a{alpha:g}_soc{soc:g}"
        for mg in point_cfg["microgrids"]:
            mg["soc_initial"] = soc
        if time_limit is not None:
            point_cfg.setdefault("solver", {})["time_limit"] = float(time_limit)
        # 复用统一参数检查，避免扫描绕过SOC边界及时间离散合法性校核。
        point_cfg = load_config(overrides=point_cfg)
        label = f"{i:03d}_alpha_{alpha:.6g}_soc_{soc:.6g}".replace(".","p")
        directory = output / label
        directories.append(directory)
        existing = directory / "branch_comparison.json"
        resolved = directory / "config_resolved.json"
        reusable = False
        if reuse_existing and existing.exists() and resolved.exists():
            old_cfg = json.loads(resolved.read_text(encoding="utf-8"))
            # 相同目录名不足以证明可复用：配置和所请求菜单都须一致。
            desired = sorted(p["id"] for p in enumerate_policies(point_cfg)) if policy_ids == "all" else sorted(
                policy_ids.split(",") if isinstance(policy_ids,str) else policy_ids)
            menu_matches = _menu_ids(directory) == desired
            reusable = old_cfg == point_cfg and menu_matches
        if reusable:
            emit(f"[扫描] 复用已计算点 α={alpha:g}, SOC={soc:g}")
            continue
        emit(f"[扫描 {i+1}/{len(points)}] α={alpha:g}, SOC={soc:g}；每MILP限时={point_cfg['solver']['time_limit']}s")
        try:
            run_experiment(point_cfg,directory,policy_ids=policy_ids,branches=("start","wait"),
                           capability=False,baselines=False,progress=emit)
        except Exception as exc:
            # 限时无incumbent时主流水线可能终止；保留已写出的分支界，继续其他点。
            failure={"case_directory":str(directory),"alpha":alpha,"soc":soc,
                     "error":f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            save_json(directory / "sensitivity_failure.json",failure)
            emit(f"[扫描未完成] {failure['error']}")
    summary=summarize_cases(directories,output)
    summary["run_failures"]=failures
    save_json(output / "sensitivity_summary.json",summary)
    return summary
