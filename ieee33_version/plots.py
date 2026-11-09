"""把求解器的真实结果转换为论文结果图，而不是预先画出预期结论。

阅读顺序：①读取 ``solution`` 的统一输出；②概率加权或按场景展开；
③把 96 个调度区间画满 24 h；④同时输出 PNG、SVG、PDF。
此模块不求解模型，也不修改结果。所有“期望”均由输入的场景概率计算。
图中的示例参数标记提醒读者：这些图不是实际电网实测或暂态仿真结果。
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tdm_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "tdm_font_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


COLORS = ["#1767a5", "#13a496", "#ef963e", "#a862ad", "#d85562", "#66829a"]
INK = "#17334a"
GREY = "#6d8090"
TASK_NAMES = {"travel": "行驶", "inspection": "巡检", "inspect": "巡检", "repair": "维修"}


def _set_style() -> None:
    """注册可在 Linux 上使用的中文字体；Word 的微软雅黑设置由文档模块处理。"""
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        # Windows/macOS优先微软雅黑或本机已有中文字体，不依赖Linux绝对路径。
        candidates = ["Microsoft YaHei", "微软雅黑", "Noto Sans CJK SC", "SimHei", "PingFang SC", "Arial Unicode MS"]
        available = {f.name for f in font_manager.fontManager.ttflist}
        chosen = next((name for name in candidates if name in available), None)
        if chosen:
            plt.rcParams["font.family"] = chosen
        else:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = candidates + ["DejaVu Sans"]
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelcolor": INK,
        "axes.edgecolor": "#a9bac8",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "xtick.color": GREY,
        "ytick.color": GREY,
        "grid.color": "#dce5ed",
        "grid.alpha": 0.65,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def _array(series: dict, key: str) -> np.ndarray:
    """统一把 JSON 列表转为浮点数组，确保不在绘图阶段改变数据值。"""
    return np.asarray(series[key], dtype=float)


def _expected(a: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """仅在场景这一维做概率加权，保留设备维、节点维和时间维。"""
    return np.tensordot(probabilities, a, axes=(0, 0))


def _stairs(ax: plt.Axes, time: np.ndarray, y: np.ndarray, dt: float, **kwargs: Any) -> Any:
    """把每个区间的恒定出力延伸到区间末端，防止 23.75 h 后留白。"""
    return ax.step(np.r_[time, time[-1] + dt], np.r_[y, y[-1]], where="post", **kwargs)


def _time_axis(ax: plt.Axes, horizon: float, xlabel: bool = True) -> None:
    """统一为全研究时域坐标，而不是仅截取两个时段。"""
    ax.set_xlim(0, horizon)
    tick_step = 3 if horizon >= 18 else max(1, horizon / 6)
    ticks = np.arange(0, horizon + 0.01, tick_step)
    if not np.isclose(ticks[-1], horizon):
        ticks = np.r_[ticks, horizon]
    ax.set_xticks(ticks)
    if xlabel:
        ax.set_xlabel("灾后时间 / h")
    ax.grid(axis="both")


def _title(fig: plt.Figure, title: str, subtitle: str) -> None:
    """区分论文图标题与解释，避免把假设藏在图注中。"""
    fig.suptitle(title, x=0.055, ha="left", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.055, 0.925, subtitle, fontsize=9.3, color=GREY)


def _save(fig: plt.Figure, outdir: Path, name: str) -> list[str]:
    """同时保存位图与可编辑矢量图；关闭画布释放内存。"""
    paths = []
    for suffix in ("png", "svg", "pdf"):
        p = outdir / f"{name}.{suffix}"
        fig.savefig(p, dpi=200 if suffix == "png" else None, bbox_inches="tight")
        paths.append(str(p))
    plt.close(fig)
    return paths


def _plot_trajectory(sol: dict, outdir: Path) -> list[str]:
    """图1：同一解中展示上送、微网分解和跨时段 SOC，直接体现三级协同。"""
    a = sol["series"]
    t = np.asarray(sol["time_hours"])
    dt, horizon = sol["dt_hours"], sol["horizon_hours"]
    p = np.asarray(sol["probabilities"])
    fig, axes = plt.subplots(3, 1, figsize=(13.6, 10), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.07, hspace=0.40)
    _title(fig, "01  24小时支撑轨迹、微网分解与储能状态", "示例参数｜曲线为场景概率加权结果；期望轨迹用于展示，实际执行采用相应信息历史下的场景控制。")
    up = _expected(_array(a, "upward_p_mw"), p)
    aux = _expected(_array(a, "target_aux_mw"), p)
    gen = _expected(_array(a, "target_power_mw"), p)
    _stairs(axes[0], t, up, dt, color=COLORS[0], lw=2, label="配网向主网净上送")
    _stairs(axes[0], t, aux, dt, color=COLORS[2], lw=1.8, label="目标机组启动辅机需求")
    _stairs(axes[0], t, gen, dt, color=COLORS[1], lw=2, label="目标机组发电出力")
    axes[0].axhline(0, color=GREY, lw=0.8)
    axes[0].set_ylabel("功率 / MW")
    axes[0].set_title("(a) TSO—DSO接口及目标机组", loc="left")
    axes[0].legend(ncol=3, loc="upper right", fontsize=9)
    mg = _expected(_array(a, "mg_export_mw"), p)
    for m in range(mg.shape[0]):
        _stairs(axes[1], t, mg[m], dt, color=COLORS[m], lw=1.8, label=f"MGO{m+1} 接口净上送")
    axes[1].axhline(0, color=GREY, lw=0.8)
    axes[1].set_ylabel("微网接口功率 / MW")
    axes[1].set_title("(b) 聚合支撑分解为各微网指令（正值：向配网上送）", loc="left")
    axes[1].legend(ncol=3, loc="upper right", fontsize=9)
    soc = _expected(_array(a, "mg_soc"), p)
    for m in range(soc.shape[0]):
        axes[2].plot(np.arange(soc.shape[-1]) * dt, soc[m], color=COLORS[m], lw=2, label=f"MGO{m+1}")
    axes[2].set_ylabel("SOC / p.u.")
    axes[2].set_title("(c) 从初始状态到24 h末端的跨期能量耦合", loc="left")
    axes[2].legend(ncol=3, loc="upper right", fontsize=9)
    for ax in axes:
        _time_axis(ax, horizon, xlabel=False)
    axes[-1].set_xlabel("灾后时间 / h")
    return _save(fig, outdir, "01_24小时支撑_分解_SOC")


def _plot_resilience(sol: dict, outdir: Path) -> list[str]:
    """图2：以实际负荷恢复曲线和 EENS 积分刻画韧性，不用启动成功率代替韧性。"""
    a = sol["series"]
    t = np.asarray(sol["time_hours"])
    dt, horizon = sol["dt_hours"], sol["horizon_hours"]
    p = np.asarray(sol["probabilities"])
    critical = _array(a, "served_critical_mw").sum(axis=1)
    general = _array(a, "served_general_mw").sum(axis=1)
    demand = _array(a, "demand_critical_mw").sum(axis=0) + _array(a, "demand_general_mw").sum(axis=0)
    restored = critical + general
    perf = np.divide(restored, demand[None, :], out=np.ones_like(restored), where=demand[None, :] > 1e-9)
    eens = np.c_[np.zeros(len(p)), np.cumsum((demand[None, :] - restored) * dt, axis=1)]
    exp = _expected(restored, p)
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.1))
    fig.subplots_adjust(left=0.075, right=0.97, top=0.85, bottom=0.10, hspace=0.43, wspace=0.26)
    _title(fig, "02  韧性表现：恢复曲线与停电损失的时间积分", "示例参数｜恢复率使用相同时间点的总需求归一化；累计未供电量由逐时实际功率差积分得到。")
    ax = axes[0, 0]
    te = np.r_[t, horizon]
    ax.fill_between(te, np.r_[exp, exp[-1]], np.r_[demand, demand[-1]], step="post", color="#e9eef3", label="未供电功率面积")
    _stairs(ax, t, demand, dt, color=GREY, ls="--", lw=1.6, label="总需求")
    _stairs(ax, t, exp, dt, color=COLORS[0], lw=2.0, label="期望已恢复负荷")
    ax.set_title("(a) 供电恢复与未供电区域", loc="left")
    ax.set_ylabel("功率 / MW")
    ax.legend(fontsize=8.5, ncol=1)
    ax = axes[0, 1]
    for s in range(len(p)):
        _stairs(ax, t, perf[s], dt, color=COLORS[s], lw=1.4, alpha=0.8, label=f"S{s+1} (p={p[s]:.2f})")
    _stairs(ax, t, _expected(perf, p), dt, color=INK, lw=2.3, label="概率加权")
    ax.set_ylim(0, 1.04)
    ax.set_title("(b) 四场景供电恢复率", loc="left")
    ax.set_ylabel("已恢复负荷 / 总需求")
    ax.legend(fontsize=8, ncol=2)
    ax = axes[1, 0]
    for s in range(len(p)):
        ax.plot(np.arange(eens.shape[-1]) * dt, eens[s], color=COLORS[s], lw=1.5, alpha=0.8)
    ax.plot(np.arange(eens.shape[-1]) * dt, _expected(eens, p), color=INK, lw=2.5, label=f"期望EENS = {float(p @ eens[:, -1]):.2f} MWh")
    ax.set_title("(c) 累计未供电量：面积越小，恢复损失越低", loc="left")
    ax.set_ylabel("累计EENS / MWh")
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    y = np.arange(len(p))
    bars = ax.barh(y, eens[:, -1], color=COLORS[:len(p)], height=0.58)
    for b, value in zip(bars, eens[:, -1]):
        ax.text(value + max(eens[:, -1].max() * 0.02, 0.005), b.get_y() + b.get_height() / 2, f"{value:.2f}", va="center", fontsize=9)
    ax.set_yticks(y, [f"S{s+1} · 概率{p[s]:.2f}" for s in y])
    ax.invert_yaxis()
    ax.set_title("(d) 24 h总未供电量与场景风险", loc="left")
    ax.set_xlabel("EENS / MWh")
    ax.set_xlim(0, max(1e-3, eens[:, -1].max()) * 1.22)
    ax.grid(axis="x")
    for ax in axes.flat[:3]:
        _time_axis(ax, horizon)
    return _save(fig, outdir, "02_韧性恢复曲线_EENS")


def _plot_crew(sol: dict, outdir: Path) -> list[str]:
    """图3：维修队真实任务与观测时刻，说明策略如何改变信息与物理网络。"""
    schedules = sol.get("schedules", [])
    if not schedules:
        return []
    horizon = sol["horizon_hours"]
    fig, ax = plt.subplots(figsize=(14.0, 6.4))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.81, bottom=0.18)
    _title(fig, "03  条件巡检—维修策略：任务执行与信息揭示时刻", f"示例参数｜选中策略：{sol.get('policy_label', sol.get('policy_id', ''))}。任务不可抢占；完成巡检后才获知该位置损伤。")
    for s, schedule in enumerate(schedules):
        for task in schedule.get("tasks", []):
            start, end = float(task["start_hour"]), float(task["end_hour"])
            kind = task.get("kind", "")
            site = str(task.get("site", ""))
            color = "#c2cdd6" if kind == "travel" else COLORS[0 if site == "A" else 1]
            hatch = "///" if kind in ("inspection", "inspect") else None
            ax.broken_barh([(start, max(0, end - start))], (s - 0.24, 0.48), facecolors=color, edgecolors="white", linewidth=0.7, hatch=hatch)
            if end - start >= 0.7 and start < horizon:
                label = TASK_NAMES.get(kind, kind) + (" " + site if site in ("A", "B") else "")
                ax.text((start + min(end, horizon)) / 2, s, label, ha="center", va="center", fontsize=8, color="white" if kind != "travel" else INK, clip_on=True)
        for site, hour in schedule.get("inspection_hours", {}).items():
            hour = float(hour)
            if hour <= horizon:
                ax.scatter(hour, s - 0.39, marker="v", s=48, color=COLORS[0 if site == "A" else 1], zorder=4)
                ax.text(hour, s - 0.48, f"{site}观测\n{hour:g} h", ha="center", va="bottom", fontsize=8)
        for site, hour in schedule.get("repair_hours", {}).items():
            hour = float(hour)
            if hour <= horizon:
                ax.scatter(hour, s + 0.28, marker="o", s=31, facecolor="white", edgecolor=COLORS[0 if site == "A" else 1], linewidth=1.6, zorder=5)
    ax.set_yticks(np.arange(len(schedules)), [f"S{i+1}" for i in range(len(schedules))])
    ax.set_ylim(-0.95, len(schedules) - 0.12)
    ax.invert_yaxis()
    _time_axis(ax, horizon)
    ax.grid(axis="y", visible=False)
    ax.set_ylabel("损伤场景")
    ax.legend(handles=[Patch(facecolor=COLORS[0], label="位置A"), Patch(facecolor=COLORS[1], label="位置B"), Patch(facecolor="#c2cdd6", label="行驶"), Patch(facecolor="white", edgecolor=GREY, hatch="///", label="巡检区间"), Line2D([0], [0], marker="v", color="none", markerfacecolor=GREY, label="完成巡检 / 获得观测"), Line2D([0], [0], marker="o", color="none", markeredgecolor=GREY, label="维修完成 / 线路可用")], ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=9)
    return _save(fig, outdir, "03_巡检维修任务_观测时刻")


def _plot_information(sol: dict, outdir: Path) -> list[str]:
    """图4：同色表示相同观测历史，必须共享决策；分组来自实际巡检记录。"""
    schedules = sol.get("schedules", [])
    if not schedules or not all("observation_history_by_step" in s for s in schedules):
        return []
    n = len(sol["time_hours"])
    dt, horizon = sol["dt_hours"], sol["horizon_hours"]
    codes: dict[str, int] = {}
    groups = np.zeros((len(schedules), n), dtype=int)
    for s, schedule in enumerate(schedules):
        for t, hist in enumerate(schedule["observation_history_by_step"][:n]):
            key = json.dumps(hist, ensure_ascii=False, sort_keys=True)
            groups[s, t] = codes.setdefault(key, len(codes))
    colors = ["#e4eaf0", "#9bc8ec", "#92d3c6", "#f4c98c", "#c0acdc", "#f0a9b0", "#68a3cf", "#64af9b", "#ddac72", "#997bbb", "#c67686"]
    cmap = ListedColormap((colors * math.ceil(len(codes) / len(colors)))[:len(codes)])
    fig, axes = plt.subplots(2, 1, figsize=(13.6, 6.9), height_ratios=[2.1, 1.0], sharex=True)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.81, bottom=0.11, hspace=0.40)
    _title(fig, "04  条件非预期性：观测到来之前，场景不能采用不同决策", "示例参数｜同一时间列中，同色单元格属于同一信息组；只有巡检结果已经不同，控制才允许分支。")
    ax = axes[0]
    ax.imshow(groups, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-0.5, vmax=max(0.5, len(codes)-0.5), extent=[0, horizon, len(schedules)-0.5, -0.5])
    ax.set_yticks(range(len(schedules)), [f"S{s+1}" for s in range(len(schedules))])
    ax.set_ylabel("损伤场景")
    ax.set_title("(a) 信息历史分组矩阵", loc="left")
    ax.grid(False)
    for s, schedule in enumerate(schedules):
        for site, hour in schedule.get("inspection_hours", {}).items():
            if float(hour) <= horizon:
                ax.plot([hour, hour], [s-0.43, s+0.43], color=INK, lw=1.6)
    count = np.array([len(np.unique(groups[:, i])) for i in range(n)])
    _stairs(axes[1], np.asarray(sol["time_hours"]), count, dt, color=COLORS[0], lw=2)
    axes[1].set_yticks(range(1, len(schedules)+1))
    axes[1].set_ylim(0.75, len(schedules)+0.3)
    axes[1].set_ylabel("可区分信息组数")
    axes[1].set_title("(b) 从共同决策到条件决策", loc="left")
    for ax in axes:
        _time_axis(ax, horizon)
    axes[0].grid(False)
    return _save(fig, outdir, "04_条件非预期性_信息分组")


def _plot_startup(sol: dict, outdir: Path) -> list[str]:
    """图5：逐场景展示完整启动辅机供能及后续输出，避免只报告一个启动时点。"""
    a = sol["series"]
    t = np.asarray(sol["time_hours"])
    dt, horizon = sol["dt_hours"], sol["horizon_hours"]
    p = np.asarray(sol["probabilities"])
    aux, gen, up = (_array(a, k) for k in ("target_aux_mw", "target_power_mw", "upward_p_mw"))
    s_count = len(p)
    fig, axes = plt.subplots(s_count, 1, figsize=(13.5, 2.1*s_count + 2.0), sharex=True, squeeze=False)
    axes = axes[:, 0]
    fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.08, hspace=0.37)
    _title(fig, "05  启动并非一个时点：持续支撑、爬坡与恢复输出", "示例参数｜配网净上送可同时服务主网负荷；不能把整条上送曲线全部解释为目标机组的启动需求。")
    for s, ax in enumerate(axes):
        _stairs(ax, t, up[s], dt, color=COLORS[0], lw=1.5, label="配网净上送")
        _stairs(ax, t, aux[s], dt, color=COLORS[2], lw=1.7, label="启动辅机需求")
        _stairs(ax, t, gen[s], dt, color=COLORS[1], lw=1.9, label="机组发电出力")
        active = np.flatnonzero(aux[s] > 1e-7)
        if active.size:
            start, end = t[active[0]], t[active[-1]] + dt
            ax.axvspan(start, end, color=COLORS[2], alpha=0.08)
            ax.text(start + 0.12, 0.97, f"启动支撑 {start:g}—{end:g} h", transform=ax.get_xaxis_transform(), va="top", color="#a76513", fontsize=8.5)
        else:
            ax.text(0.02, 0.85, "本场景未启动", transform=ax.transAxes, fontsize=9, color=GREY)
        ax.set_ylabel("功率 / MW")
        ax.set_title(f"S{s+1} · 场景概率 {p[s]:.2f}", loc="left", fontsize=10)
        ax.axhline(0, color=GREY, lw=0.6)
        _time_axis(ax, horizon, xlabel=s == s_count - 1)
    axes[0].legend(ncol=3, loc="upper right", fontsize=8.5)
    return _save(fig, outdir, "05_各场景完整启动_爬坡_支撑")


def _plot_feasibility(sol: dict, outdir: Path) -> list[str]:
    """图6：同时呈现 SOC 的场景差异、节点电压及储能动作的物理校核入口。"""
    a = sol["series"]
    t = np.asarray(sol["time_hours"])
    dt, horizon = sol["dt_hours"], sol["horizon_hours"]
    p = np.asarray(sol["probabilities"])
    soc, v = _array(a, "mg_soc"), _array(a, "voltage_pu")
    charge, discharge = _array(a, "mg_charge_mw"), _array(a, "mg_discharge_mw")
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.85, bottom=0.08, hspace=0.44, wspace=0.25)
    _title(fig, "06  指令分解后的物理状态：SOC、电池动作与电压", "示例参数｜本图对应线性潮流和已声明设备约束；潮流方程残差、整数可行性等数值校核另见 validation.json。")
    for m in range(min(3, soc.shape[1])):
        ax = axes[m, 0]
        te = np.arange(soc.shape[-1]) * dt
        ax.fill_between(te, soc[:, m].min(axis=0), soc[:, m].max(axis=0), color=COLORS[m], alpha=0.17, label="四场景范围")
        ax.plot(te, _expected(soc[:, m], p), color=COLORS[m], lw=2, label="期望SOC")
        ax.set_ylabel("SOC / p.u.")
        ax.set_title(f"MGO{m+1} · 跨期能量状态", loc="left", fontsize=10)
        ax.legend(fontsize=8.5, ncol=2)
    ax = axes[0, 1]
    vp_min, vp_max = v.min(axis=(0, 1)), v.max(axis=(0, 1))
    te = np.r_[t, horizon]
    ax.fill_between(te, np.r_[vp_min, vp_min[-1]], np.r_[vp_max, vp_max[-1]], step="post", color=COLORS[0], alpha=0.20, label="全部节点/场景范围")
    _stairs(ax, t, vp_min, dt, color=COLORS[0], lw=1.2)
    _stairs(ax, t, vp_max, dt, color=COLORS[0], lw=1.2)
    ax.set_title("节点电压包络", loc="left", fontsize=10)
    ax.set_ylabel("电压幅值 / p.u.")
    ax.legend(fontsize=8)
    ch = _expected(charge, p).sum(axis=0)
    dis = _expected(discharge, p).sum(axis=0)
    ax = axes[1, 1]
    _stairs(ax, t, dis, dt, color=COLORS[2], lw=1.7, label="期望总放电")
    _stairs(ax, t, -ch, dt, color=COLORS[0], lw=1.7, label="期望总充电（负值）")
    ax.axhline(0, color=GREY, lw=0.6)
    ax.set_title("微网群储能充放电动作", loc="left", fontsize=10)
    ax.set_ylabel("功率 / MW")
    ax.legend(fontsize=8)
    ax = axes[2, 1]
    sa, sb = _array(a, "switch_A"), _array(a, "switch_B")
    _stairs(ax, t, _expected(sa, p), dt, color=COLORS[0], lw=1.8, label="A线路闭合概率")
    _stairs(ax, t, _expected(sb, p), dt, color=COLORS[1], lw=1.8, label="B通道投运概率")
    ax.set_ylim(-0.04, 1.04)
    ax.set_title("修复完成后仍由电气可行性决定是否投运", loc="left", fontsize=10)
    ax.set_ylabel("场景概率加权开关状态")
    ax.legend(fontsize=8)
    for ax in axes.flat:
        _time_axis(ax, horizon, xlabel=False)
    for ax in axes[-1]:
        ax.set_xlabel("灾后时间 / h")
    return _save(fig, outdir, "06_物理状态_分解可行性")


def _branch_record(value: Any) -> dict:
    """兼容完整解及紧凑分支摘要，避免图模块与实验调度器耦合。"""
    if not isinstance(value, dict):
        return {}
    solver = value.get("solver", value)
    return {
        "upper": solver.get("objective", value.get("objective", value.get("upper_bound", value.get("best_cost")))),
        "lower": solver.get("lower_bound", value.get("lower_bound")),
    }


def _finite(value: Any) -> bool:
    try:
        return value is not None and bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _plot_decision(comparison: dict | None, outdir: Path) -> list[str]:
    """图7：使用可行上界和求解下界画 D 的区间；界不分离就不声称策略优越。"""
    if not comparison:
        return []
    start = _branch_record(comparison.get("start", comparison.get("current_start", {})))
    wait = _branch_record(comparison.get("wait", comparison.get("current_wait", {})))
    if not _finite(start.get("upper")) or not _finite(wait.get("upper")):
        return []
    su, wu = float(start["upper"]), float(wait["upper"])
    sl = float(start["lower"]) if _finite(start.get("lower")) else None
    wl = float(wait["lower"]) if _finite(wait.get("lower")) else None
    dl = comparison.get("D_lower", wl-su if wl is not None else None)
    du = comparison.get("D_upper", wu-sl if sl is not None else None)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.1), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(left=0.09, right=0.96, top=0.79, bottom=0.19, wspace=0.40)
    _title(fig, "07  当前启动还是等待：D值和求解界一起报告", "相同24 h时域、场景先验、资源与信息规则｜D = 最优等待成本 − 最优当前启动成本。正值支持当前启动。")
    ax = axes[0]
    upper = [su, wu]
    lower = [sl, wl]
    for y, (u, l, color) in enumerate(zip(upper, lower, COLORS[:2])):
        ax.scatter(u, y, s=90, color=color, zorder=3, marker="o")
        if l is not None:
            ax.plot([l, u], [y, y], color=color, lw=5, alpha=0.6)
            ax.scatter(l, y, s=50, marker="|", color=color, zorder=4)
            text = f"[{l:.3f}, {u:.3f}]"
        else:
            text = f"可行成本 {u:.3f}；未提供下界"
        ax.text(0.04, y+0.23, text, transform=ax.get_yaxis_transform(), fontsize=9, color=color)
    ax.set_yticks([0, 1], ["当前启动", "当前等待"])
    ax.set_ylim(-0.5, 1.6)
    ax.set_xlabel("目标成本 / 模型货币单位")
    ax.set_title("(a) 每个分支的最优值所在区间", loc="left")
    ax.grid(axis="x")
    ax = axes[1]
    ax.axvline(0, color=GREY, ls="--", lw=1)
    if _finite(dl) and _finite(du):
        dl, du = float(dl), float(du)
        ax.plot([dl, du], [0, 0], color=COLORS[0], lw=7, solid_capstyle="round")
        ax.scatter([dl, du], [0, 0], s=70, color=COLORS[0], zorder=3)
        center = (dl+du)/2
        ax.text(center, 0.25, f"D ∈ [{dl:.3f}, {du:.3f}]", ha="center", fontsize=11, color=COLORS[0])
        verdict = "区间全为正：当前启动更优" if dl > 0 else ("区间全为负：等待更优" if du < 0 else "区间跨零：现有求解精度不足以判定")
        ax.text(0.5, 0.08, verdict, transform=ax.transAxes, ha="center", fontsize=10, color=INK)
        margin = max((du-dl)*0.25, abs(dl)*0.15, abs(du)*0.15, 0.01)
        ax.set_xlim(min(dl, 0)-margin, max(du, 0)+margin)
    else:
        ax.text(0.5, 0.6, "缺少有效分支下界\n仅能比较已找到的可行策略成本", transform=ax.transAxes, ha="center", va="center", fontsize=11)
    ax.set_ylim(-0.6, 0.7)
    ax.set_yticks([])
    ax.set_xlabel("D / 模型货币单位")
    ax.set_title("(b) 决策判据及证据强度", loc="left")
    ax.grid(axis="x")
    return _save(fig, outdir, "07_启动等待_D区间_求解界")


def _plot_capability(capability: dict | None, outdir: Path) -> list[str]:
    """图8：如提供真实支撑点，则画条件切片；不将二维图称为完整24 h域。"""
    if not capability:
        return []
    if capability.get("slices"):
        return _plot_capability_slices(capability, outdir)
    points = capability.get("points", capability.get("support_points", capability.get("samples", [])))
    if not points:
        return []
    if isinstance(points[0], dict):
        x = np.array([r.get("p1_mw", r.get("x", np.nan)) for r in points], dtype=float)
        y = np.array([r.get("p2_mw", r.get("y", np.nan)) for r in points], dtype=float)
        cost = np.array([r.get("cost", r.get("objective", np.nan)) for r in points], dtype=float)
    else:
        arr = np.asarray(points, dtype=float)
        x, y = arr[:, 0], arr[:, 1]
        cost = arr[:, 2] if arr.shape[1] > 2 else np.full_like(x, np.nan)
    good = np.isfinite(x) & np.isfinite(y)
    x, y, cost = x[good], y[good], cost[good]
    if not len(x):
        return []
    fig, ax = plt.subplots(figsize=(9.2, 7.7))
    fig.subplots_adjust(left=0.13, right=0.92, top=0.81, bottom=0.14)
    _title(fig, "08  24小时条件可调度域的二维投影", "固定离散方案及其余明确条件｜边界点由保留全部跨期SOC和网络约束的线性子问题计算。")
    if np.isfinite(cost).any():
        sc = ax.scatter(x, y, c=cost, cmap="viridis", s=50, edgecolor="white", linewidth=0.7)
        fig.colorbar(sc, ax=ax, label="对应轨迹成本 / 模型货币单位", shrink=0.80)
    else:
        ax.scatter(x, y, color=COLORS[0], s=45, label="支撑方向优化得到的可行点")
    if len(x) >= 3:
        from scipy.spatial import ConvexHull, QhullError
        try:
            hull = ConvexHull(np.c_[x, y])
            ids = np.r_[hull.vertices, hull.vertices[0]]
            ax.fill(x[ids], y[ids], color=COLORS[0], alpha=0.08)
            ax.plot(x[ids], y[ids], color=COLORS[0], lw=1.3, label="可行支撑点凸包（内近似）")
        except QhullError:
            pass
    ax.set_xlabel(capability.get("x_label", "接口功率分量1 / MW"))
    ax.set_ylabel(capability.get("y_label", "接口功率分量2 / MW"))
    ax.grid()
    ax.legend(fontsize=9, loc="best")
    fig.text(0.13, 0.045, "二维投影能展示能力的形状；任意拼接不同时刻的投影点仍不保证整条轨迹可分解。", fontsize=9, color=GREY)
    return _save(fig, outdir, "08_条件24小时域_二维投影")


def _plot_capability_slices(cap: dict, outdir: Path) -> list[str]:
    """把实际 P–Q 支撑优化的截面与可分解的24 h见证轨迹放在同一张图。

    上排的每个点都来自一个完整24 h可行解的投影，凸包是内近似。
    下排展示其中一条完整见证；它不会把不同方向优化得到的单时段极值拼接。
    """
    slices = [s for s in cap["slices"] if s.get("points_pq")]
    if len(slices)>3:
        slices=[slices[i] for i in np.rint(np.linspace(0,len(slices)-1,3)).astype(int)]
    if not slices:
        return []
    ncols = len(slices)
    fig = plt.figure(figsize=(14, 9))
    fig.subplots_adjust(left=0.075, right=0.97, top=0.84, bottom=0.09, hspace=0.55, wspace=0.31)
    gs = fig.add_gridspec(3, max(ncols, 2), height_ratios=[1.5, 1, 1])
    _title(fig, "08  代表时刻切片与完整24小时轨迹分解（全天演变见图12）", "示例参数｜固定整数方案后的连续子问题；投影保留全时域SOC、网络与信息约束，离散方案改变后需要重新计算。")
    for j, sl in enumerate(slices):
        ax = fig.add_subplot(gs[0, j])
        xy = np.asarray(sl["points_pq"], dtype=float)
        ax.scatter(xy[:, 0], xy[:, 1], color=COLORS[0], s=24, zorder=3, label="方向优化可行点")
        hull = sl.get("inner_hull_vertices", [])
        if len(hull) >= 3:
            ids = np.r_[hull, hull[0]].astype(int)
            ax.fill(xy[ids, 0], xy[ids, 1], color=COLORS[0], alpha=0.12)
            ax.plot(xy[ids, 0], xy[ids, 1], color=COLORS[0], lw=1.4, label="可行点凸包内近似")
        elif len(xy) >= 2:
            # 等式和固定历史可能使截面退化成线段；如实画出而非补造面积。
            span = np.ptp(xy, axis=0)
            order = np.argsort(xy[:, int(np.argmax(span))])
            ax.plot(xy[order, 0], xy[order, 1], color=COLORS[0], lw=1.5)
            ax.text(0.03, 0.04, "该条件截面为线段或单点", transform=ax.transAxes, color=GREY, fontsize=8)
        ref = sl.get("reference_pq")
        if ref is not None:
            ax.scatter(ref[0], ref[1], marker="*", s=110, color=COLORS[2], edgecolor="white", zorder=4, label="原求解接口点")
        hour = sl.get("hour", float(sl.get("time_step", 0)) * cap["dt_hours"])
        ax.set_title(f"S{int(sl.get('scenario_index', 0))+1} · t={hour:g} h 的P–Q投影", loc="left", fontsize=10)
        ax.set_xlabel("向主网上送P / MW")
        ax.set_ylabel("向主网上送Q / Mvar")
        ax.grid()
        if j == 0:
            ax.legend(fontsize=7.7, loc="best")
    # 见证必须来自一次完整的分解求解，不能由不同P–Q投影的极值拼接。
    dec = cap.get("alternative_decomposition", {})
    if not dec.get("success"):
        dec = cap.get("reference_decomposition", {})
    witness = dec.get("witness", {})
    if not witness:
        # 某些支持方向求解得到的见证也保留完整轨迹。
        witnesses = cap.get("witnesses", {})
        values = list(witnesses.values()) if isinstance(witnesses, dict) else list(witnesses)
        for candidate in values:
            if isinstance(candidate, dict) and candidate.get("upward_p_mw") is not None:
                witness = candidate
                break
    if witness:
        dt, horizon = float(cap["dt_hours"]), float(cap["horizon_hours"])
        t = np.asarray(cap.get("time_hours", np.arange(cap["n_steps"]) * dt), dtype=float)
        s = int(cap.get("scenario_index", 0))
        up = np.asarray(witness["upward_p_mw"], dtype=float)
        mg = np.asarray(witness["mg_export_mw"], dtype=float)
        ax = fig.add_subplot(gs[1, :])
        _stairs(ax, t, up[s], dt, color=INK, lw=2.0, label="被分解的聚合接口轨迹")
        for m in range(mg.shape[1]):
            _stairs(ax, t, mg[s, m], dt, color=COLORS[m], lw=1.3, label=f"MGO{m+1}接口指令")
        ax.set_ylabel("功率 / MW")
        ax.set_title("(b) 同一次完整轨迹分解的接口与微网出力见证", loc="left", fontsize=10)
        _time_axis(ax, horizon, xlabel=False)
        ax.legend(ncol=4, fontsize=8, loc="upper right")
        ax = fig.add_subplot(gs[2, :])
        energies = np.asarray(witness["mg_energy_mwh"], dtype=float)
        for m in range(energies.shape[1]):
            ax.plot(np.arange(energies.shape[-1])*dt, energies[s, m], color=COLORS[m], lw=1.6, label=f"MGO{m+1}")
        ax.set_ylabel("储能电量 / MWh")
        ax.set_title("(c) 与该分解轨迹配对的全部跨期能量状态", loc="left", fontsize=10)
        _time_axis(ax, horizon)
        ax.legend(ncol=3, fontsize=8, loc="upper right")
    else:
        ax = fig.add_subplot(gs[1:, :])
        ax.axis("off")
        ax.text(0.5, 0.5, "本次仅有投影支撑点，未提供完整分解见证。\n不能由这些二维图单独推断任意24 h轨迹都可执行。", ha="center", va="center", transform=ax.transAxes, fontsize=12)
    return _save(fig, outdir, "08_条件P_Q域投影_24小时分解见证")


def plot_all(solution: dict, outdir: str | Path, comparison: dict | None = None,
             capability: dict | None = None, cost_surface: dict | None = None) -> list[str]:
    """从统一求解结果生成完整图包，返回所有图文件路径。

    Parameters
    ----------
    solution : dict
        ``solution.extract_solution`` 的 JSON 可序列化结果，必须含真实 ``series``。
    outdir : str | Path
        结果目录；会创建目录并写入 ``figures_manifest.json``。
    comparison : dict, optional
        ``start``、``wait`` 分支结果；支持 ``D_lower``、``D_upper`` 显式区间。
        下界不存在时不画虚假的最优值差异结论。
    capability : dict, optional
        支撑方向优化实际得到的 ``points``，不接收手绘的“预期域”。

    Notes
    -----
    没有可行解时返回空列表，并记录原因；不会用虚构曲线填补失败结果。
    图中期望值只用于汇总展示，不作为跨场景可执行的统一指令。
    """
    _set_style()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    if not solution.get("series"):
        (out / "figures_manifest.json").write_text(json.dumps({"files": [], "reason": "没有可行解的实际时序，未生成结果图。"}, ensure_ascii=False, indent=2), encoding="utf-8")
        return []
    probabilities = np.asarray(solution["probabilities"], dtype=float)
    if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1):
        raise ValueError("场景概率必须非负且总和为1，不能在绘图时静默修改概率。")
    paths: list[str] = []
    for draw in (_plot_trajectory, _plot_resilience, _plot_crew, _plot_information, _plot_startup, _plot_feasibility):
        paths.extend(draw(solution, out))
    paths.extend(_plot_decision(comparison, out))
    paths.extend(_plot_capability(capability, out))
    paths.extend(_plot_capability_24h(capability, out))
    paths.extend(_plot_conditional_cost_surface(cost_surface, out))
    manifest = {
        "files": paths,
        "source": "实际求解输出，参数为可替换的示例值",
        "horizon_hours": solution["horizon_hours"],
        "dispatch_step_hours": solution["dt_hours"],
        "scenario_count": len(probabilities),
        "notes": ["所有时序图覆盖完整研究时域。", "期望轨迹用于汇总，不表示跨信息组共用该期望指令。", "韧性由实际恢复曲线和未供电量刻画；最优性证据另见求解界。"],
    }
    (out / "figures_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths


def plot_rolling(replay: dict, outdir: str | Path) -> list[str]:
    """用真实滚动回放数据展示“执行—观测—后验—锁历史—重算”的闭环。

    输入是 ``rolling.run_rolling_replay`` 的返回值或其 JSON 文件内容。
    左上窗口条形图使用每轮 ``current_hour`` 与 ``original_terminal_hour``；
    后验和SOC使用每轮实际记录；底部执行曲线使用 ``executed_solution``。
    本函数不生成任何观测、不修改后验，也不把固定抢修策略解释为路线滚动优化。
    若滚动未成功得到可校核执行轨迹，拒绝绘制完整结果图。
    """
    _set_style()
    rounds = replay.get("rounds", [])
    executed = replay.get("executed_solution", {})
    if not rounds or not executed.get("series"):
        return []
    if not replay.get("executed_validation", {}).get("passed", False):
        raise ValueError("滚动执行轨迹未通过校核，不能生成完整执行结果图。")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    horizon = float(replay["horizon_hours"])
    dt = float(executed["dt_hours"])
    now = np.array([r["current_hour"] for r in rounds], dtype=float)
    terminal = np.array([r["original_terminal_hour"] for r in rounds], dtype=float)
    if not np.allclose(terminal, horizon):
        raise ValueError("每轮必须保持原始研究终点，不能将移动终点画为固定终点。")
    scenario_ids = sorted({sid for r in rounds for sid in r["posterior"]})
    posterior = np.array([[r["posterior"].get(sid, 0.) for r in rounds] for sid in scenario_ids])
    if not np.allclose(posterior.sum(axis=0), 1):
        raise ValueError("真实后验的每轮概率和必须为1。")
    counts = [int(r["compatible_scenario_count"]) for r in rounds]
    transitions = []
    for count in counts:
        if not transitions or count != transitions[-1]:
            transitions.append(count)
    transition_text = " → ".join(map(str, transitions))
    series = executed["series"]
    t = np.asarray(executed["time_hours"], dtype=float)
    soc = _array(series, "mg_soc")[0]
    current_soc = np.asarray([r["current_soc"] for r in rounds], dtype=float)
    # 状态记录与实际执行轨迹应一致，不可用后验期望值取代真实状态。
    checkpoints = np.rint(now / dt).astype(int)
    if not np.allclose(soc[:, checkpoints].T, current_soc, atol=2e-5):
        raise ValueError("每轮真实SOC与最终执行记录不一致，需先修正回放数据。")
    fig = plt.figure(figsize=(14.5, 11.1))
    fig.subplots_adjust(left=0.075, right=0.965, top=0.855, bottom=0.075, hspace=0.58, wspace=0.24)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.45, 1.0, 1.0])
    _title(fig, "10  实际观测驱动的滚动调度：固定历史、更新未来", f"实际分支 {replay.get('realized_scenario_id', '')}｜固定抢修策略；{len(rounds)}次再求解；兼容场景数 {transition_text}；所有轮次保持同一{horizon:g} h终点。")

    # (a) 已执行前缀灰色，剩余重算窗口蓝色；执行一步用橙色细块强调。
    ax = fig.add_subplot(gs[0, :])
    for i, r in enumerate(rounds):
        h = float(r["current_hour"])
        ax.broken_barh([(0, h)], (i-.30, .60), facecolors="#cbd5de", edgecolors="white", linewidth=.6)
        ax.broken_barh([(h, horizon-h)], (i-.30, .60), facecolors="#d5e8f7", edgecolors="white", linewidth=.6)
        ax.broken_barh([(h, min(dt, horizon-h))], (i-.30, .60), facecolors=COLORS[2], edgecolors="none")
        ax.scatter(h, i, s=21, color=COLORS[0], zorder=3)
        ax.text(horizon + .18, i, f"{counts[i]}场景", va="center", fontsize=8.8, color=COLORS[0])
    ax.axvline(horizon, color=INK, lw=1.3)
    ax.set_yticks(range(len(rounds)), [f"第{i+1}轮 · {h:g} h" for i, h in enumerate(now)])
    ax.set_ylim(len(rounds)-.5, -.65)
    _time_axis(ax, horizon)
    ax.grid(axis="y", visible=False)
    ax.set_title("(a) 已执行部分不可改写；每轮仅重算当前时刻至原始24 h末端", loc="left", fontsize=11, pad=24)
    ax.legend(handles=[Patch(facecolor="#cbd5de", label="锁定已执行历史与当前状态"), Patch(facecolor="#d5e8f7", label="重算剩余时域"), Patch(facecolor=COLORS[2], label=f"下发当前{dt*60:g}分钟指令")], ncol=3, fontsize=8.7, loc="upper center", bbox_to_anchor=(.52, 1.04))

    # (b) 后验热图逐轮直接读取，0表示在该轮观测下被排除。
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(posterior, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    for i in range(posterior.shape[0]):
        for j in range(posterior.shape[1]):
            ax.text(j, i, f"{posterior[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if posterior[i,j]>.55 else INK)
    ax.set_xticks(range(len(rounds)), [f"{h:g}" for h in now])
    ax.set_yticks(range(len(scenario_ids)), scenario_ids)
    ax.set_xlabel("再求解时刻 / h（按轮次排列）")
    ax.set_ylabel("原始损伤场景")
    ax.set_title("(b) 只按已揭示巡检结果更新后验概率", loc="left", fontsize=11)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=.035, pad=.025)
    cbar.set_label("条件概率", fontsize=8)

    # (c) 真实SOC状态与重算时刻标记，初始状态和末端仍留在整条曲线中。
    ax = fig.add_subplot(gs[1, 1])
    for m in range(soc.shape[0]):
        ax.plot(np.arange(soc.shape[-1])*dt, soc[m], color=COLORS[m], lw=1.7, label=f"MGO{m+1}")
        ax.scatter(now, current_soc[:,m], color=COLORS[m], edgecolor="white", linewidth=.5, s=23, zorder=4)
    ax.set_ylabel("真实SOC / p.u.")
    ax.set_title("(c) 每轮继承真实状态，禁止重置SOC", loc="left", fontsize=11)
    _time_axis(ax, horizon)
    ax.legend(ncol=3, fontsize=8, loc="best")

    # (d) 最终一条实际分支的执行记录，不是事后完美信息重新优化。
    ax = fig.add_subplot(gs[2, :])
    up = _array(series, "upward_p_mw")[0]
    gen = _array(series, "target_power_mw")[0]
    _stairs(ax, t, up, dt, color=COLORS[0], lw=1.8, label="实际配网向主网净上送")
    _stairs(ax, t, gen, dt, color=COLORS[1], lw=1.8, label="实际目标机组发电")
    observed_hours = sorted({float(o["revealed_hour"]) for r in rounds for o in r["observations"]})
    for j, h in enumerate(observed_hours):
        ax.axvline(h, color=COLORS[3], ls="--", lw=1, alpha=.7,
                   label="损伤观测真正到达" if j==0 else None)
    ax.axhline(0, color=GREY, lw=.6)
    ax.set_ylabel("功率 / MW")
    ax.set_title("(d) 沿真实场景执行的24 h闭环轨迹", loc="left", fontsize=11)
    _time_axis(ax, horizon)
    ax.legend(ncol=3, fontsize=8.5, loc="upper right")
    fallback = int(replay.get("fallback_count", 0))
    fig.text(.075, .020, f"数据：rolling_replay.json；执行路径全时域校核通过。重算失败后使用上轮可行剩余计划的次数：{fallback}。抢修路线与已开始任务不在本回放中重新选择。", fontsize=9, color=GREY)
    paths = _save(fig, out, "10_滚动过程_后验_锁历史_真实状态")
    manifest_path = out / "figures_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"] = list(dict.fromkeys(manifest.get("files", []) + paths))
        manifest["rolling"] = {"rounds": len(rounds), "realized_scenario_id": replay.get("realized_scenario_id"), "fallback_count": fallback}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths


def _plot_capability_24h(cap: dict | None, outdir: Path) -> list[str]:
    """图12：时间—P—Q三维截面序列与同一条全时域可分解轨迹。

    每个面是一个时刻的投影内近似，各面不连接成虚构的高维轨迹域。
    红线取自同一次完整轨迹分解，而非拼接各时刻最大输出。
    """
    if not cap: return []
    slices=[s for s in cap.get("slices",[]) if s.get("points_pq")]
    if not slices: return []
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from scipy.spatial import ConvexHull, QhullError
    _set_style()
    horizon=float(cap["horizon_hours"]);dt=float(cap["dt_hours"]);n=int(cap["n_steps"])
    fig=plt.figure(figsize=(12.8,8.2))
    ax=fig.add_axes([.015,.12,.835,.75],projection="3d",computed_zorder=False)
    norm=Normalize(0,horizon);cmap=plt.get_cmap("viridis")
    _title(fig,"12  24小时能力域演变：时间—P—Q投影与完整轨迹",
        f"{len(slices)}个实际截面｜每个方向均求解完整{n}时段LP｜固定离散方案｜场景：{cap.get('scenario_name','S'+str(cap.get('scenario_index',0)+1))}")
    for sl in slices:
        hour=float(sl["hour"]);points=np.asarray(sl["points_pq"],dtype=float)
        ref=sl.get("reference_pq")
        if ref is not None:points=np.vstack((points,np.asarray(ref,dtype=float)))
        points=np.unique(np.round(points,11),axis=0);color=cmap(norm(hour))
        hull=None
        if len(points)>=3:
            try:hull=ConvexHull(points).vertices
            except QhullError:pass
        if hull is not None:
            poly=[(hour,float(p),float(q)) for p,q in points[hull]]
            face=Poly3DCollection([poly],facecolors=[color],edgecolors=[color],linewidths=.7,alpha=.14,zorder=1)
            ax.add_collection3d(face)
            boundary=np.vstack((points[hull],points[hull[0]]))
            ax.plot(np.full(len(boundary),hour),boundary[:,0],boundary[:,1],color=color,lw=.8,alpha=.9)
        elif len(points)>1:
            order=np.argsort(points[:,int(np.argmax(np.ptp(points,axis=0)))])
            ax.plot(np.full(len(points),hour),points[order,0],points[order,1],color=color,lw=1.1)
        else:ax.scatter([hour],points[:,0],points[:,1],color=[color],s=12)
    reference=cap.get("reference_decomposition",{})
    has_trajectory=bool(reference.get("success"))
    if has_trajectory:
        s=int(cap.get("scenario_index",0));w=reference["witness"]
        p=np.asarray(w["upward_p_mw"])[s];q=np.asarray(w["upward_q_mvar"])[s]
        t=np.asarray(cap["time_hours"])
        tt=np.repeat(np.r_[t,horizon],2)[1:-1]
        ax.plot(tt,np.repeat(p,2),np.repeat(q,2),color="#d8415a",lw=2.1,zorder=20)
        fig.legend(handles=[Line2D([0],[0],color="#d8415a",lw=2,label=f"同一条{n}时段轨迹（整条分解校核通过）")],
                  loc="upper center",bbox_to_anchor=(.48,.895),fontsize=9)
    ax.set(xlim=(0,horizon),xlabel="灾后时间 / h",ylabel="接口 P / MW",zlabel="接口 Q / Mvar")
    ax.set_xticks(np.linspace(0,horizon,9));ax.set_box_aspect((2.2,1.25,1.0),zoom=1.02);ax.view_init(elev=25,azim=-58)
    ax.tick_params(labelsize=8);ax.xaxis.labelpad=13;ax.yaxis.labelpad=13;ax.zaxis.labelpad=9
    cbax=fig.add_axes([.90,.24,.016,.48])
    fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cbax,ticks=np.linspace(0,horizon,5),label="截面对应时刻 / h")
    fig.text(.06,.058,"每个彩色截面是完整24h模型在一个时刻的投影；它们不能任意组合成一条可行轨迹。",fontsize=9,color=GREY)
    fig.text(.06,.029,"有功正值为向主网上送、负值为向配网受电。退化截面如实显示为线段/单点；有限方向仅给内近似。",fontsize=9,color=GREY)
    return _save(fig,outdir,"12_24小时能力域_Time_P_Q_完整轨迹")


def _plot_conditional_cost_surface(surface: dict | None, outdir: Path) -> list[str]:
    """图13：同一组真实24h LP网格结果的三维曲面和俯视状态图。"""
    if not surface:return []
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    _set_style()
    k=np.asarray(surface["startup_multipliers"],dtype=float);soc=np.asarray(surface["soc_levels"],dtype=float)
    z=np.asarray(surface["objective_grid"],dtype=float);status=np.asarray(surface["status_grid"],dtype=object)
    valid=(status=="optimal")&np.isfinite(z);z=np.where(valid,z,np.nan);x,y=np.meshgrid(k,soc)
    fig=plt.figure(figsize=(14.5,7.5));gs=fig.add_gridspec(1,2,width_ratios=[1.03,1.0],wspace=.30)
    fig.subplots_adjust(left=.065,right=.88,bottom=.18,top=.80)
    ax=fig.add_subplot(gs[0,0],projection="3d",computed_zorder=False);top=fig.add_subplot(gs[0,1])
    _title(fig,"13  初始条件与24小时条件最优代价：曲面 + 可行区域",
        f"固定{surface['policy_id']}的启动、开关和电池模式｜{z.size}个参数点逐点重解完整{surface['n_steps']}时段LP｜含{surface.get('scenario_count',4)}情景期望费用")
    finite=z[valid]
    if finite.size:
        low,high=float(finite.min()),float(finite.max())
        if np.isclose(low,high):low-=.5;high+=.5
        norm=Normalize(low,high)
        if finite.size>=3:
            ax.plot_surface(x,y,z,cmap="viridis",norm=norm,linewidth=.4,edgecolor="#ffffff88",antialiased=True,alpha=.94,zorder=1)
        ax.scatter(x[valid],y[valid],z[valid],c=z[valid],cmap="viridis",norm=norm,s=9,depthshade=False)
    else:
        norm=Normalize(0,1);ax.text2D(.2,.5,"该网格没有已认证可行最优点",transform=ax.transAxes)
    ax.set(xlabel="启动需求倍率 κ",ylabel="统一初始SOC",zlabel="最小费用 / 千元")
    ax.view_init(elev=27,azim=-129);ax.set_title("(a) 固定恢复方案下的全时域最小费用",loc="left",fontsize=11,pad=12)
    ax.set_box_aspect((1.15,1,.85));ax.tick_params(labelsize=8);ax.xaxis.labelpad=10;ax.yaxis.labelpad=10;ax.zaxis.labelpad=10
    # 灰色只表示该固定方案LP已证不可行；未知和参数超界分别着色。
    state_code=np.zeros(status.shape)
    state_code[status=="infeasible"]=1
    state_code[status=="unknown"]=2
    state_code[status=="invalid_parameter"]=3
    state_cmap=ListedColormap(["#ffffff","#d8dde4","#e9d5ec","#cbddea"])
    top.pcolormesh(x,y,state_code,cmap=state_cmap,vmin=0,vmax=3,shading="nearest",edgecolors="white",linewidth=.35)
    mesh=top.pcolormesh(x,y,np.ma.masked_invalid(z),cmap="viridis",norm=norm,shading="nearest",edgecolors="white",linewidth=.3)
    top.set(xlabel="启动辅助功率轮廓倍率 κ",ylabel="三个微网共同初始SOC")
    top.set_title("(b) 同一采样结果的俯视图",loc="left",fontsize=11,pad=12)
    top.set_xticks(k);top.tick_params(axis="x",rotation=0,labelsize=8);top.set_yticks(soc)
    handles=[]
    for code,label,color in [("infeasible","固定方案LP已证不可行","#d8dde4"),("unknown","未获可靠求解结论","#e9d5ec"),("invalid_parameter","参数超出允许范围","#cbddea")]:
        if np.any(status==code):handles.append(Patch(facecolor=color,label=label))
    ref=surface.get("reference_point")
    if ref is not None:
        a,b,c=ref["startup_multiplier"],ref["soc_initial"],ref["objective"]
        ax.scatter([a],[b],[c],marker="*",s=125,color="#d8415a",edgecolors="white",linewidth=.6,depthshade=False,zorder=5)
        top.scatter([a],[b],marker="*",s=150,color="#d8415a",edgecolors="white",linewidth=.7)
        handles.append(Line2D([0],[0],marker="*",ls="",color="#d8415a",markersize=10,label=f"参考采样点：κ={a:g}，SOC={b:g}"))
    if handles:fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.52,.875),ncol=min(3,len(handles)),fontsize=9)
    if finite.size:
        cbax=fig.add_axes([.905,.22,.018,.51]);fig.colorbar(ScalarMappable(norm=norm,cmap="viridis"),cax=cbax,label="条件最小期望费用 / 千元")
    fig.text(.065,.065,"每格均重新求解完整24h模型；网格间连面只用于显示。灰色不等于所有启动、拓扑和调度方案均不可行。",fontsize=9,color=GREY)
    note="已加入失供损失常数项，颜色对应总期望恢复费用。"
    if not surface.get("startup_selected",True):note+=" 当前参考方案未启机，启动倍率在该方案下不产生作用。"
    fig.text(.065,.033,note,fontsize=9,color=GREY)
    return _save(fig,outdir,"13_24小时条件代价曲面_初始SOC_启动倍率")
