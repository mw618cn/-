#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成与真实计算输出对应的中文论文式Word，以及可继续编辑的公式源。

用法：python scripts/build_paper.py --results results/demo --output 论文式模型与算例分析_可编辑.docx

执行顺序：读取配置/解/校核 → 建立公式目录 → 写研究问题和模型 →
写实际算例表图 → 写边界/替换说明 → 保存docx、LaTeX和公式索引。
公式使用Word原生OMML节点，不使用公式图片，也不声称是MathType OLE对象。
"""
from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
FONT = "微软雅黑"
BLUE = "173C59"
TEAL = "0B7A78"


# 一、OMML构造：每个数学对象均为可编辑结构，而不是导入位图。
def node(tag: str, **attrs: str):
    """建立带命名空间的Word数学XML节点。"""
    n = OxmlElement(tag)
    for k, v in attrs.items():
        n.set(qn(k.replace("__", ":")), str(v))
    return n


def run(value: Any):
    """数学文本片段；Word中可直接点击修改变量或符号。"""
    r = node("m:r")
    props = node("w:rPr")
    fonts = node("w:rFonts", w__ascii="Cambria Math", w__hAnsi="Cambria Math")
    props.append(fonts)
    r.append(props)
    t = node("m:t")
    t.text = str(value)
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    return r


def items(*parts):
    """将字符串和结构公式统一为一组数学节点。"""
    out = []
    for p in parts:
        if isinstance(p, (list, tuple)):
            out.extend(items(*p))
        elif isinstance(p, str) or isinstance(p, (int, float)):
            out.append(run(p))
        else:
            out.append(deepcopy(p))
    return out


def box(tag: str, *parts):
    n = node(tag)
    for p in items(*parts):
        n.append(p)
    return n


def sub(base, index):
    """真正的OMML下标结构。"""
    return box("m:sSub", box("m:e", base), box("m:sub", index))


def sup(base, power):
    """真正的OMML上标结构。"""
    return box("m:sSup", box("m:e", base), box("m:sup", power))


def ss(base, index, power):
    return box("m:sSubSup", box("m:e", base), box("m:sub", index), box("m:sup", power))


def frac(numerator, denominator):
    """真正的OMML分式；分子分母可分别编辑。"""
    return box("m:f", box("m:num", numerator), box("m:den", denominator))


def summ(index, expression, upper=None):
    """带上下限的OMML求和，而不是文本中的Σ字符拼接。"""
    p = node("m:naryPr")
    p.append(node("m:chr", m__val="∑"))
    p.append(node("m:limLoc", m__val="undOvr"))
    if upper is None:
        p.append(node("m:supHide", m__val="1"))
    return box("m:nary", p, box("m:sub", index), box("m:sup", upper or ""), box("m:e", expression))


def norm(expression):
    return items("‖", expression, "‖")


def formulas():
    """公式的论文标识、LaTeX源和Word结构只在此处定义一次。"""
    s = sub
    u = sup
    specs = []

    def add(key, title, latex, *body):
        specs.append({"key": key, "title": title, "latex": latex, "body": items(*body), "number": len(specs) + 1})

    add("time", "时间与场景", r"\mathcal T=\{0,\ldots,N-1\},\quad N=H/\Delta t=96,\quad \sum_{\omega\in\Omega}\pi_\omega=1",
        "T={0,…,N−1},   N=", frac("H", "Δt"), "=96,   ", summ("ω∈Ω", [s("π", "ω")]), "=1")
    add("resilience", "服务恢复比例", r"r_{t\omega}=\frac{\sum_n(p^{c,\mathrm{srv}}_{nt\omega}+p^{g,\mathrm{srv}}_{nt\omega})}{\sum_n(\bar p^c_{nt}+\bar p^g_{nt})}",
        s("r", "tω"), "=", frac(summ("n", [ss("p", "ntω", "c,srv"), "+", ss("p", "ntω", "g,srv")]), summ("n", [ss("p̄", "nt", "c"), "+", ss("p̄", "nt", "g")])))
    add("eens", "缺供电量", r"\mathrm{EENS}=\sum_\omega\pi_\omega\sum_{t,n}\Delta t[(\bar p^c_{nt}-p^{c,\mathrm{srv}}_{nt\omega})+(\bar p^g_{nt}-p^{g,\mathrm{srv}}_{nt\omega})]",
        "EENS=", summ("ω", [s("π", "ω"), summ("t,n", ["Δt[(", ss("p̄", "nt", "c"), "−", ss("p", "ntω", "c,srv"), ")+(", ss("p̄", "nt", "g"), "−", ss("p", "ntω", "g,srv"), ")]"])]))
    add("objective", "期望恢复成本", r"\min J=\sum_\omega\pi_\omega[C^{\mathrm{shed}}_\omega+C^{\mathrm{grid}}_\omega+C^{\mathrm{DG}}_\omega+C^{\mathrm{target}}_\omega+C^{\mathrm{start}}_\omega+C^{\mathrm{ESS}}_\omega+C^{\mathrm{crew}}_\omega]",
        "min J=", summ("ω∈Ω", [s("π", "ω"), "[", ss("C", "ω", "shed"), "+", ss("C", "ω", "grid"), "+", ss("C", "ω", "DG"), "+", ss("C", "ω", "target"), "+", ss("C", "ω", "start"), "+", ss("C", "ω", "ESS"), "+", ss("C", "ω", "crew"), "]"]))
    add("shed", "分级缺供损失", r"C^{\mathrm{shed}}_\omega=\sum_{t,n}\Delta t[v^c_{nt}(\bar p^c_{nt}-p^{c,\mathrm{srv}}_{nt\omega})+v^g_{nt}(\bar p^g_{nt}-p^{g,\mathrm{srv}}_{nt\omega})]",
        ss("C", "ω", "shed"), "=", summ("t,n", ["Δt[", ss("v", "nt", "c"), "(", ss("p̄", "nt", "c"), "−", ss("p", "ntω", "c,srv"), ")+", ss("v", "nt", "g"), "(", ss("p̄", "nt", "g"), "−", ss("p", "ntω", "g,srv"), ")]"]))
    add("branch", "当前动作条件分支", r"J^*_{\rm start}=\min_{x\in\mathcal F,\,y_{0\omega}=1}J(x),\qquad J^*_{\rm wait}=\min_{x\in\mathcal F,\,y_{0\omega}=0}J(x)",
        ss("J", "start", "*"), "=min{J(x):x∈F,", s("y", "0ω"), "=1},   ", ss("J", "wait", "*"), "=min{J(x):x∈F,", s("y", "0ω"), "=0}")
    add("D", "启动与等待的价值差", r"D=J^*_{\rm wait}-J^*_{\rm start};\quad D>0:\mathrm{start},\ D<0:\mathrm{wait}",
        "D=", ss("J", "wait", "*"), "−", ss("J", "start", "*"), ";   D>0:启动,  D<0:等待")
    add("Dbound", "可认证的行动区间", r"L_a\le J_a^*\le U_a\ (a\in\{s,w\}),\quad D\in[L_w-U_s,\ U_w-L_s]",
        s("L", "a"), "≤", ss("J", "a", "*"), "≤", s("U", "a"), " (a∈{s,w}),   D∈[", s("L", "w"), "−", s("U", "s"), ", ", s("U", "w"), "−", s("L", "s"), "]")
    add("serve", "负荷与保护约束", r"0\le p^{k,\mathrm{srv}}_{nt\omega}\le\bar p^k_{nt};\quad p^{c,\mathrm{srv}}_{mt\omega}=\bar p^c_{mt},\ m\in\mathcal M_{\rm protected}",
        "0≤", ss("p", "ntω", "k,srv"), "≤", ss("p̄", "nt", "k"), ";   ", ss("p", "mtω", "c,srv"), "=", ss("p̄", "mt", "c"), ", m∈", s("M", "protected"))
    add("resource", "光伏与可控资源边界", r"0\le p^{\rm PV}_{mt\omega}\le\bar p^{\rm PV}_{mt},\quad 0\le p^{\rm DG}_{mt\omega}\le\bar P_m^{\rm DG}",
        "0≤", ss("p", "mtω", "PV"), "≤", ss("p̄", "mt", "PV"), ",   0≤", ss("p", "mtω", "DG"), "≤", ss("P̄", "m", "DG"))
    add("soc", "储能跨期递推", r"e_{m,t+1,\omega}=e_{mt\omega}+\Delta t(\eta_m^{\rm ch}p^{\rm ch}_{mt\omega}-p^{\rm dis}_{mt\omega}/\eta_m^{\rm dis})",
        s("e", "m,t+1,ω"), "=", s("e", "mtω"), "+Δt(", ss("η", "m", "ch"), ss("p", "mtω", "ch"), "−", frac(ss("p", "mtω", "dis"), ss("η", "m", "dis")), ")")
    add("socbound", "初态、状态边界与末端储备", r"e_{m0\omega}=e_m^0,\quad\underline e_m\le e_{mt\omega}\le\bar e_m,\quad e_{mN\omega}\ge e_m^{\rm reserve}",
        s("e", "m0ω"), "=", ss("e", "m", "0"), ",   ", s("e̲", "m"), "≤", s("e", "mtω"), "≤", s("ē", "m"), ",   ", s("e", "mNω"), "≥", ss("e", "m", "reserve"))
    add("modes", "充放互斥与模式保持", r"0\le p^{\rm ch}_{mt\omega}\le\bar P_m^{\rm ch}b_{m,g(t),\omega},\quad0\le p^{\rm dis}_{mt\omega}\le\bar P_m^{\rm dis}(1-b_{m,g(t),\omega})",
        "0≤", ss("p", "mtω", "ch"), "≤", ss("P̄", "m", "ch"), s("b", "m,g(t),ω"), ",   0≤", ss("p", "mtω", "dis"), "≤", ss("P̄", "m", "dis"), "(1−", s("b", "m,g(t),ω"), ")")
    add("mgp", "微网PCC净外送", r"p^{\uparrow}_{mt\omega}=p^{\rm DG}_{mt\omega}+p^{\rm PV}_{mt\omega}+p^{\rm dis}_{mt\omega}-p^{\rm ch}_{mt\omega}-p^{c,\rm srv}_{mt\omega}-p^{g,\rm srv}_{mt\omega}",
        ss("p", "mtω", "↑"), "=", ss("p", "mtω", "DG"), "+", ss("p", "mtω", "PV"), "+", ss("p", "mtω", "dis"), "−", ss("p", "mtω", "ch"), "−", ss("p", "mtω", "c,srv"), "−", ss("p", "mtω", "g,srv"))
    add("mgq", "微网无功净外送", r"q^{\uparrow}_{mt\omega}=q^{\rm inv}_{mt\omega}-\tan\varphi_m(p^{c,\rm srv}_{mt\omega}+p^{g,\rm srv}_{mt\omega})",
        ss("q", "mtω", "↑"), "=", ss("q", "mtω", "inv"), "−tan", s("φ", "m"), "(", ss("p", "mtω", "c,srv"), "+", ss("p", "mtω", "g,srv"), ")")
    add("inv", "逆变器内接容量多边形", r"\cos\theta_k(p^{\rm PV}_{mt\omega}+p^{\rm dis}_{mt\omega}-p^{\rm ch}_{mt\omega})+\sin\theta_kq^{\rm inv}_{mt\omega}\le S_m^{\rm inv}\cos(\pi/K_f)",
        "cos", s("θ", "k"), "(", ss("p", "mtω", "PV"), "+", ss("p", "mtω", "dis"), "−", ss("p", "mtω", "ch"), ")+sin", s("θ", "k"), ss("q", "mtω", "inv"), "≤", ss("S", "m", "inv"), "cos", frac("π", s("K", "f")))
    add("nodep", "网络有功节点平衡", r"\sum_{e\in\delta^-(n)}P_{et\omega}-\sum_{e\in\delta^+(n)}P_{et\omega}+p^{\rm gen}_{nt\omega}+p^{\rm grid}_{nt\omega}-p^{\rm aux}_{nt\omega}-p^{\rm srv}_{nt\omega}=0",
        summ("e∈δ⁻(n)", s("P", "etω")), "−", summ("e∈δ⁺(n)", s("P", "etω")), "+", ss("p", "ntω", "gen"), "+", ss("p", "ntω", "grid"), "−", ss("p", "ntω", "aux"), "−", ss("p", "ntω", "srv"), "=0")
    add("nodeq", "网络无功节点平衡", r"\sum_{e\in\delta^-(n)}Q_{et\omega}-\sum_{e\in\delta^+(n)}Q_{et\omega}+q^{\rm gen}_{nt\omega}+q^{\rm grid}_{nt\omega}-q^{\rm aux}_{nt\omega}-q^{\rm srv}_{nt\omega}=0",
        summ("e∈δ⁻(n)", s("Q", "etω")), "−", summ("e∈δ⁺(n)", s("Q", "etω")), "+", ss("q", "ntω", "gen"), "+", ss("q", "ntω", "grid"), "−", ss("q", "ntω", "aux"), "−", ss("q", "ntω", "srv"), "=0")
    add("line", "线路状态与容量", r"\cos\theta_kP_{et\omega}+\sin\theta_kQ_{et\omega}\le\bar S_e\cos(\pi/K_f)z_{et\omega},\quad |P_{et\omega}|,|Q_{et\omega}|\le\bar S_ez_{et\omega}",
        "cos", s("θ", "k"), s("P", "etω"), "+sin", s("θ", "k"), s("Q", "etω"), "≤", s("S̄", "e"), "cos", frac("π", s("K", "f")), s("z", "etω"), ";  |P|,|Q|≤", s("S̄", "e"), s("z", "etω"))
    add("voltage", "线性化电压降的状态放松", r"-M_e(1-z_{et\omega})\le v_{jt\omega}-v_{it\omega}+2(r_eP_{et\omega}+x_eQ_{et\omega})/S_{\rm base}\le M_e(1-z_{et\omega})",
        "−", s("M", "e"), "(1−", s("z", "etω"), ")≤", s("v", "jtω"), "−", s("v", "itω"), "+", frac(["2(", s("r", "e"), s("P", "etω"), "+", s("x", "e"), s("Q", "etω"), ")"], s("S", "base")), "≤", s("M", "e"), "(1−", s("z", "etω"), ")")
    add("vbound", "电压范围与参考", r"\underline V^2\le v_{nt\omega}\le\bar V^2,\quad v_{T0,t,\omega}=1;\quad z_{A,t,\omega}=0\Rightarrow v_{M3,t,\omega}=1",
        u("V̲", "2"), "≤", s("v", "ntω"), "≤", u("V̄", "2"), ",   ", s("v", "T0,t,ω"), "=1;   ", s("z", "A,t,ω"), "=0⇒", s("v", "M3,t,ω"), "=1")
    add("topology", "径向恢复结构", r"z_{L0,t,\omega}=1-z_{B,t,\omega},\quad z_{e,t,\omega}\le a_{e,t,\omega},\quad z_{e,t+1,\omega}\ge z_{e,t,\omega}\ (e=A,B)",
        s("z", "L0,t,ω"), "=1−", s("z", "B,t,ω"), ",   ", s("z", "e,t,ω"), "≤", s("a", "e,t,ω"), ",   ", s("z", "e,t+1,ω"), "≥", s("z", "e,t,ω"), " (e=A,B)")
    add("start", "一次性启动选择", r"y_{k\omega}\in\{0,1\},\quad\sum_{k\in\mathcal K}y_{k\omega}\le1,\quad k\Delta t\in\{0,1,\ldots,8\}\ {\rm h}",
        s("y", "kω"), "∈{0,1},   ", summ("k∈K", s("y", "kω")), "≤1,   kΔt∈{0,1,…,8} h")
    add("aux", "完整启动厂用电轮廓", r"p^{\rm aux}_{t\omega}=\sum_{k\in\mathcal K}a^P_{t-k}y_{k\omega},\quad q^{\rm aux}_{t\omega}=\sum_{k\in\mathcal K}a^Q_{t-k}y_{k\omega}",
        ss("p", "tω", "aux"), "=", summ("k∈K", [ss("a", "t−k", "P"), s("y", "kω")]), ",   ", ss("q", "tω", "aux"), "=", summ("k∈K", [ss("a", "t−k", "Q"), s("y", "kω")]))
    add("online", "可用状态与出力", r"o_{t\omega}=\sum_k\mathbf1(t\ge k+L)y_{k\omega},\quad\underline P^{\rm G}o_{t\omega}\le p^{\rm G}_{t\omega}\le\sum_k\bar P_{t-k}^{\rm ramp}y_{k\omega}",
        s("o", "tω"), "=", summ("k", ["1(t≥k+L)", s("y", "kω")]), ";   ", sup("P̲", "G"), s("o", "tω"), "≤", ss("p", "tω", "G"), "≤", summ("k", [ss("P̄", "t−k", "ramp"), s("y", "kω")]))
    add("ramp", "发电机逐步爬坡", r"-R^{\downarrow}\le p^{\rm G}_{t\omega}-p^{\rm G}_{t-1,\omega}\le R^{\uparrow}",
        "−", sup("R", "↓"), "≤", ss("p", "tω", "G"), "−", ss("p", "t−1,ω", "G"), "≤", sup("R", "↑"))
    add("grid", "主网剩余供电能力", r"0\le p^{\rm grid}_{t\omega}\le\alpha\bar P^{\rm grid},\quad|q^{\rm grid}_{t\omega}|\le\alpha\bar Q^{\rm grid},\quad\alpha\in[0,1]",
        "0≤", ss("p", "tω", "grid"), "≤α", sup("P̄", "grid"), ",   |", ss("q", "tω", "grid"), "|≤α", sup("Q̄", "grid"), ",   α∈[0,1]")
    add("handover", "形成可用净出力的诊断时刻", r"\tau^{\rm net}_\omega=\min\{t\Delta t:p^{\rm G}_{t\omega}-p^{\rm aux}_{t\omega}\ge P^{\rm useful}\}",
        ss("τ", "ω", "net"), "=min{tΔt:", ss("p", "tω", "G"), "−", ss("p", "tω", "aux"), "≥", sup("P", "useful"), "}")
    add("crew", "连续实体任务时间", r"T^{\rm next}_{\omega}=T^{\rm end}_{\omega}+\tau_{ij},\quad T^{\rm end}_{\omega}=T^{\rm begin}_{\omega}+d^{\rm task}_{i,\theta_i(\omega)}",
        ss("T", "ω", "next"), "=", ss("T", "ω", "end"), "+", s("τ", "ij"), ",   ", ss("T", "ω", "end"), "=", ss("T", "ω", "begin"), "+", ss("d", "i,θᵢ(ω)", "task"))
    add("history", "由已完成巡检决定的信息历史", r"\mathcal H^\pi_t(\omega)=\{(i,\theta_i(\omega),T^{\rm ins,\pi}_{i\omega}):T^{\rm ins,\pi}_{i\omega}\le t\Delta t\}",
        ss("H", "t", "π"), "(ω)={(i,", s("θ", "i"), "(ω),", ss("T", "iω", "ins,π"), "):", ss("T", "iω", "ins,π"), "≤tΔt}")
    add("NA", "条件非预期性", r"\mathcal H^\pi_t(\omega)=\mathcal H^\pi_t(\omega')\Rightarrow u_{t\omega}=u_{t\omega'}",
        ss("H", "t", "π"), "(ω)=", ss("H", "t", "π"), "(ω′)⇒", s("u", "tω"), "=", s("u", "tω′"))
    add("avail", "物理维修可用性", r"a_{i,t,\omega}=\mathbf1(t\Delta t\ge T^{\rm rep,\pi}_{i\omega}),\quad i\in\{A,B\}",
        s("a", "i,t,ω"), "=1(tΔt≥", ss("T", "iω", "rep,π"), "),   i∈{A,B}")
    add("posterior", "观测后的条件概率", r"\pi_{\omega\mid h_t}=\frac{\pi_\omega\mathbf1[\mathcal H_t(\omega)=h_t]}{\sum_{\nu\in\Omega}\pi_\nu\mathbf1[\mathcal H_t(\nu)=h_t]}",
        s("π", "ω|hₜ"), "=", frac([s("π", "ω"), "1[", s("H", "t"), "(ω)=", s("h", "t"), "]"], summ("ν∈Ω", [s("π", "ν"), "1[", s("H", "t"), "(ν)=", s("h", "t"), "]"])))
    add("menu", "有限因果队伍策略菜单", r"\Pi=\{A,B\}\times\{0,1\}^{\{\rm light,heavy\}},\quad|\Pi|=8,\qquad J_a^*=\min_{\pi\in\Pi}J^*_{a,\pi}",
        "Π={A,B}×", sup("{0,1}", "{light,heavy}"), ",   |Π|=8,   ", ss("J", "a", "*"), "=", s("min", "π∈Π"), ss("J", "a,π", "*"))
    add("projection", "固定离散方案的24小时条件域", r"\mathcal C(z,h)=\{(\mathbf P^\uparrow,\mathbf Q^\uparrow):\exists x\in\mathcal F(z,h),\ (P^\uparrow_{t\omega},Q^\uparrow_{t\omega})=-(P_{L0,t,\omega}+P_{B,t,\omega},Q_{L0,t,\omega}+Q_{B,t,\omega})\}",
        "C(z,h)={(", sup("P", "↑"), ",", sup("Q", "↑"), "):∃x∈F(z,h),   (", ss("P", "tω", "↑"), ",", ss("Q", "tω", "↑"), ")=−(P_L0+P_B,Q_L0+Q_B)}")
    add("support", "指定时段的支持函数查询", r"\sigma_{\mathcal C,t}(d)=\max_{x\in\mathcal F(z,h)}d_P P^\uparrow_{t\omega}+d_Q Q^\uparrow_{t\omega}",
        s("σ", "C,t"), "(d)=", s("max", "x∈F(z,h)"), "[", s("d", "P"), ss("P", "tω", "↑"), "+", s("d", "Q"), ss("Q", "tω", "↑"), "]")
    add("inner", "可行见证的条件内近似", r"\mathcal C^{\rm in}=\operatorname{conv}\{w^1,\ldots,w^K\}\subseteq\mathcal C(z,h)\quad\text{only for one common fixed-discrete convex set}",
        sup("C", "in"), "=conv{", sup("w", "1"), ",…," ,sup("w", "K"), "}⊆C(z,h)  （同一固定离散凸集合）")
    add("decomp", "固定整条接口轨迹的分解校核", r"\operatorname{find}\ x\in\mathcal F(z,h)\quad\operatorname{s.t.}\ (\mathbf P^\uparrow,\mathbf Q^\uparrow)=(\widehat{\mathbf P}^\uparrow,\widehat{\mathbf Q}^\uparrow)",
        "find x∈F(z,h),   s.t. (", sup("P", "↑"), ",", sup("Q", "↑"), ")=(", sup("P̂", "↑"), ",", sup("Q̂", "↑"), ")")
    add("value", "能力与代价联合表示", r"V(w;z,h)=\min\{J(x):x\in\mathcal F(z,h),\ A_{\rm PCC}x=w\}",
        "V(w;z,h)=min{J(x):x∈F(z,h), ", s("A", "PCC"), "x=w}")
    add("menubound", "菜单总体上下界", r"L_a=\min_{\pi\in\Pi}L_{a,\pi},\quad U_a=\min_{\pi\in\Pi}U_{a,\pi},\quad L_a\le J_a^*\le U_a",
        s("L", "a"), "=", s("min", "π∈Π"), s("L", "a,π"), ",   ", s("U", "a"), "=", s("min", "π∈Π"), s("U", "a,π"), ",   ", s("L", "a"), "≤", ss("J", "a", "*"), "≤", s("U", "a"))
    add("optional", "可选协同的集合包含关系", r"\mathcal F_{\rm no\mbox{-}coop}\subseteq\mathcal F_{\rm optional\mbox{-}coop}\Rightarrow J^*_{\rm optional\mbox{-}coop}\le J^*_{\rm no\mbox{-}coop}",
        s("F", "no-coop"), "⊆", s("F", "optional-coop"), "⇒", ss("J", "optional-coop", "*"), "≤", ss("J", "no-coop", "*"))
    add("rolling", "预设重算时刻之间的状态继承", r"s_{\tau_{r+1}}=F(s_{\tau_r},u_{\tau_r:\tau_{r+1}-1},o_{\tau_r+1:\tau_{r+1}}),\quad\mathcal T_r=\{\tau_r,\ldots,N-1\}",
        s("s", "τᵣ₊₁"), "=F(", s("s", "τᵣ"), ",", s("u", "τᵣ:τᵣ₊₁−1"), ",", s("o", "τᵣ+1:τᵣ₊₁"), "),   ", s("T", "r"), "={τᵣ,…,N−1}")
    return specs


# 二、文字和表格样式：保证中文排版、表头与公式编号可直接修改。
def set_font(r, size=None, bold=None, color=None):
    r.font.name = FONT
    r._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), FONT)
    if size is not None:
        r.font.size = Pt(size)
    if bold is not None:
        r.bold = bold
    if color:
        r.font.color.rgb = RGBColor.from_string(color)


def setup_styles(doc):
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.0)
    sec.bottom_margin = Cm(1.8)
    sec.left_margin = Cm(2.1)
    sec.right_margin = Cm(2.0)
    for name in ("Normal", "Body Text", "Caption", "Title", "Subtitle", "Heading 1", "Heading 2", "Heading 3"):
        style = doc.styles[name]
        style.font.name = FONT
        style._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), FONT)
    doc.styles["Normal"].font.size = Pt(10.5)
    doc.styles["Normal"].paragraph_format.line_spacing = 1.35
    doc.styles["Normal"].paragraph_format.space_after = Pt(5)
    for name, size in (("Heading 1", 16), ("Heading 2", 12.5), ("Heading 3", 11)):
        st = doc.styles[name]
        st.font.size = Pt(size)
        st.font.color.rgb = RGBColor.from_string(BLUE)
        st.paragraph_format.keep_with_next = True
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(footer.add_run("主配微协同韧性恢复 · 可编辑模型与算例说明  |  "), 8, color="64748B")
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)


def p(doc, text, bold=False):
    q = doc.add_paragraph()
    set_font(q.add_run(text), bold=bold)
    return q


def table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(headers):
        cell = t.rows[0].cells[j]
        shade = OxmlElement("w:shd")
        shade.set(qn("w:fill"), BLUE)
        cell._tc.get_or_add_tcPr().append(shade)
        set_font(cell.paragraphs[0].add_run(str(h)), 9, True, "FFFFFF")
    for row in rows:
        cells = t.add_row().cells
        no_split = OxmlElement("w:cantSplit")
        t.rows[-1]._tr.get_or_add_trPr().append(no_split)
        for j, value in enumerate(row):
            cells[j].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            q = cells[j].paragraphs[0]
            q.paragraph_format.space_after = Pt(3)
            q.paragraph_format.space_before = Pt(3)
            set_font(q.add_run(str(value)), 8.5)
    # 重复表头，跨页阅读时仍能解释每列。
    repeat = OxmlElement("w:tblHeader")
    t.rows[0]._tr.get_or_add_trPr().append(repeat)
    if widths:
        for row in t.rows:
            for cell, width in zip(row.cells, widths):
                cell.width = Cm(width)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return t


def load_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(v, digits=4):
    if v is None:
        return "未记录"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, (int, float)):
        return f"{v:.{digits}f}" if math.isfinite(v) else str(v)
    if isinstance(v, list):
        return " / ".join(fmt(x, digits) for x in v)
    return str(v)


def flatten(obj, prefix=""):
    """保存原配置路径，便于从论文参数表定位到JSON进行替换。"""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        for i, v in enumerate(obj):
            out.extend(flatten(v, f"{prefix}[{i}]"))
    else:
        txt = json.dumps(obj, ensure_ascii=False) if isinstance(obj, list) else str(obj)
        if len(txt) <= 180:
            out.append((prefix, txt))
    return out


def add_figure(doc, path, caption):
    if not path.exists():
        return False
    q = doc.add_paragraph()
    q.alignment = WD_ALIGN_PARAGRAPH.CENTER
    q.add_run().add_picture(str(path), width=Cm(16.2))
    q = doc.add_paragraph(style="Caption")
    q.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(q.add_run(caption), 9, color="475569")
    return True


def result_value(result, key, default=None):
    return result.get("summary", {}).get(key, default)


def build(results: Path, output: Path):
    """主流程：每个结果数值均来自磁盘JSON，未计算项目如实标记。"""
    cfg = load_json(results / "config_resolved.json")
    sol = load_json(results / "solution.json")
    comparison = load_json(results / "branch_comparison.json")
    validation = load_json(results / "validation.json")
    rolling = load_json(results / "rolling" / "rolling_replay.json")
    baseline = load_json(results / "solution_no_upward.json")
    wait_solution = load_json(results / "solution_wait.json")
    external_results = results.parent / "external_support"
    external_comparison = load_json(external_results / "branch_comparison.json")
    external_config = load_json(external_results / "config_resolved.json")
    capability_path = next(iter(sorted((results / "capability").glob("*.json"))), None) if (results / "capability").exists() else None
    capability = load_json(capability_path) if capability_path else {}
    specs = formulas()
    spec_by_key = {x["key"]: x for x in specs}
    used_specs = []
    doc = Document()
    setup_styles(doc)

    def heading(text, level=1):
        doc.add_heading(text, level)

    def eq(key):
        entry = spec_by_key[key]
        if entry not in used_specs:
            entry["number"] = len(used_specs) + 1
            used_specs.append(entry)
        q = doc.add_paragraph()
        q.paragraph_format.space_before = Pt(4)
        q.paragraph_format.space_after = Pt(5)
        q.paragraph_format.keep_together = True
        math_obj = node("m:oMath")
        for n in entry["body"]:
            math_obj.append(deepcopy(n))
        q._p.append(math_obj)
        set_font(q.add_run(f"    ({entry['number']})"), 9)
        return entry["number"]

    # 三、论文正文：先阐明研究问题与贡献，再给完整约束和求解依据。
    q = doc.add_paragraph()
    q.paragraph_format.space_before = Pt(46)
    set_font(q.add_run("主配微协同韧性恢复"), 27, True, BLUE)
    q = doc.add_paragraph()
    set_font(q.add_run("启动—等待价值、24小时条件能力域与巡检驱动恢复"), 17, True, TEAL)
    p(doc, "论文式模型、求解逻辑与可复现实验说明", True)
    p(doc, "配套文件：完整Python研究原型、集中配置参数、实际计算结果与图件、OMML可编辑公式、同编号LaTeX公式源。")
    p(doc, "版本说明：设备参数是演示参数；网络、机组和队伍规模用于验证逻辑。结果应在替换工程数据和补充对标后形成正式论文结论。")
    p(doc, "公式格式：Word原生OMML，可直接进入公式编辑；同时提供LaTeX源。此文件不包含原生MathType OLE对象。")
    p(doc, f"构建时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}  |  结果目录：{results.relative_to(ROOT) if results.is_relative_to(ROOT) else results}")
    doc.add_page_break()

    heading("摘要")
    p(doc, "针对灾害后主网供电受限、配电通道损坏和微网能源有限同时存在的恢复问题，建立主网—配网—微网协同的24小时多场景恢复模型。模型在保障微网受保护关键服务的条件下，联合考虑上级机组外部启动、储能跨期调度、巡检信息揭示、非抢占抢修和网络重构。通过当前启动与当前等待两个条件分支区分物理可行性与行动价值；在固定离散恢复方案后，以全时域线性规划查询条件接口能力并保存可实施轨迹见证。利用实际观测历史约束未区分场景的控制一致性，防止恢复策略提前使用未来损坏信息。研究原型采用有限因果队伍策略枚举和SciPy/HiGHS随机混合整数线性规划，输出缺供电量、服务恢复曲线、启动过程和24小时资源轨迹，并保留求解上下界与校核记录。模型用于形成可复核研究基准，其结果尚不构成真实系统的暂态黑启动安全证明。")
    p(doc, "关键词：供电韧性；主配微协同；外部启动；多时段可调度域；巡检信息；条件非预期性")

    heading("1 研究问题与贡献定位")
    heading("1.1 从“能否启动”到“当前是否值得启动”", 2)
    p(doc, "微网群的瞬时功率足够，并不意味着可以在启动全过程中持续供电；能完成启动，也不意味着当前启动在有限能源与未知修复时长下更合适。合理比较必须同时保留本地关键服务、全时域储能状态、网络通道和后续行动机会。等待分支在当前时段不启动，以后仍可根据巡检结果选择启动。")
    p(doc, "用α表示主网剩余供电比例：α=0是外部供电完全丧失的特例，α>0表示仍有有限支撑。α并非通过人为指定“某小时主网自动恢复”实现；被启动机组何时形成出力由启动选择、阶段需求和爬坡决定。本文不预设存在唯一、单调的临界α。")
    table(doc, ["候选贡献", "方法与承担的工作", "必须由结果验证的增量"], [
        ["① 启动承诺与等待价值", "同一24小时模型的当前动作条件分支；物理可行性与价值差D；上下界认证", "在哪些状态下应等待，以及信息/资源机会成本如何改变当前行动"],
        ["② 多时段条件能力与分解", "保留SOC、网络和信息约束的支持函数查询；完整见证轨迹；固定整条接口轨迹再分解", "有限能力信息能否支持正确判定；表示的保守性、执行性与代价差"],
        ["③ 巡检驱动恢复协同", "实体巡检完成后揭示损坏；观测历史条件非预期性；8个因果队伍策略", "检查什么、何时检查如何同时改变信息、修复时刻与启动价值"],
    ])
    p(doc, "三点构成一个闭环：巡检与维修改变信息和拓扑，更新多时段条件能力；能力与代价支撑启动/等待判断；执行后由新观测和实测SOC更新下一次判断。原型的强项是模型与计算结果可复核；自适应域压缩及其决策误差理论仍需在该基准上继续研发，不能把条件LP图直接写成已完成的通用新算法。")

    heading("1.2 主、配、微的角色及协同载体", 2)
    table(doc, ["角色", "本层决策", "跨层耦合"], [
        ["TSO/主网", "当前启动、候选启动时刻、启动阶段与净发电爬坡", "向DSO提出连续P/Q支撑需求，启动后经可用通道回馈负荷"],
        ["DSO/配网", "巡检抢修、A/B开关、弱/强通道切换及网络安全", "接收微网PCC能力，约束可送达的启动支撑，并反馈信息和修复时刻"],
        ["MGO/微网", "DG/PV、储能充放、SOC、受保护服务及PCC外送", "对外支撑与本地服务竞争有限功率/能量，接受完整可分解接口轨迹"],
    ])
    p(doc, "协同不是三个框之间的箭头，而是同一节点平衡中互相影响的P/Q、同一启动任务的持续需求、同一条线路的可用状态，以及共享观测历史。PCC净外送已扣除微网自身负荷，不能拿内部总发电替代上送能力。")

    heading("1.3 与已核对文献的关系", 2)
    p(doc, "杨明通等已有多时段SOC、递归鲁棒能力域与后向递推，因此不能声称既有MPAR均为静态单时段。王炜歆等已有输配协同启动、有限能源、关键负荷与接力支撑，因此外部启动和持续供能本身不是本文的新颖性。Zhang等已有能源耗尽与恢复后再切负荷的随机建模。本文应把差异落在条件信息、当前启动承诺价值以及服务该行动判定的可实施能力表示，并通过进一步对标确认新颖性。")

    heading("2 系统、场景与建模边界")
    heading("2.1 网络、时间和参数", 2)
    p(doc, "模型包含T0、T1、D0、M1、M2、M3共6个节点，T0为残余外部电源与目标机组接入点，T1、D0分别代表主网和配网负荷，M1—M3为微网。线路依次为T0—T1、T1—D0弱线L0、同走廊修复强线B、D0—M1、D0—M2和受损线A（D0—M3）。L0与B互斥投入，A断开时M3以已具备的本地构网条件运行。不同电压层级的理想变比已在标幺值中处理。")
    eq("time")
    p(doc, "默认连续调度步长15分钟、时域24小时，共96步。启动候选采用整小时网格且默认在前8小时内允许，共9个起点；储能充放模式默认整小时保持，连续功率与SOC仍每15分钟优化。这两类决策网格均可配置，模式保持限制是降低整数规模的保守控制分辨率，可改为15分钟。有限的启动候选和模式网格均属于所求模型的明确范围。")
    table(doc, ["符号", "含义及单位"], [
        ["t、N、Δt；ω、πω", "调度时段、时段数、步长（h）；损坏场景与先验概率"],
        ["m、n、e；k", "微网、网络节点、线路；启动候选/容量多边形面索引由上下文区分"],
        ["p̄c、p̄g；pc,srv、pg,srv", "关键/普通需求及实际服务功率（MW）"],
        ["e、pch、pdis、b", "储能能量（MWh）、充/放电功率（MW）、模式二进制变量"],
        ["pPV、pDG、qinv", "光伏、DG有功（MW）与逆变器无功（Mvar）"],
        ["P、Q；v；z、a", "线路有功/无功、节点平方电压（pu²）、闭合状态与修复可用性"],
        ["y、o、paux、pG", "启动选择、在线状态、阶段厂用电与目标机组输出"],
        ["α、π、H、u", "剩余外部支撑比例、队伍策略、观测历史、当时控制动作"],
        ["J、D；L、U", "恢复成本及行动价值差（千元）；有效下界与可行上界"],
        ["C(z,h)、w、V(w)", "系统条件接口域、完整P/Q轨迹与该轨迹的实现成本"],
    ])
    heading("2.2 隐蔽损坏、巡检和合理假设", 2)
    p(doc, "两条线路A、B各有轻/重损坏，组合为4个场景。其概率由JSON显式设置，可以包含相关性。PV与负荷在本版为确定性曲线；不把已经存在的损坏四场景夸大为气象、负荷、维修和通信全部联合不确定性。损坏类型仅在实体队伍完成相应巡检时揭示，维修时长取决于该类型。")
    table(doc, ["保留的机制", "本版明确假设"], [
        ["关键服务与跨期能源", "微网受保护关键负荷每时段强制满足；其余负荷允许分级削减；有末端储备"],
        ["外部启动与爬坡", "一次性、无中止/重启的固定阶段轮廓；参数为示例，不对应某真实大型火电机组"],
        ["物理巡检和修复", "一支队伍、两受损点、准确巡检、非抢占任务、旅行时间矩阵"],
        ["网络安全近似", "无损线性化P/Q潮流、平方电压、多边形视在容量；未验证暂态、励磁、频率和保护"],
        ["初始状态", "微网已具备自启动/构网及初始走廊建压的必要条件，求解其后的连续恢复调度"],
        ["信息交换", "集中求解、信息按巡检历史揭示；通信中断、攻击与异步ADMM不在本版约束内"],
    ])

    heading("3 服务韧性、目标与启动—等待判定")
    heading("3.1 韧性落实到终端服务", 2)
    p(doc, "用r表示终端负荷的实际服务比例，而非可用发电容量比例。普通EENS按未服务电量统计；加权EENS使用明确的负荷权重。恢复曲线面积、24小时缺供电量和关键服务连续满足共同反映韧性。机组启动概率与可用净出力时刻用于解释机制，不能取代终端服务评价。")
    eq("resilience")
    eq("eens")
    p(doc, "受保护负荷在整个时域中不得失供；其他“优先负荷”通过较高缺供成本体现优先级。两者必须区分，不能将所有优先负荷都称为硬保证。配置可以让单位缺供损失随灾后经历时间增长；这属于外生时间权重，不等同于逐负荷内生连续停电时长模型。")
    eq("serve")
    heading("3.2 一致的期望恢复成本", 2)
    eq("objective")
    eq("shed")
    p(doc, "各成本统一为千元：功率MW乘步长小时得到MWh，再乘千元/MWh；一次启动成本单位为千元，不再乘时长。DG、目标机组、购电与储能吞吐成本均从配置读取，队伍成本由旅行/检查/修复实际占用时间计算。末端储备为硬约束；若将末端能量价值加入目标，必须在配置与两个比较分支中同时使用。代码中的目标常数保留完整缺供损失基数，因此报告的是完整成本而非仅变量部分。")
    heading("3.3 当前动作的公平比较及D区间", 2)
    eq("branch")
    eq("D")
    p(doc, "两个分支共享同一初始状态、同一损坏先验、同一队伍策略菜单、同一目标及信息规则。当前启动要求t=0开始任务；当前等待禁止t=0开始，但以后仍可启动。若当前启动不可行而等待可行，应先报告可行性选择；两个分支都不可行时不计算无穷大之差。")
    eq("Dbound")
    p(doc, "在时间限制下，求解器可能只给出可行策略。可行目标为上界，合法对偶界为下界。D区间全正才能认证当前启动更优，全负才能认证等待更优；区间包含0时报告“尚不能认证”，或继续计算。只有两个可行解的成本差不能当成真实最优D。有限菜单内的结论也不能外推为所有车辆路径策略的结论。")
    p(doc, "SciPy/HiGHS的相对MIP间隙针对传入求解器的变量目标cᵀx计算；本模型另有缺供基准常数C0，因此它与原费用J=cᵀx+C0的相对差通常不同。状态0代表达到求解器容差，并不表示数学上零间隙。论文应优先列原费用的上下界、绝对差及(U−L)/max(|U|,小正数)，避免把0.5%的求解器设定直接称为原总费用0.5%精度。")
    eq("optional")
    p(doc, "若允许协同模型包含完全不协同方案，且资源、约束和信息条件相同，其最优成本不会比不协同更高。反直觉结果可以是“强制现在启动劣于等待”，不能将其改写为“可选最优协同必然有害”。")

    heading("4 微网多时段资源与网络约束")
    heading("4.1 资源、SOC和PCC", 2)
    eq("resource")
    eq("soc")
    eq("socbound")
    p(doc, "SOC递推包含充放电损耗，能量从t=0一直连接到t=N。初态、上下限与末端储备同时约束所有损坏场景。末端储备避免为提高24小时内指标而无条件耗尽储能，但其数值必须来自后续供电需求或敏感性分析，不能通过任意提高储备制造等待优势。")
    eq("modes")
    p(doc, "b是充放模式二进制变量，g(t)把15分钟时段映射到配置的模式保持段。固定离散方案构造条件能力域时，必须把b一并固定，才能得到单个连续凸集合；不同模式最优解的凸组合通常不能直接用于原混合整数系统。")
    eq("mgp")
    eq("mgq")
    eq("inv")
    p(doc, "逆变器的P项是PV加放电减充电，Q项来自逆变器无功调节；本版DG按单位功率因数运行。容量多边形取圆的内接近似，面法向角和右端cos(π/Kf)必须匹配。增大面数可以改善近似精度，也增加线性约束数量。")
    heading("4.2 潮流、电压、容量与重构", 2)
    eq("nodep")
    eq("nodeq")
    p(doc, "线路统一按T0→T1、T1→D0、D0→微网定义正方向。因此主配接口向上支撑等于−(P_L0+P_B)，微网净外送等于相应D0→Mm支路潮流的相反数。接口指标符号在代码、CSV和图中保持一致。")
    eq("line")
    eq("voltage")
    eq("vbound")
    p(doc, "v表示电压幅值的平方；r、x采用与Sbase一致的标幺值。断开线路容量为零，电压关系用M放松；M应足够覆盖允许电压差且避免过大。T0提供参考电压，孤岛M3在A断开时采用本地参考。此处理在优化层面规定电压可行性，不推导孤岛的真实频率动态。节点平衡中的gen是资源净注入的简写，在微网节点包括DG、PV、放电和充电；源程序将这些项分别列入矩阵，不重复计算PCC外送。")
    eq("topology")
    p(doc, "A、B只有在维修完成后才能闭合；闭合后保持恢复状态。B闭合同时退出弱线L0，因此不会产生并联闭环。这个特定六节点结构通过构造保持径向，不等同于一般配电网径向约束算法。修复时间来自队伍日程，通电开关由MILP选择。")

    heading("5 外部启动、承诺与可用净出力")
    eq("start")
    eq("aux")
    p(doc, "启动变量只选择一次候选起点，所有启动阶段厂用电由同一变量卷积生成。因此模型必须连续供应完整阶段轮廓，而不是只检查某时刻功率峰值。厂用无功同样进入节点平衡。阶段参数需要由真实设备资料替换；没有经过工程匹配的百kW级微网不能直接声称可以启动数百MW火电机组。")
    eq("online")
    eq("ramp")
    eq("grid")
    p(doc, "α仅缩放初始可用外部电源，不随任意预设时刻跳变。源程序取Qgrid上限为外部有功参考容量的一半，并随α缩放。目标机组完成启动后按可用状态和爬坡上限发电，其输出经同一网络传递；机组P/Q另受在线无功边界与同形式内接容量多边形约束。无启动时，目标机组有功无功都为零。默认最大2.4MW是虚拟可控演示机组，并不代表大型火电机组的真实启动能力。一次性不可中止的假设意味着启动后阶段需求必须一直承担；若扩展为允许中止，应增设状态、再启动条件及损失代价。")
    eq("handover")
    p(doc, "τnet报告目标机组首次达到配置阈值的可用净出力时刻。它只是诊断指标；送达具体负荷仍取决于同一时刻的潮流与线路状态。不能把“机组开始发电”直接等同于所有微网已经解除全部支撑责任，也不能把该时刻当成暂态安全认证。")

    heading("6 巡检、抢修与信息物理耦合")
    heading("6.1 为什么不用额外的信息奖励", 2)
    p(doc, "巡检的价值来自它允许改变后续启动、储能和抢修行动，并减少实际缺供和恢复成本，而不是单独最大化信息熵。损坏很不确定的线路未必值得先查；某个检查也可能因为占用队伍而推迟关键通道修复。原型通过同一成本目标比较实体策略，避免给巡检加入任意权重的信息奖励。")
    eq("crew")
    eq("history")
    eq("NA")
    p(doc, "u包括当时需要作出的启动、开关、资源调度及其他场景控制。只有当已经完成的巡检使两个场景的观测历史不同，控制才可以分叉。任务连续执行是物理承诺，条件非预期性是信息约束，二者作用不同，不能用一种代替另一种。")
    p(doc, "源程序没有对所有SOC再重复写场景等式：在相同初态下，未区分场景此前的充放控制逐时相同，SOC递推按归纳自然得到相同状态。若后续加入不同的未观测外生扰动，应重新检查状态观测和可执行性定义，不能直接沿用这个推论。")
    eq("avail")
    eq("posterior")
    p(doc, "离线场景树求解用先验概率与条件非预期性表达所有可执行分支；观测发生后做真正滚动重求时，再按后验缩小场景集合。只对已经观测到的内容更新，不能读取未来场景标签。完美信息版本移除非预期性，只用于刻画不现实的信息上限。")
    heading("6.2 可落地的有限策略菜单", 2)
    eq("menu")
    p(doc, "先选A或B作为首次巡检点。首次结果为轻或重时分别选择：先修当前点再检查/修复另一点，或先去另一点检查/修复再返回。2×2×2形成8个因果策略。旅行、检查与修复按配置网格对齐；每个策略在4种损坏组合下产生真实任务日程。选择策略时共同评估其信息与电力后果。")
    p(doc, "该方法的可验证范围是此8策略菜单。它没有任意多队伍、道路拥堵、连续出发时间或随机排队服务；如果引入排队或马尔可夫过程，需要增加真实等待机制、服务分布及观测规则，不能只给已知维修时长贴上随机服务名称。后续可用列生成或混合整数路径模型替换菜单生成器，保留信息—电力约束接口。")

    heading("7 多时段条件能力域、投影与分解")
    eq("projection")
    p(doc, "z代表固定的启动、开关和充放模式等离散选择，h代表所采用的信息条件；F(z,h)仍保留4场景的完整24小时资源、网络及条件非预期性约束。固定整数形成的LP是原系统的条件可行子集，不能当成包含所有离散恢复方案的完整域。")
    eq("support")
    p(doc, "选某一时段的P/Q方向优化时，其他95个时段仍保留在LP中。得到的二维点是该条件全时域集合的投影，旁边应保存完整资源轨迹作为可行见证。展示多个时间切片可观察能力演变；这些切片的笛卡尔积一般大于真实轨迹集合，不能任意逐时拼接。")
    eq("inner")
    p(doc, "内近似的依据：若每个见证xk属于同一个固定离散凸集合F(z,h)，任意非负且和为1的权重λk形成的Σλkxk仍满足所有线性约束；接口线性映射后得到Σλkwk。因此可行点凸包位于条件投影域内。离散方案或信息条件不同时，这个证明不再直接成立。")
    eq("decomp")
    p(doc, "分解校核固定一整条接口P/Q轨迹，再求微网内部资源和网络变量；若LP可行，才获得该条件模型下可执行的指令。单独把各时段最大外送相连，或把不同场景的独立最优轨迹拼接，均不提供这种保证。保存的资源轨迹可进一步做独立功率平衡、SOC、线路和信息一致性校核。")
    eq("value")
    p(doc, "能力—代价表示不仅回答某轨迹能否实现，也回答内部实现成本。当前原型用实际LP查询和见证形成诊断，MILP仍直接保留全部约束。真正的决策导向压缩需要进一步规定合法下界、可行上界、查询方向选择和停止准则，并用完整MILP验证D判定误差；不应把目前有限切片图等同于已经完成该理论贡献。")

    heading("8 求解流程、ADMM位置与滚动扩展")
    table(doc, ["步骤", "执行内容", "输出/正确性依据"], [
        ["1 参数与曲线", "读取JSON；检验概率、单位、边界和网格；生成96步PV/负荷", "config_resolved.json，明确全部演示值"],
        ["2 场景与实体策略", "构造4损坏场景、8因果队伍策略及其任务日程", "观测时刻、维修时刻、历史分组"],
        ["3 随机MILP", "对每个策略建立同一电力模型；保留条件非预期性", "可行上界、对偶下界、间隙、用时"],
        ["4 当前动作比较", "分别限制当前启动/等待；按相同菜单汇总上下界", "D可行解差与认证区间分别报告"],
        ["5 轨迹与域", "固定可行整数方案；全时域LP支持查询；固定整条轨迹再分解", "条件域、完整见证、残差"],
        ["6 校核与图件", "约束残差、SOC、关键服务和信息一致性；出24小时结果", "validation.json、CSV、PNG/SVG/PDF"],
        ["7 执行与观测更新", "固定已选队伍策略，执行前缀后按完成巡检更新后验并重求未来功率", "滚动JSON/CSV、完整执行路径、逐轮校核与回退记录"],
    ])
    eq("menubound")
    p(doc, "菜单界的依据：每个候选都有L(a,π)≤J*(a,π)≤U(a,π)，对同一个有限菜单分别取最小值，仍得到min L≤min J*≤min U。这是菜单内的界；如果漏掉未搜索的候选而未提供合法下界，则不能对更大策略集合认证。")
    p(doc, "MILP用SciPy调用HiGHS。菜单总体下界必须取各候选下界的最小值；只取最佳可行候选自己的下界通常不能界定整个菜单最优值。未求解候选若无合法下界，不能对完整菜单作认证。基准、动作分支和能力查询均须记录实际求解状态，超时保留的可行解不能写成已证明全局最优。")
    heading("8.1 ADMM是否继续使用", 2)
    p(doc, "默认原型使用集中MILP作为可核查基准，因此无需ADMM。若后续需要隐私或分布式计算，可先固定外层启动、维修策略、开关和储能模式，将连续凸子问题按TSO/DSO/MGO拆分，以PCC有功无功及必要状态作为一致性变量。ADMM只承担这个固定离散连续问题的数值协调，其收敛条件取决于凸性、可行性和通信假设。把二进制启动、拓扑和路径直接塞入ADMM，不能自动得到全局最优或稳定收敛保证。")
    heading("8.2 离线因果策略与真实滚动的区别", 2)
    eq("rolling")
    p(doc, "完整场景树求解产生每种可能观测下的可执行策略，并非已经完成逐步闭环仿真。真实滚动需要先执行至下一重算时刻，再按实测SOC、任务位置/剩余时长和已揭示损坏更新状态，保留启动与维修承诺，在剩余[t,24h]重求。研究终点保持原24小时，不能每次无说明地延长为新的24小时。式中τr为重算时刻；两次重算之间按15分钟应用上轮已经求出的因果计划，而非默认每15分钟都重新优化。")
    p(doc, "配套rolling.py实现固定已选抢修策略的功率调度滚动回放，默认演示采用8个重算时刻，可通过decision_hours配置更密的重算网格。重算时只根据已完成巡检筛除不兼容场景并更新后验，固定t<当前时刻的控制和t≤当前时刻的SOC；已开始的模式块与启动承诺不得撤销。原始开关事件网格保留，终点仍为原24小时。此实现不会在每轮重新选择队伍路线；若本轮没有通过校核的可行解，则继承并重新校核上一轮剩余计划。")

    # 四、算例只读本次运行，不预填“预期改善率”。
    heading("9 演示算例与实际计算结果")
    p(doc, "以下表格和插图直接读取随包结果。所有数值应理解为本配置、本信息规则、本策略菜单与已报告求解精度下的演示结果；不代表真实工程性能。更换参数后重新运行即可替换此章，公式与配置路径保持一致。")
    heading("9.1 参数设置与替换位置", 2)
    if cfg:
        preferred = []
        other = []
        for key, value in flatten(cfg):
            row = [key, value]
            if any(w in key.lower() for w in ("solver", "plot", "output")):
                other.append(row)
            else:
                preferred.append(row)
        table(doc, ["配置路径（可直接定位JSON）", "本次演示值"], preferred)
        if other:
            table(doc, ["数值求解与输出配置", "设置"], other)
    else:
        p(doc, "当前目录没有config_resolved.json，参数表不填虚构值。先运行代码，再以--results指向结果目录重建本文档。")
    p(doc, "参数表保留原JSON路径而非重新命名，便于把设备资料和线路数据逐项替换。改变步长后需同时核对旅行/巡检/维修及启动阶段对齐规则；改变成本单位后需同步修改VOLL、燃料、购电、储能吞吐和一次启动成本。")
    heading("9.2 策略与求解质量", 2)
    solver = sol.get("solver", {})
    if sol:
        table(doc, ["项目", "实际记录"], [
            ["策略", sol.get("policy_label", sol.get("policy_id", "未记录"))],
            ["动作分支/信息", f"{sol.get('branch','未记录')} / {sol.get('information','未记录')}"],
            ["状态", solver.get("message", solver.get("status", "未记录"))],
            ["可行目标/下界", f"{fmt(solver.get('objective'))} / {fmt(solver.get('lower_bound'))}"],
            ["相对MIP间隙", fmt(solver.get("mip_gap"), 6)],
            ["求解时间（s）", fmt(solver.get("solve_seconds"), 2)],
        ])
        upper, lower = solver.get("objective"), solver.get("lower_bound")
        if isinstance(upper, (int, float)) and isinstance(lower, (int, float)):
            p(doc, f"所选策略自身的原费用绝对界差为{fmt(upper-lower)}千元，按可行费用归一的差为{fmt(100*(upper-lower)/max(abs(upper),1e-9),2)}%。这与HiGHS表中的变量目标相对间隙口径不同。整个菜单的下界还需读取下表分支聚合值。")
    else:
        p(doc, "没有solution.json，本章不报告目标值或性能改善率。")
    metrics = [
        ("期望成本", "expected_cost"), ("期望缺供电量（MWh）", "expected_eens_mwh"),
        ("期望加权缺供电量", "expected_weighted_eens_mwh"),
        ("期望已服务电量（MWh）", "expected_restored_energy_mwh"),
        ("启动概率", "start_probability"), ("达到可用净出力概率", "handover_probability"),
        ("期望可用净出力时刻（h）", "expected_handover_hour"),
    ]
    if sol.get("summary"):
        table(doc, ["韧性/机制指标", "实际值"], [(label, fmt(result_value(sol, key))) for label, key in metrics])
        p(doc, "缺供电量与已服务电量描述终端韧性，启动/可用净出力描述恢复机制。若期望时刻只对成功达到阈值的场景条件平均，应同时读取其概率；未达到阈值的场景不能默认为0小时。")
        p(doc, "这里的“启动概率”是先验场景概率加权的启动选择比例，“达到可用净出力概率”是简化模型内的阈值统计；二者都不是设备实测启动成功率，也没有把暂态故障和模型误差作为额外随机事件。")
        scene_eens = result_value(sol, "scenario_eens_mwh", [])
        scene_starts = result_value(sol, "scenario_start_hour", [])
        scene_net = result_value(sol, "scenario_handover_hour", [])
        scene_rows = []
        for i, name in enumerate(sol.get("scenario_names", [])):
            sched = sol.get("schedules", [])[i] if i < len(sol.get("schedules", [])) else {}
            scene_rows.append([name, fmt(sol.get("probabilities", [])[i], 3), fmt(scene_eens[i]) if i < len(scene_eens) else "未记录", fmt(scene_starts[i], 2) if i < len(scene_starts) else "未记录", fmt(scene_net[i], 2) if i < len(scene_net) else "未记录", f"{fmt(sched.get('repair_hours',{}).get('A'),2)} / {fmt(sched.get('repair_hours',{}).get('B'),2)}"])
        if scene_rows:
            table(doc, ["场景", "概率", "EENS/MWh", "启动/h", "净出力/h", "修复A/B/h"], scene_rows)
    if sol.get("cost_components"):
        table(doc, ["期望成本构成（原始输出键）", "值"], [(k, fmt(v)) for k, v in sol["cost_components"].items()])
    screening = results / "policy_screening.csv"
    if screening.exists():
        import csv
        with screening.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            keys = list(rows[0])
            # 原始CSV保留全部列；Word挑选易比较的列，避免横向过宽。
            keep = [k for k in keys if any(w in k.lower() for w in ("policy", "branch", "objective", "bound", "gap", "seconds"))][:7]
            if not keep:
                keep = keys[:6]
            table(doc, keep, [[fmt(r.get(k)) for k in keep] for r in rows])
    heading("9.3 当前启动/等待与D", 2)
    if comparison:
        rows = []
        for key, value in comparison.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if not isinstance(v, (dict, list)):
                        rows.append((f"{key}.{k}", fmt(v)))
            elif not isinstance(value, list):
                rows.append((key, fmt(value)))
        table(doc, ["分支比较原始记录", "值"], rows)
        p(doc, "应同时阅读分支可行状态、上下界和D区间。如果结果只提供可行成本差，应将其称为当前可行策略差；若认证区间跨零，不能据此断言真实最优行动。若某分支不可行，先讨论约束与物理原因，不计算∞−∞。")
        if comparison.get("D_lower") is not None and comparison.get("D_upper") is not None:
            dl, du = comparison["D_lower"], comparison["D_upper"]
            decision = "现在启动" if dl > 0 else "当前等待" if du < 0 else "尚不能认证哪种行动更优"
            p(doc, f"本次计算给出D∈[{fmt(dl)},{fmt(du)}]千元，因此在所声明菜单内可支持的判断为：{decision}。两可行策略的代价差为{fmt(comparison.get('D_estimate'))}千元，它是展示值，真实最优D以区间为准。")
        if sol.get("summary") and wait_solution.get("summary"):
            p(doc, f"立即动作可行策略的EENS为{fmt(result_value(sol,'expected_eens_mwh'))}MWh，等待分支可行策略为{fmt(result_value(wait_solution,'expected_eens_mwh'))}MWh；相应可用净出力时刻为{fmt(result_value(sol,'expected_handover_hour'),2)}h与{fmt(result_value(wait_solution,'expected_handover_hour'),2)}h。此差异同时反映完整优化后的供电和能源代价，不能只归因于启动时间的一个指标。")
        fixed = comparison.get("fixed_crew", {})
        best_agg = comparison.get("free", {})
        if fixed.get("objective") is not None and best_agg.get("objective") is not None:
            improvement = fixed["objective"] - best_agg["objective"]
            p(doc, f"固定策略{fixed.get('policy_id')}与菜单内当前最好可行策略{best_agg.get('policy_id')}的可行费用相差{fmt(improvement)}千元。只有当最佳方案上界低于固定策略下界，才能认证其最优价值改善；当前{'满足' if best_agg['objective'] < fixed.get('lower_bound',-math.inf) else '尚不满足'}该充分条件，不能把可行值排序直接宣称为已证明的策略优越性。")
            if str(best_agg.get("policy_id", "")).endswith(("_11", "_00")):
                p(doc, "当前最好策略在首次轻/重观测后采用相同的任务先后规则。因此上述差异主要支持首巡点及修复优先顺序的选择，不是对“观测后自适应改变维修顺序”优势的证明；信息约束的作用需要进一步专项消融。")
    else:
        p(doc, "当前结果没有branch_comparison.json，因此不报告D数值或临界边界。可执行相应分支比较命令后重建本文档。")
    heading("9.4 图件如何回答研究问题", 2)
    figdir = results / "figures"
    figs = sorted(figdir.glob("*.png")) if figdir.exists() else []
    # 跨状态D图由独立汇总器生成，保留它的来源目录，不复制成伪单算例输出。
    comparison_figdir = results.parent / "decision_comparison" / "figures"
    if comparison_figdir.exists():
        existing_names = {x.name for x in figs}
        figs.extend(x for x in sorted(comparison_figdir.glob("11*.png")) if x.name not in existing_names)
    captions = {
        "01": "24小时资源与接口轨迹：核对阶段支撑、净外送与SOC跨期消耗，不能只看瞬时峰值。",
        "02": "终端服务恢复与累计缺供：用曲线面积和EENS解释韧性，而非只报告启动时刻。",
        "03": "场景相关任务日程：巡检时刻与修复时刻分开标记，说明实体任务和信息获取的代价。",
        "04": "条件信息历史与场景分组：未区分场景必须共用控制，分叉须发生在实际观测后。",
        "05": "启动阶段与可用净输出：显示完整厂用电轮廓、爬坡及可用阈值，避免只画点火成功。",
        "06": "物理状态与分解校核：联合检查线路电压、场景SOC及功率平衡，解释执行可行性。",
        "07": "启动/等待的代价差与认证区间：区间跨零应继续计算或报告未认证。",
        "08": "条件能力投影与见证：切片来自完整24小时约束，不代表可任意逐时拼接。",
        "09": "协同与禁反送基线：同资源同需求下比较实际服务和缺供面积，说明协同收益的物理来源。",
        "10": "固定队伍策略的滚动回放：展示固定24小时终点、观测后验、已执行历史与实测SOC继承。",
        "11": "不同外部支撑状态的D区间：离散样本分别完整重求，不用两点插值宣称唯一临界阈值。",
    }
    for i, fig in enumerate(figs, 1):
        prefix = fig.name[:2]
        add_figure(doc, fig, f"图{i}  {captions.get(prefix, fig.stem.replace('_',' '))}")
        if prefix in captions:
            p(doc, f"阅读要点：{captions[prefix]}具体数值以图中坐标及对应CSV为准，不能从演示图推导未计算的速度提升或普遍临界值。")
    if not figs:
        p(doc, "尚未发现PNG图件；先执行出图流程。脚本不会用手绘趋势冒充模型输出。")
    if baseline.get("summary") and sol.get("summary"):
        p(doc, f"本次协同可行策略服务电量为{fmt(result_value(sol,'expected_restored_energy_mwh'))}MWh，禁反送基线为{fmt(result_value(baseline,'expected_restored_energy_mwh'))}MWh；EENS分别为{fmt(result_value(sol,'expected_eens_mwh'))}MWh与{fmt(result_value(baseline,'expected_eens_mwh'))}MWh。由于默认α=0，禁止向上支撑也阻断了目标机组初始启动电源，较大的收益主要来自这个明确的工程设定，不能外推为任何灾情下协同都有相同幅度收益。")
    if sol.get("summary"):
        scenario_eens = result_value(sol, "scenario_eens_mwh", [])
        if scenario_eens and max(scenario_eens) - min(scenario_eens) < 1e-6:
            p(doc, "本次所选可行策略的4场景EENS数值相同，且当前动作并未随巡检结果改变。这说明演示算例已验证模型求解链与信息约束，但尚未充分展示“信息改变启动选择”的第三贡献。正式论文需增加公平的SOC、损坏先验、通道与启动代价敏感性，并如实保留未发生策略分叉的情形。")
        elif scenario_eens:
            p(doc, f"不同损坏场景EENS在{fmt(min(scenario_eens))}—{fmt(max(scenario_eens))}MWh之间，说明恢复轨迹确实受到通道修复影响。场景差异本身尚不能等同于主动信息的净价值，还需与固定揭示时刻、固定检查策略或理想完全信息进行公平对照。")
            sh = result_value(sol, "scenario_start_hour", [])
            if sh and len({x for x in sh}) == 1:
                p(doc, "此初始状态下，各场景的启动选择时刻相同，因此本例尚未展示巡检后启动动作发生分叉。它支持多场景物理与信息模型的可执行性，而“信息改变启动边界”的进一步机理仍需补充状态和对照实验。")
    heading("9.5 约束与分解校核", 2)
    if validation:
        table(doc, ["校核记录", "实际值"], [(k, fmt(v)) for k, v in flatten(validation)])
    else:
        p(doc, "未发现validation.json，应先执行校核再使用结果。")
    if capability:
        summary_rows = []
        for k, v in capability.items():
            if not isinstance(v, (list, dict)):
                summary_rows.append((k, fmt(v)))
        if summary_rows:
            table(doc, ["条件域记录", "实际值"], summary_rows)
        p(doc, "完整的条件域采样、LP可行见证和接口轨迹分解记录保存在capability目录。支持函数求解失败的方向应保留状态，不能把缺少边界点解释为物理不可行。")
    p(doc, "数值校核依次检查约束矩阵残差、变量边界、整数偏差、SOC递推、受保护服务、接口平衡和非预期性。在当前线性模型中通过校核，只说明满足该模型与容差；进入工程应用前仍须进行交流潮流、启动冲击、频率与保护校核。")
    heading("9.6 固定队伍策略的真实滚动回放", 2)
    if rolling:
        p(doc, f"沿实际场景{rolling.get('realized_scenario_id')}进行{rolling.get('reoptimizations',len(rolling.get('rounds',[])))}次重算，抢修策略固定为{rolling.get('fixed_crew_policy',{}).get('id','未记录')}，原终点{rolling.get('horizon_hours',24)}h，回退次数{rolling.get('fallback_count','未记录')}。各轮全时域条件费用包含不同的已执行历史及后验，不能将其逐轮下降解释成同一问题的算法改善。")
        table(doc, ["重算时刻/h", "剩余/h", "兼容场景与后验", "实测/继承SOC", "回退", "校核"], [
            [fmt(r.get("current_hour"),2), fmt(r.get("remaining_horizon_hours"),2), "; ".join(f"{k}:{fmt(v,3)}" for k,v in r.get("posterior",{}).items()), fmt(r.get("current_soc"),3), fmt(r.get("fallback")), fmt(r.get("validation",{}).get("passed"))]
            for r in rolling.get("rounds",[])
        ])
        p(doc, "此回放证明已执行前缀、SOC与已开始任务得以继承，并在实际观测后筛选场景。队伍策略没有滚动重排；最后一次重算后沿其因果分支继续执行至24小时，执行路径需再次通过全时域校核。")
    else:
        p(doc, "该结果目录尚未包含rolling/rolling_replay.json，本版不填回放数值。rolling.py提供固定队伍策略的观测更新与历史锁定实现，可执行相应命令后重建本文档。")
    heading("9.7 不同主网剩余供电状态下的行动价值", 2)
    if external_comparison and comparison:
        state_rows = []
        for label, config, comp in (("主算例", cfg, comparison), ("补充外部支撑状态", external_config, external_comparison)):
            state_rows.append([label, fmt(config.get("alpha"),2), fmt(comp.get("start",{}).get("objective")), fmt(comp.get("wait",{}).get("objective")), f"[{fmt(comp.get('D_lower'))}, {fmt(comp.get('D_upper'))}]", comp.get("decision","未记录")])
        table(doc, ["状态", "α", "启动上界/千元", "等待上界/千元", "D区间/千元", "认证判断"], state_rows)
        p(doc, "两种状态分别完整重求8个队伍策略和当前动作分支。补充状态的完整配置、轨迹、求解界和校核位于results/external_support，不能只修改图上的D而不重求模型。比较时α表示剩余外部供电比例，增大α意味着主网可用供电增加。若D区间异号，可以证明这些离散状态下的最优当前动作不同；这并不证明二者之间只有一个临界值，也不支持任意两点线性插值。")
        s = external_comparison.get("D_lower")
        u = external_comparison.get("D_upper")
        if s is not None and u is not None:
            p(doc, f"补充状态α={fmt(external_config.get('alpha'),2)}的D∈[{fmt(s)},{fmt(u)}]千元；{'区间全负，认证当前等待优于强制现在启动' if u < 0 else '区间全正，认证当前启动更优' if s > 0 else '区间跨零，仍不能认证当前行动优劣'}。此处等待仍允许以后启动，其费用同时包含后续最优行动。")
    else:
        p(doc, "本结果目录尚未提供跨外部供电状态的完整分支计算，不据此绘制临界值或外推D的符号。可配置α并重新求解后追加此表。")

    heading("10 如何形成正式算例与下一步实验")
    table(doc, ["实验", "保持一致的条件", "应展示的结论"], [
        ["当前启动与等待", "同初态、先验、资源、菜单和目标", "D区间、可行/不可行区域、实际服务差；不预设单阈值"],
        ["固定与因果队伍策略", "同一电力优化和队伍资源", "先查何处如何改变观测、修复时刻和启动动作"],
        ["理想完全信息", "同物理参数，提前揭示仅作参照", "信息代价上限；不作为可执行方案"],
        ["条件域与完整模型", "同一固定离散方案及全时域约束", "轨迹可分解性、投影切片误用的风险与代价"],
        ["SOC/α/修复时长敏感性", "一次只改变声明变量，完整重求分支", "决策边界可能非单调；每格保留求解状态"],
        ["后续压缩方法", "完整模型为基准、合法界和执行校核", "行动误判率、代价损失、信息量和总耗时"],
    ])
    p(doc, "演示参数首先服务物理一致性：微网可用功率、能量、无功能力与通道容量必须与启动轮廓相匹配。真实参数下若当前启动始终更优，或始终不可行，应如实报告并调整研究问题；不能预填“协同为负”或通过不公平基线制造结论。")
    p(doc, "创新点②若作为论文主要方法贡献，下一步必须实现按D认证需求进行的自适应查询与细化，并对照固定精度域、已有多时段MPAR及完整模型。当前代码提供正确性基准、条件域与见证，不替代这一步研究。创新点③需进一步阅读内生信息和主动巡检最新全文，避免与相邻研究重复。")

    heading("附录A 代码、模型与文件对应关系")
    table(doc, ["模块/文件", "负责内容", "对应本文"], [
        ["configs/*.json；config.py", "全部演示参数、输入校验和96步曲线", "第2、9.1节"],
        ["crew.py", "4场景、8策略、旅行/巡检/维修和观测历史", "第6节"],
        ["linear.py；model.py", "稀疏变量/约束、MILP/LP调用及物理/信息模型", "第3—6、8节"],
        ["solution.py；experiments.py", "提取实际指标，筛选策略和比较当前动作", "第8—9节"],
        ["rolling.py", "固定队伍策略，更新后验并锁定历史后重求功率调度", "第8.2、9.6节"],
        ["capability.py", "固定离散方案的全时域LP、投影和分解", "第7节"],
        ["validation.py；tests/", "数值和物理校核、因果性检查", "第9.5节"],
        ["plotting.py", "从计算结果出24小时PNG/SVG/PDF", "第9.4节"],
        ["docs/code_walkthrough.md", "每个模块和关键代码块的详细中文解释", "配套阅读"],
        ["scripts/build_paper.py", "读取结果重建本Word与公式源", "全文"],
    ])
    p(doc, "修改参数后应重新求解、校核和出图，再重建本文档；不要只改论文表格而保留旧结果。源代码中的中文模块说明与分段注释解释变量、约束和输出，详细运行顺序见README及代码说明书。")
    heading("附录B 可编辑公式与转换")
    p(doc, "所有正文公式均使用OMML结构，具有真实的上下标、分式和求和节点。Word中点击公式后可编辑；LaTeX源按相同编号保存于equations/equations.tex，公式索引保存在formula_catalog.json。安装MathType后，可按本机MathType版本支持的Word公式转换功能处理；转换成功与否应在本机检查，本交付不把OMML称为MathType原生对象。")
    p(doc, f"本文件共{len(specs)}条同编号数学公式。若使用不支持OMML的软件，显示可能变化，建议用Microsoft Word打开并保存副本。")
    heading("参考文献（已读材料与明确的核对范围）")
    refs = [
        "[1] 杨明通, 刘一欣, 郭力, 张宇轩, 王中冠, 王成山. 基于数据驱动安全域的输-配-微多层级分布式经济调度方法[J/OL]. 中国电机工程学报, 2025-02-26网络首发版本. https://link.cnki.net/urlid/11.2107.tm.20250225.1334.007. 本文依据用户提供版本核对SOC、递归鲁棒域与MPQP机制；如使用正式卷期，应按正式发表版本更新题录。",
        "[2] 王炜歆, 王小君, 许寅, 等. 考虑输配协同的电网并行恢复分区及机组启动次序统一优化决策方法. 中国电机工程学报, 2024, 44(3):859—871. DOI:10.13334/j.0258-8013.pcsee.222233.",
        "[3] Zhang M, Cai S, Xie Y, Zhou B, Zheng W, Wu Q, Wen J. Supply Resilience Constrained Scheduling of MEGs for Distribution System Restoration: A Stochastic Model and FW-PH Algorithm. IEEE Transactions on Smart Grid, 2025,16(1):194—208. DOI:10.1109/TSG.2024.3446873.",
        "[4] 吴熙, 唐子逸, 徐青山, 周亦洲. 基于Q学习算法的综合能源系统韧性提升方法. 电力自动化设备, 2020,40(4). DOI:10.16081/j.epae.202002006.",
        "[5] 唐坤霆, 周永智, 李宝聚, 韦巍. 考虑热惯性的极端冰雪灾害下综合能源系统韧性提升. 电力系统自动化, 2024,48(21):129—137. DOI:10.7500/AEPS20240109001.",
    ]
    for ref in refs:
        p(doc, ref)
    p(doc, "未核对全文的网页题录和其他工具生成的评述不作为本文已验证的模型或图件依据。关于“首次”“无人研究”等表述，应在后续系统查新后谨慎决定。")

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)
    eqdir = ROOT / "equations"
    eqdir.mkdir(exist_ok=True)
    tex = [r"\documentclass[11pt]{article}", r"\usepackage{amsmath,amssymb}", r"\begin{document}", "% Equation numbers match the editable OMML equations in the Word report."]
    catalog = []
    for entry in used_specs:
        tex.extend([f"% ({entry['number']}) {entry['title']}", r"\begin{equation}", entry["latex"], r"\tag{" + str(entry["number"]) + "}", r"\end{equation}"])
        catalog.append({k: v for k, v in entry.items() if k != "body"})
    tex.append(r"\end{document}")
    (eqdir / "equations.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    (eqdir / "formula_catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"word": str(output), "equations": len(specs), "result_source": str(results), "embedded_figures": len(figs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "demo")
    parser.add_argument("--output", type=Path, default=ROOT / "论文式模型与算例分析_可编辑.docx")
    args = parser.parse_args()
    print(json.dumps(build(args.results.resolve(), args.output.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
