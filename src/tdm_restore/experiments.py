"""可复现实验流水线：有限策略枚举→条件分支比较→校核→导出→出图。

枚举范围内保留每个MILP的目标上界和下界。即使限时返回可行解，也不
把它写成已证明的最优。D区间跨零时结论为尚未判定，而非强行画阈值。
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import time
import numpy as np
from .config import load_config
from .crew import enumerate_policies
from .model import build_model
from .linear import solve
from .solution import extract_solution
from .validation import validate_solution


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
        from .capability import compute_capability
        progress("[能力域] 固定离散方案，保留全部24h约束投影与轨迹分解",flush=True)
        cap=compute_capability(bestmodel,freebest["result"]["x"],cfg,out/"capability",directions=cfg["experiment"]["capability_directions"])
    save_json(out/"branch_comparison.json",comparison)
    if screening:
        with (out/"policy_screening.csv").open("w",encoding="utf-8-sig",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(screening[0]));writer.writeheader();writer.writerows(screening)
    export_series(solution,out)
    from .plotting import plot_all
    plot_all(solution,out/"figures",comparison=comparison,capability=cap)
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
    from .plotting import _set_style, _save, _stairs
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
