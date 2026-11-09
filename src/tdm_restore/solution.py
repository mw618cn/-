"""把求解变量转换为带单位的结果、费用分项与韧性指标。

求解状态、可行目标上界和对偶下界一并输出；启动概率不冒充动态仿真
成功率。韧性以同一需求曲线下的恢复供电面积、EENS和保护负荷服务体现。
"""
from __future__ import annotations
import numpy as np
from .config import build_profiles


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
