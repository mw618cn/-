"""有限因果抢修策略：显式行程时间、巡检结束揭示损伤、不可中断维修。

八个候选策略不是任意车辆路径问题；只在首个观测完成后按损伤选择次序。
通过物理日程得到各场景信息历史，供电力模型建立条件非预期性约束。
"""
from __future__ import annotations
import copy
import itertools
import numpy as np


def build_scenarios(cfg):
    """复制联合损伤场景，不根据未来真实损伤提前改变观测前控制。"""
    return copy.deepcopy(cfg["scenarios"])


def enumerate_policies(cfg):
    """2个先巡检地点×轻伤后2种次序×重伤后2种次序，共8种策略。"""
    policies=[]
    for first, light, heavy in itertools.product(("A","B"),(True,False),(True,False)):
        pid=f"{first}_{int(light)}{int(heavy)}"
        policies.append({"id":pid,"label":f"先巡{first}；轻伤{'先修本点' if light else '先去另一点'}；重伤{'先修本点' if heavy else '先去另一点'}",
                         "first":first,"repair_first_if_light":light,"repair_first_if_heavy":heavy})
    return policies


def schedule_policy(cfg, policy, scenarios=None):
    """对各损伤场景展开同一因果策略，返回可用性与每时段已知观测历史。

    维修完成后线路才可合闸。巡检结束时损伤已知；在t时刻只允许使用
    end_hour<=t的观测。无法从尚未完成的维修时长偷看真实损伤。
    """
    scenarios=build_scenarios(cfg) if scenarios is None else scenarios
    result=[]; dt=cfg["step_minutes"]/60; n=cfg["n_steps"]
    first=policy["first"]; other="B" if first=="A" else "A"
    for scene in scenarios:
        clock=0.0; position="depot"; tasks=[]; inspection={}; repairs={}
        def travel(site):
            nonlocal clock,position
            if site==position: return
            duration=cfg["crew"]["travel_hours"][position][site]
            tasks.append({"site":site,"kind":"travel","start_hour":clock,"end_hour":clock+duration,"from":position})
            clock+=duration; position=site
        def inspect(site):
            nonlocal clock
            duration=cfg["crew"]["inspection_hours"][site]
            tasks.append({"site":site,"kind":"inspect","start_hour":clock,"end_hour":clock+duration})
            clock+=duration; inspection[site]=clock
        def repair(site):
            nonlocal clock
            duration=cfg["crew"]["repair_hours"][site][scene["damage"][site]]
            tasks.append({"site":site,"kind":"repair","start_hour":clock,"end_hour":clock+duration})
            clock+=duration; repairs[site]=clock
        travel(first); inspect(first)
        repair_first=policy["repair_first_if_"+scene["damage"][first]]
        if repair_first:
            repair(first); travel(other); inspect(other); repair(other)
        else:
            travel(other); inspect(other); repair(other); travel(first); repair(first)
        histories=[]
        for t in np.arange(n)*dt:
            histories.append(tuple((site,scene["damage"][site],inspection[site]) for site in sorted(inspection) if inspection[site]<=t+1e-9))
        result.append({"scenario_id":scene["id"],"tasks":tasks,"inspection_hours":inspection,"repair_hours":repairs,
                       "observation_history_by_step":histories,
                       "availability":{site:(np.arange(n)*dt>=repairs[site]-1e-9).astype(int).tolist() for site in ("A","B")}})
    return result
