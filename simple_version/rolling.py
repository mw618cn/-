"""真实观测驱动的固定抢修策略滚动回放。

每轮保留原始 [0,24 h] 终点与历史状态；只按已完成巡检更新后验信息，
固定已执行的功率/能量/离散决定，然后重算尚未执行的未来。抢修策略在
本模块内锁定，不重新选择路线，也不撤销已经开始的维修。这样可以检验
“执行—观测—更新—再优化”的闭环，而不把一次性情景树冒称滚动求解。
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from power_model import solve
from power_model import build_model
from analysis import extract_solution
from analysis import validate_solution


_CONTINUOUS_STAGE = (
    'battery_charge','battery_discharge','pv_used','dg_power','inverter_q',
    'served_critical','served_general','target_power','target_q','flow_p','flow_q',
    'voltage_sq','grid_p','grid_q',
)


def _jsonable(value: Any) -> Any:
    """把数值数组和非有限求解界转换为可移植 JSON；不暴露原始大解向量。"""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float,np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _save(path: Path, value: Any) -> None:
    """写入 UTF-8 数据文件，保留中文观测名称与单位。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')


def _scenario_index(model, scenario_id: str) -> int:
    """按稳定情景 ID 寻址，情景过滤后不沿用旧数组行号。"""
    for i,scene in enumerate(model.metadata['scenarios']):
        if scene['id']==scenario_id:
            return i
    raise ValueError(f'当前模型不含实际情景 {scenario_id}')


def _transfer_plan(old_model, old_x, new_model) -> np.ndarray:
    """将上一轮各兼容情景的整条可行计划映射至过滤后的模型。

    保留初始开关事件网格，因此除情景轴外各变量块完全同形；这既用于
    失败时安全回退，也用于把最终真实分支转成单情景的执行记录。
    """
    old_ids={s['id']:i for i,s in enumerate(old_model.metadata['scenarios'])}
    new_ids=[s['id'] for s in new_model.metadata['scenarios']]
    x=np.zeros_like(new_model.c,dtype=float)
    for name,new_idx in new_model.var_indices.items():
        if name not in old_model.var_indices:
            raise ValueError(f'无法回退：上轮模型缺少变量块 {name}')
        old_idx=old_model.var_indices[name]
        if old_idx.shape[1:]!=new_idx.shape[1:]:
            raise ValueError(f'滚动变量块 {name} 的动作网格改变，不能重写历史')
        for s,sid in enumerate(new_ids):
            x[new_idx[s]]=np.asarray(old_x)[old_idx[old_ids[sid]]]
    return x


def _lock_executed_prefix(model, old_model, old_x, realized_id: str, now_step: int) -> dict:
    """把过去真正执行的决定和当前状态锁定到所有兼容情景。

    连续命令锁 t<now；能量状态锁 t<=now。启动、开关与电池模式按
    各自决定起点锁定；已经开始的小时模式块即使跨过 now 也不可修改。
    未来状态和未来模式仍是优化变量，不能因为“锁历史”而固定未来。
    """
    old_s=_scenario_index(old_model,realized_id)
    count=0
    blocks={}
    max_adjustment=0.0

    def fix(name: str, positions) -> None:
        nonlocal count,max_adjustment
        new_idx=model.var_indices[name]
        old_idx=old_model.var_indices[name]
        source=np.asarray(old_x)[old_idx[old_s,...,positions]]
        # 使用逐情景的相同真实执行前缀，而非继续保留已经排除情景的过去。
        local_count=0
        for s in range(new_idx.shape[0]):
            cols=np.asarray(new_idx[s,...,positions]).ravel()
            values=np.asarray(source).ravel().copy()
            is_int=np.asarray(model.integrality[cols])!=0
            original_values=values.copy()
            integer_error=np.max(np.abs(values[is_int]-np.rint(values[is_int])),initial=0.0)
            bound_error=max(np.max(model.lb[cols]-values,initial=0.0),
                            np.max(values-model.ub[cols],initial=0.0))
            if integer_error>2e-5 or bound_error>2e-5:
                raise ValueError(f'历史块{name}偏离新模型物理界或整数性，不能通过裁剪改写历史')
            values[is_int]=np.rint(values[is_int])
            # 仅消除已通过2e-5容差检查的浮点噪声；大幅违约已在上方拒绝。
            values=np.clip(values,model.lb[cols],model.ub[cols])
            max_adjustment=max(max_adjustment,float(np.max(np.abs(values-original_values),initial=0.0)))
            model.lb[cols]=values
            model.ub[cols]=values
            local_count+=len(cols)
        count+=local_count
        blocks[name]=local_count

    for name in _CONTINUOUS_STAGE:
        if name in model.var_indices:
            fix(name,slice(0,now_step))
    if 'energy' in model.var_indices:
        fix('energy',slice(0,now_step+1))
    for name,slot_key in (
        ('target_start','start_slots'),
        ('battery_mode','battery_mode_slots'),
        ('switch_A','event_slots'),
        ('switch_B','event_slots'),
    ):
        if name in model.var_indices:
            slots=np.asarray(model.metadata[slot_key],dtype=int)
            fix(name,np.flatnonzero(slots<now_step))
    model.metadata['rolling_current_step']=int(now_step)
    model.metadata['rolling_fixed_prefix']=True
    return {'fixed_variable_count':count,'by_block':blocks,
            'max_numeric_prefix_adjustment':max_adjustment,
            'past_dispatch_condition':'t < current_step',
            'state_condition':'t <= current_step',
            'discrete_condition':'decision block start < current_step'}


def run_rolling_replay(cfg, policy, initial_model, initial_result,
                       realized_scenario_id='HH',
                       decision_hours=(.5,1,1.5,2,4,8,12,18), outdir=None):
    """沿一个实际损伤情景运行真实的观测—后验—再求解回放。

    参数 ``initial_model/result`` 必须是本策略的已校核可行解；不在这里
    再择抢修策略。所有轮次的需求终点始终为 cfg.horizon_hours（默认24），
    而不是在18时再向后延伸24小时。过去成本仍在完整目标中，报告为
    “给定历史后的全时域条件期望成本”，不能与不同历史直接解释为改进。

    每轮求解没有合法 incumbent 时，验证并继续执行上一轮兼容分支的
    剩余计划；该动作会记录 fallback。若连回退也不可行，立即报错，
    不生成虚构执行曲线。

    输出
    ----
    dict
        rounds 为观测/后验/动作/状态/校核；executed_solution 为沿实际分支
        执行的完整24小时轨迹。指定 outdir 时另写逐轮与CSV文件。
    """
    if initial_result.get('x') is None:
        raise ValueError('滚动回放需要已有可行的初始24小时计划')
    if initial_model.metadata.get('policy',{}).get('id')!=policy['id']:
        raise ValueError('初始模型抢修策略与锁定策略不一致')
    if initial_model.metadata.get('information','causal')!='causal':
        raise ValueError('实际回放要求初始计划满足因果信息约束，不能使用完美信息参照')
    initial_check=validate_solution(initial_model,initial_result,cfg)
    if not initial_check['passed']:
        raise ValueError('初始计划物理校核失败，不能执行')

    dt=cfg['step_minutes']/60
    horizon=float(cfg['horizon_hours'])
    hours=[float(h) for h in decision_hours]
    if hours!=sorted(set(hours)) or any(h<=0 or h>=horizon for h in hours):
        raise ValueError('滚动决定时刻必须递增、唯一，且严格位于(0,原始终点)')
    if any(not np.isclose(h/dt,round(h/dt)) for h in hours):
        raise ValueError('滚动时刻必须与15分钟调度网格对齐')
    actual_initial_s=_scenario_index(initial_model,realized_scenario_id)
    initial_scenes=initial_model.metadata['scenarios']
    original_priors={s['id']:float(s['probability']) for s in initial_scenes}
    original_histories={s['id']:initial_model.metadata['schedules'][i]['observation_history_by_step']
                        for i,s in enumerate(initial_scenes)}
    actual_history=initial_model.metadata['schedules'][actual_initial_s]['observation_history_by_step']
    fixed_events=np.asarray(initial_model.metadata['event_slots'],dtype=int).tolist()
    old_model=initial_model
    old_result=initial_result
    rounds=[]
    output=Path(outdir) if outdir is not None else None
    if output is not None:
        output.mkdir(parents=True,exist_ok=True)

    # 01 执行初始策略到第一个重算时刻；其后每轮仅用当时已经完成的巡检。
    for hour in hours:
        step=int(round(hour/dt))
        observed=actual_history[step]
        signature=json.dumps(observed,sort_keys=True,ensure_ascii=False)
        compatible=[s for s in initial_scenes
                    if json.dumps(original_histories[s['id']][step],sort_keys=True,ensure_ascii=False)==signature]
        mass=sum(original_priors[s['id']] for s in compatible)
        if mass<=0:
            raise RuntimeError('实际观测在原始联合场景中概率为零，须显式扩展场景集')
        post_cfg=copy.deepcopy(cfg)
        post_cfg['scenarios']=copy.deepcopy(compatible)
        for scene in post_cfg['scenarios']:
            scene['probability']=original_priors[scene['id']]/mass
        post_cfg['recovery_event_slots']=fixed_events
        # 不重置时间与SOC；完整原始模型通过前缀约束继承已经执行的一切。
        new_model=build_model(post_cfg,policy,
                              branch=initial_model.metadata.get('branch','free'),
                              information='causal',
                              coordination=initial_model.metadata.get('coordination',True),
                              fixed_switches=initial_model.metadata.get('fixed_switches',False))
        lock=_lock_executed_prefix(new_model,old_model,old_result['x'],
                                   realized_scenario_id,step)

        # 02 重算剩余控制；没有可行解时保留原本可行的条件计划，而非停供或编数。
        trial=solve(new_model,post_cfg['solver'])
        check=validate_solution(new_model,trial,post_cfg)
        fallback=not (trial.get('x') is not None and check['passed'])
        solver_attempt={k:v for k,v in trial.items() if k!='x'}
        if fallback:
            inherited_x=_transfer_plan(old_model,old_result['x'],new_model)
            fallback_result={
                'success':False,'status':1,
                'message':'本轮重算未取得通过校核的解，继续上一轮已校核的因果剩余计划',
                'x':inherited_x,
                'objective':float(new_model.c@inherited_x+new_model.objective_constant),
                'lower_bound':trial.get('lower_bound',-np.inf),
                'mip_gap':None,'solve_seconds':trial.get('solve_seconds',0.0),'node_count':0,
            }
            check=validate_solution(new_model,fallback_result,post_cfg)
            if not check['passed']:
                raise RuntimeError(f'{hour:g}h重算失败且继承计划未通过校核，禁止生成执行数据：{check}')
            trial=fallback_result

        # 03 只记录实际时刻的动作与SOC；未来真实损伤仅在其巡检完成后才出现。
        s=_scenario_index(new_model,realized_scenario_id)
        ix=new_model.var_indices
        xx=np.asarray(trial['x'])
        def at(name):
            return xx[ix[name]][s,...,step].tolist()
        phase=np.searchsorted(np.asarray(new_model.metadata['event_slots']),step,side='right')-1
        current_starts=np.flatnonzero(np.asarray(new_model.metadata['start_slots'])==step)
        action={
            'battery_charge_mw':at('battery_charge'),
            'battery_discharge_mw':at('battery_discharge'),
            'mg_dg_mw':at('dg_power'),
            'target_power_mw':float(xx[ix['target_power']][s,step]),
            'grid_power_mw':float(xx[ix['grid_p']][s,step]),
            'upward_p_mw':float(-xx[ix['flow_p']][s,1,step]-xx[ix['flow_p']][s,2,step]),
            'switch_A':int(round(xx[ix['switch_A']][s,phase])),
            'switch_B':int(round(xx[ix['switch_B']][s,phase])),
            'start_now':bool(len(current_starts) and xx[ix['target_start']][s,current_starts[0]]>.5),
        }
        current_energy=xx[ix['energy']][s,:,step]
        capacity=np.asarray([m['energy_mwh'] for m in cfg['microgrids']])
        record={
            'current_hour':hour,'original_terminal_hour':horizon,
            'remaining_horizon_hours':horizon-hour,
            'observations':[{'site':a,'damage':b,'revealed_hour':float(c)} for a,b,c in observed],
            'posterior':{s['id']:float(s['probability']) for s in post_cfg['scenarios']},
            'compatible_scenario_count':len(compatible),'fixed_crew_policy_id':policy['id'],
            'fixed_prefix':lock,'fallback':fallback,'solver_attempt':solver_attempt,
            'conditional_full_horizon_cost':float(trial['objective']),
            'cost_interpretation':'含已执行历史成本；仅同一历史与同一后验下的分支可直接比较',
            'current_soc':(current_energy/capacity).tolist(),'next_dispatch_action':action,
            'validation':check,
        }
        rounds.append(record)
        if output is not None:
            _save(output/f'round_{hour:05.2f}h.json',record)
        old_model,old_result=new_model,trial

    # 04 最后一次因果计划沿真实分支继续执行至原始24h终点，形成完整执行轨迹。
    final_cfg=copy.deepcopy(cfg)
    actual=next(s for s in initial_scenes if s['id']==realized_scenario_id)
    final_cfg['scenarios']=[copy.deepcopy(actual)]
    final_cfg['scenarios'][0]['probability']=1.0
    final_cfg['recovery_event_slots']=fixed_events
    final_model=build_model(final_cfg,policy,
                            branch=initial_model.metadata.get('branch','free'),
                            information='causal',
                            coordination=initial_model.metadata.get('coordination',True),
                            fixed_switches=initial_model.metadata.get('fixed_switches',False))
    final_x=_transfer_plan(old_model,old_result['x'],final_model)
    final_result={'success':False,'status':1,'message':'实际情景路径执行记录；不是单情景完美信息重新优化',
                  'x':final_x,'objective':float(final_model.c@final_x+final_model.objective_constant),
                  'lower_bound':-np.inf,'mip_gap':None,'solve_seconds':0.0,'node_count':0}
    final_check=validate_solution(final_model,final_result,final_cfg)
    if not final_check['passed']:
        raise RuntimeError(f'实际执行路径未通过全时域校核：{final_check}')
    executed=extract_solution(final_model,final_result,final_cfg)
    replay={
        'realized_scenario_id':realized_scenario_id,'fixed_crew_policy':policy,
        'decision_hours':hours,'horizon_hours':horizon,'n_steps':cfg['n_steps'],
        'reoptimizations':len(rounds),'fallback_count':sum(r['fallback'] for r in rounds),
        'initial_validation':initial_check,'rounds':rounds,
        'executed_solution':executed,'executed_validation':final_check,
        'scope':'固定抢修策略；真实巡检观测筛选情景并更新后验；固定已执行前缀；24h固定终点滚动调度',
        'note':'最后重算之后沿其因果计划继续执行；未来观测触发的情景分支均遵守原信息规则。',
    }
    if output is not None:
        _save(output/'rolling_replay.json',replay)
        _save(output/'executed_solution.json',executed)
        with (output/'rolling_rounds.csv').open('w',encoding='utf-8-sig',newline='') as f:
            fields=['current_hour','remaining_horizon_hours','compatible_scenario_count',
                    'observations','posterior','fallback','solve_seconds','conditional_full_horizon_cost',
                    'current_soc','next_dispatch_action']
            writer=csv.DictWriter(f,fieldnames=fields)
            writer.writeheader()
            for rec in rounds:
                row={k:rec.get(k) for k in fields}
                row['solve_seconds']=rec['solver_attempt']['solve_seconds']
                for key in ('observations','posterior','current_soc','next_dispatch_action'):
                    row[key]=json.dumps(_jsonable(row[key]),ensure_ascii=False)
                writer.writerow(row)
    return _jsonable(replay)
