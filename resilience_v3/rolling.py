"""快速连续MPC：继承已执行前缀，重解剩余连续调度，只执行一时段。

策略/拓扑/启动的离散计划在周期或事件时重新选择；两次离散求解之间
只临时固定计划，并非把尚未执行的计划当成物理承诺。
"""
import copy
import numpy as np
from scipy import sparse
from power_model import freeze,solve_lp,residual,LinearModel,validate
from reachability import certify_commands,island_reserve_tube
from information import option


def refresh_continuous(c,state,result,m,r,compatible,real_pv):
    fixed=freeze(m,r['x']);md=copy.copy(m.metadata);v=m.var_indices
    origin=md['state']['step'];t=state['step']-origin
    if t<0 or t>=len(md['hours']):return None
    lb=fixed.lb.copy();ub=fixed.ub.copy();probs=md['forecast']['probabilities']
    conditional=np.zeros_like(probs);conditional[compatible]=probs[compatible]/sum(probs[compatible])
    reward=np.zeros_like(m.c);cost=np.zeros_like(m.c);original_cost=md['cost_objective']
    for s in range(len(probs)):
        for key,weight in [('served_c',c['critical_weight']),('served_g',c['general_weight'])]:reward[v[key][s]]=-conditional[s]*c['dt']*weight
    for key,idx in v.items():
        if np.asarray(idx).ndim>=2 and np.asarray(idx).shape[0]==len(probs):
            for s in range(len(probs)):cost[idx[s]]=original_cost[idx[s]]*conditional[s]/probs[s]
    for s in compatible:
        for key in ('served_c','served_g','pv','dg','charge','discharge','pcc_p','pcc_q','mg_q','grid_p','grid_q','target_p','target_q','flow_p','flow_q'):
            if key not in result:continue
            history=np.asarray(result[key][origin:state['step']])
            if not len(history):continue
            values=np.moveaxis(history,0,-1);cols=v[key][s,...,:t]
            lb[cols]=ub[cols]=values
        cols=v['energy'][s,:,:t+1];values=np.asarray(result['energy_history'][origin:state['step']+1]).T
        lb[cols]=ub[cols]=values
        ub[v['pv'][s,:,t]]=np.minimum(ub[v['pv'][s,:,t]],real_pv)
    md['forecast']=copy.copy(md['forecast']);md['forecast']['pv']=md['forecast']['pv'].copy()
    for scenario in compatible:md['forecast']['pv'][scenario,:,t]=np.minimum(md['forecast']['pv'][scenario,:,t],real_pv)
    if option(c,'reachability.enforce_island_reserve',True):
        md['reserve_tube']=island_reserve_tube(c,md['state'],md['forecast'])
        for scenario in compatible:
            for j in range(len(c['microgrids'])):
                cols=v['energy'][scenario,j,t:]
                lb[cols]=np.maximum(lb[cols],md['reserve_tube']['lower'][j,t:]);ub[cols]=np.minimum(ub[cols],md['reserve_tube']['upper'][j,t:])
        if np.any(lb>ub+1e-8):return None
    stage=LinearModel(reward,np.zeros_like(m.integrality),lb,ub,m.A,m.row_lb,m.row_ub,v,md)
    primary=solve_lp(stage)
    if primary['x'] is None or residual(stage,primary['x'])>2e-5:return None
    x=primary['x'];best=float(reward@x)
    # 二阶段只调整未执行部分，不累积过去各次求解的目标下限。
    second=LinearModel(cost,stage.integrality,lb,ub,sparse.vstack([stage.A,sparse.csr_matrix(reward.reshape(1,-1))],format='csr'),
                       np.r_[stage.row_lb,-np.inf],np.r_[stage.row_ub,best+c['restore_tolerance']*c['denominator']],v,md)
    rr=solve_lp(second)
    if rr['x'] is not None and residual(second,rr['x'])<=2e-5:x=rr['x']
    if not validate(stage,x)['passed']:return None
    tube=certify_commands(stage,x,start_index=t,active_scenarios=compatible) if option(c,'reachability.certify_commands',True) else None
    if tube is not None:
        if not tube['passed']:return None
        md['backward_certificate']=tube
    score=float((md['past_reward']-reward@x)/c['denominator'])
    return stage,{'x':x,'R_plan':score,'status':0,'source':'fixed_plan_LP','R_lower':None,'R_upper':None,
                  'scope':'条件于当前兼容预测样本及暂时固定的离散计划；不是完整策略类最优值界'}
