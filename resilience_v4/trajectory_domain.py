"""固定离散方案/信息树后的完整轨迹列协调。

MGO完整可行见证组成列，统一使用一组lambda覆盖全部微网、时段和样本。
限制主问题实际消去MGO内部列，以整轨迹凸组合代入全局物理行；求解后
恢复内部指令并复核。不是逐时拼接，也不把限制主问题的值当MILP上界。
"""
import time
import numpy as np
from scipy import sparse
from information import option

LOCAL_KEYS=('pv','dg','charge','discharge','energy','served_c','served_g','mg_q','pcc_p','pcc_q')


def index_map(global_model,local_model):
    c=global_model.metadata['cfg'];g=global_model.var_indices;l=local_model.var_indices
    mapping=np.full(len(local_model.c),-1,int)
    for key in LOCAL_KEYS:
        cols=g[key][:,c['mg_nodes'],:] if key in ('served_c','served_g') else g[key]
        mapping[l[key].ravel()]=np.asarray(cols).ravel()
    if np.any(mapping<0) or len(np.unique(mapping))!=len(mapping):raise RuntimeError('轨迹列与全局微网变量映射不完整')
    return mapping


def build_columns(m,x):
    from capability import build_resource_model
    from power_model import solve_lp,residual
    start=time.perf_counter();md=m.metadata;c=md['cfg'];v=m.var_indices;fc=md['forecast']
    modes=np.rint(x[v['battery_mode']][:,:,md['mode_map']])
    local=build_resource_model(c,md['state'],fc,modes);mapping=index_map(m,local)
    reference=np.asarray(x)[mapping].copy()
    if residual(local,reference)>2e-5:raise RuntimeError('全局候选不是局部完整轨迹可行见证，不能生成域')
    vertices=[reference];labels=['verified_incumbent'];idx=local.var_indices;n=len(fc['hours']);probs=fc['probabilities']
    dt=c['dt'];limit=int(option(c,'domain.max_columns',6));directions=[]
    def direction(label,weights,sign=-1.):
        obj=np.zeros_like(local.c)
        obj[idx['pcc_p']]=sign*probs[:,None,None]*dt*np.asarray(weights)[None,None,:]
        directions.append((label,obj))
    direction('total_export',np.ones(n))
    direction('startup_support',(np.arange(n)<len(c['generator']['startup_profile_mw'])).astype(float))
    direction('solar_charging',np.maximum(.05,fc['pv'].mean(axis=(0,1))/max(1e-9,float(fc['pv'].mean(axis=(0,1)).max()))),sign=1.)
    demand_obj=np.zeros_like(local.c)
    demand_obj[idx['served_c']]=-probs[:,None,None]*dt*c['critical_weight']
    demand_obj[idx['served_g']]=-probs[:,None,None]*dt*c['general_weight']
    directions.append(('local_service',demand_obj))
    directions.append(('local_operation_cost',local.c.copy()))
    direction('late_export',np.linspace(.05,1.,n))
    direction('ramp_shape',np.linspace(-1.,1.,n))
    queries=[]
    for name,obj in directions:
        if len(vertices)>=limit:break
        r=solve_lp(local,obj);ok=r['x'] is not None and residual(local,r['x'])<=2e-5
        duplicate=ok and any(np.max(np.abs(z-r['x']))<1e-7 for z in vertices)
        queries.append({'direction':name,'status':r['status'],'feasible':ok,'duplicate':bool(duplicate),'seconds':r['seconds']})
        if ok and not duplicate:vertices.append(r['x']);labels.append(name)
    return {'local_model':local,'mapping':mapping,'V':np.column_stack(vertices),'labels':labels,'queries':queries,
            'generation_seconds':time.perf_counter()-start,'modes':modes,
            'scope':'固定电池模式和因果信息树的联合微网全轨迹凸包；共同lambda，不是所有离散模式的完整域'}


def restricted_master(m,context):
    """替换内部变量，并显式保留被替换列的当前界（含已执行前缀/PV更新）。"""
    from power_model import LinearModel
    if np.any(m.integrality):raise ValueError('轨迹列主问题只适用于已固定离散的连续模型')
    mapping=context['mapping'];V=context['V'];nc=V.shape[1]
    keep=np.ones(len(m.c),bool);keep[mapping]=False;remaining=np.flatnonzero(keep);nr=len(remaining)
    substitute=sparse.csr_matrix(m.A[:,mapping]@V)
    A=sparse.hstack([m.A[:,remaining],substitute],format='csr')
    bounds_rows=sparse.hstack([sparse.csr_matrix((len(mapping),nr)),sparse.csr_matrix(V)],format='csr')
    simplex=sparse.csr_matrix((np.ones(nc),(np.zeros(nc,int),nr+np.arange(nc))),shape=(1,nr+nc))
    A=sparse.vstack([A,bounds_rows,simplex],format='csr')
    reduced=LinearModel(np.r_[m.c[remaining],m.c[mapping]@V],np.zeros(nr+nc),
        np.r_[m.lb[remaining],np.zeros(nc)],np.r_[m.ub[remaining],np.ones(nc)],A,
        np.r_[m.row_lb,m.lb[mapping],1.],np.r_[m.row_ub,m.ub[mapping],1.],
        {'lambda':np.arange(nr,nr+nc)},dict(m.metadata))
    return reduced,remaining


def solve_with_domain(m,context):
    from power_model import solve_lp,residual
    reduced,remaining=restricted_master(m,context);r=solve_lp(reduced)
    if r['x'] is None or residual(reduced,r['x'])>2e-5:return {'x':None,'status':r['status'],'seconds':r['seconds']}
    weights=r['x'][reduced.var_indices['lambda']];x=np.zeros_like(m.c);x[remaining]=r['x'][:len(remaining)];x[context['mapping']]=context['V']@weights
    if residual(m,x)>2e-5:raise RuntimeError('域主问题恢复的全局指令未通过物理约束')
    summary={'used':True,'columns':context['V'].shape[1],'original_variables':len(m.c),'master_variables':len(reduced.c),
        'eliminated_local_variables':len(context['mapping']),'lambda':weights.tolist(),'lambda_min':float(weights.min()),
        'lambda_sum_error':abs(float(weights.sum())-1),'full_model_residual':residual(m,x),'seconds':r['seconds'],
        'scope':'当前离散方案/信息树下的限制主问题；目标为可行恢复下界，不是原MILP上界'}
    return {'x':x,'status':r['status'],'objective':float(m.c@x),'seconds':r['seconds'],'domain':summary}


def coordinate(m,record):
    """用完整轨迹域替换MGO内部决策列，优化后恢复可下发指令。"""
    from power_model import freeze,LinearModel,validate
    c=m.metadata['cfg'];ctx=build_columns(m,record['x']);fixed=freeze(m,record['x'])
    primary=solve_with_domain(fixed,ctx)
    if primary['x'] is None:raise RuntimeError('包含已校核参考列的域主问题无解，不能静默退出域协调')
    cost=m.metadata['cost_objective'];eps=c['restore_tolerance']*m.metadata['denominator']
    second=LinearModel(cost,np.zeros_like(m.integrality),fixed.lb,fixed.ub,
        sparse.vstack([fixed.A,sparse.csr_matrix(m.c.reshape(1,-1))],format='csr'),
        np.r_[fixed.row_lb,-np.inf],np.r_[fixed.row_ub,primary['objective']+eps],m.var_indices,m.metadata)
    result=solve_with_domain(second,ctx);best=result if result['x'] is not None else primary
    x=best['x'];check=validate(m,x)
    if not check['passed']:raise RuntimeError('域分解后的指令未通过网络/信息/任务联合校核')
    R=float((m.metadata['past_reward']-m.c@x)/m.metadata['denominator'])
    if R<record['R_plan']-c['restore_tolerance']-2e-5:raise RuntimeError('域协调丢失已包含参考轨迹的恢复水平')
    updated=dict(record);updated.update(x=x,R_plan=R,cost_plan=float(cost@x+m.metadata['cost_constant']),
        R_lower=max(record.get('R_lower',R),R),domain=best['domain'],source=record.get('source','verified')+'+trajectory_domain_master')
    updated['domain'].update(generation_seconds=ctx['generation_seconds'],directions=ctx['labels'],query_count=len(ctx['queries']))
    m.metadata['domain_context']=ctx;m.metadata['domain_certificate']=updated['domain']
    return updated
