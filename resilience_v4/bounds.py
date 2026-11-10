"""最大恢复目标的外界：整数松弛 + 有限盒上的拉格朗日下界。

只放松整数性，不固定整数、不加入轨迹内近似限制。即使数值对偶残差
不为零，也将剩余线性项在有界变量盒上最小化，避免把LP可行目标误当界。
同一模型的多个上界取min；有限策略并集的上界取max。
"""
import time
import numpy as np
from scipy import sparse
from scipy.optimize import linprog


def class_bounds(items):
    """items是同一动作下不同因果策略的求解记录，不丢弃尚无解的候选。"""
    lower=max((r['R_lower'] for r in items if r.get('R_lower') is not None),default=None)
    possible=[r for r in items if not r.get('certified_infeasible',False)]
    upper=max((r.get('R_upper') if r.get('R_upper') is not None else 1. for r in possible),default=None)
    return lower,upper


def finite_box(m):
    """原物理恒等式推出的PCC界；不收紧真实可行集。一般模型需显式有界。"""
    lo=m.lb.copy();hi=m.ub.copy();md=m.metadata
    if 'cfg' in md and 'forecast' in md:
        c=md['cfg'];v=m.var_indices;fc=md['forecast'];start=md['state']['step']
        for j,mg in enumerate(c['microgrids']):
            node=c['mg_nodes'][j];d=c['profiles']['critical'][node,start:]+c['profiles']['general'][node,start:]
            if 'pcc_p' in v:
                cols=v['pcc_p'][:,j,:]
                lo[cols]=np.maximum(lo[cols],-mg['charge_mw']-d)
                hi[cols]=np.minimum(hi[cols],fc['pv'][:,j,:]+mg['dg_max_mw']+mg['discharge_mw'])
            if 'pcc_q' in v:
                cols=v['pcc_q'][:,j,:];qd=d*c['load']['reactive_ratio'][node]
                lo[cols]=np.maximum(lo[cols],-mg['inverter_mva']-np.maximum(qd,0))
                hi[cols]=np.minimum(hi[cols],mg['inverter_mva']-np.minimum(qd,0))
    return lo,hi


def lagrangian_lower(c,A_ub,b_ub,A_eq,b_eq,lower,upper,ineq_dual,eq_dual):
    """任意λ<=0、自由μ均给min原问题的下界；盒项包含全部站立性余项。"""
    lam=np.minimum(np.asarray(ineq_dual,float),0.);mu=np.asarray(eq_dual,float)
    if not (np.isfinite(lam).all() and np.isfinite(mu).all()):return None
    remainder=np.asarray(c)-A_ub.T@lam-A_eq.T@mu
    # 无界列的非零余项会导致-inf；不能靠四舍五入抹去。
    chosen=np.where(remainder>=0,lower,upper)
    if np.any(~np.isfinite(chosen)&(remainder!=0)):return None
    box=np.where(remainder==0,0,remainder*np.where(np.isfinite(chosen),chosen,0))
    terms=np.r_[np.asarray(b_ub)*lam,np.asarray(b_eq)*mu,box].astype(np.longdouble)
    guard=float(1e-9*(1+np.abs(terms).sum()))
    value=float(terms.sum())-guard
    if not np.isfinite(value):return None
    return {'cost_lower_bound':value,'roundoff_guard':guard,
            'box_remainder_l1':float(np.abs(box).sum()),'dual_sign_projected':True}


def relaxation_bound(m,seconds=90.):
    """独立有时限的完整LP松弛；超时无有效对偶时明确未取得新界。"""
    start=time.perf_counter();lo,hi=finite_box(m)
    answer={'source':'LP_relaxation_lagrangian','valid':False,'seconds':0.,'scope':'当前完整MILP的连续松弛；不含轨迹列限制'}
    if np.any(lo>hi+1e-9):
        answer.update(status=2,reason='原物理界矛盾',certified_infeasible=True);return answer
    eq=np.isfinite(m.row_lb)&np.isfinite(m.row_ub)&(np.abs(m.row_lb-m.row_ub)<1e-10)
    low=np.isfinite(m.row_lb)&~eq;high=np.isfinite(m.row_ub)&~eq
    ae=m.A[eq];be=m.row_lb[eq];au=sparse.vstack([m.A[high],-m.A[low]],format='csr');bu=np.r_[m.row_ub[high],-m.row_lb[low]]
    res=linprog(m.c,A_ub=au,b_ub=bu,A_eq=ae,b_eq=be,bounds=list(zip(lo,hi)),method='highs',
                options={'time_limit':max(.001,float(seconds)),'presolve':True})
    answer.update(status=int(res.status),message=str(res.message),seconds=time.perf_counter()-start,
                  certified_infeasible=bool(res.status==2),bounded_columns=int(np.isfinite(lo).sum()+np.isfinite(hi).sum()))
    y=getattr(getattr(res,'ineqlin',None),'marginals',None);z=getattr(getattr(res,'eqlin',None),'marginals',None)
    if y is None or z is None:return answer
    certificate=lagrangian_lower(m.c,au,bu,ae,be,lo,hi,y,z)
    if certificate is None:return answer
    md=m.metadata;upper=(md['past_reward']-certificate['cost_lower_bound'])/md['denominator']
    universal=min(1.,(md['past_reward']+md['remaining_reward_max'])/md['denominator'])
    answer.update(certificate,valid=True,R_upper=float(min(universal,upper)),LP_objective=None if res.fun is None else float(res.fun))
    if res.x is not None:
        ax=m.A@res.x
        answer['LP_primal_residual']=float(max(np.max(m.row_lb-ax,initial=0),np.max(ax-m.row_ub,initial=0),np.max(lo-res.x,initial=0),np.max(res.x-hi,initial=0)))
    return answer


def apply_bound(record,certificate,tolerance=2e-5):
    """原记录就地收紧，保存数值矛盾而不是静默裁掉错误界。"""
    record.setdefault('bound_certificates',[]).append(certificate)
    if certificate.get('certified_infeasible'):
        if record.get('x') is not None:raise RuntimeError('松弛不可行与已校核整数可行解矛盾')
        record['certified_infeasible']=True;record['status']=2
    if not certificate.get('valid'):return
    value=certificate['R_upper'];old=record.get('R_upper');lower=record.get('R_lower')
    if lower is not None and value<lower-tolerance:raise RuntimeError('LP外界低于可行解下界，需检查模型/数值证书')
    record['R_upper']=float(max(lower if lower is not None else -np.inf,min(old if old is not None else 1.,value)))
    if lower is not None:record['absolute_R_gap']=max(0.,record['R_upper']-lower)


def scenario_relaxation_bound(m,seconds=120.):
    """放松NA后的分场景LP外界；保留每场景全部24h网络/能源/任务约束。

    只用于给外界，不作为可执行完美信息策略。外部因果作业树固定时，
    非NA行可按场景分块。若发现其他跨场景行，则回退完整LP，不擅自删除。
    """
    from power_model import LinearModel
    md=m.metadata;fc=md.get('forecast',{});probs=np.asarray(fc.get('probabilities',[]))
    if len(probs)<2 or not fc.get('crew_plan'):return relaxation_bound(m,seconds)
    ns=len(probs);owner=np.full(len(m.c),-1,int)
    for key,idx in m.var_indices.items():
        idx=np.asarray(idx)
        if idx.ndim>=2 and idx.shape[0]==ns:
            for s in range(ns):owner[idx[s].ravel()]=s
    if np.any(np.abs(m.c[owner<0])>1e-12):return relaxation_bound(m,seconds)
    names=md.get('row_names')
    if names is None or len(names)!=m.A.shape[0]:return relaxation_bound(m,seconds)
    rowmask=np.array([not name.startswith('NA_') for name in names]);A=m.A[rowmask];lower=m.row_lb[rowmask];upper=m.row_ub[rowmask]
    # 每个保留行至多含一个场景的非零列；全局独立行可逐块保留。
    coo=A.tocoo();rowmin=np.full(A.shape[0],ns,int);rowmax=np.full(A.shape[0],-1,int);owned=owner[coo.col]>=0
    np.minimum.at(rowmin,coo.row[owned],owner[coo.col[owned]]);np.maximum.at(rowmax,coo.row[owned],owner[coo.col[owned]])
    if np.any((rowmax>=0)&(rowmin!=rowmax)):return relaxation_bound(m,seconds)
    lo,hi=finite_box(m);results=[];start=time.perf_counter();total=0.;any_valid=False
    global_cap=min(1.,(md['past_reward']+md['remaining_reward_max'])/md['denominator'])
    for s,prob in enumerate(probs):
        allowance=(seconds-(time.perf_counter()-start))/max(1,ns-s)
        if allowance<=.01:
            results.append({'scenario':s,'valid':False,'R_upper':float(prob*global_cap),'source':'universal_demand'});total+=prob*global_cap;continue
        cols=np.flatnonzero((owner==s)|(owner<0));rows=(rowmax==s)|(rowmax<0)
        child=LinearModel(m.c[cols],np.zeros(len(cols)),lo[cols],hi[cols],A[rows][:,cols],lower[rows],upper[rows],{},
            {'past_reward':float(prob*md['past_reward']),'denominator':md['denominator'],'remaining_reward_max':float(prob*md['remaining_reward_max'])})
        r=relaxation_bound(child,allowance);results.append({'scenario':s,**r})
        if r.get('certified_infeasible'):
            return {'source':'scenario_information_relaxation','valid':False,'certified_infeasible':True,'seconds':time.perf_counter()-start,'scenarios':results}
        any_valid=any_valid or r.get('valid',False);total+=r['R_upper'] if r.get('valid') else prob*global_cap
    return {'source':'scenario_information_relaxation','valid':any_valid,'R_upper':float(min(global_cap,total)),
        'seconds':time.perf_counter()-start,'scenarios':results,'relaxed_NA_rows':int((~rowmask).sum()),
        'scope':'原模型放松信息一致性后的可分场景外界；每块有拉格朗日证书或普适需求界；不是可执行策略'}


def model_fingerprint(m):
    """缓存只复用于数值矩阵、界、目标和整数性逐项一致的同一个问题。"""
    import hashlib
    h=hashlib.sha256(str(m.A.shape).encode())
    # 原始可行域相同而已积累收益/归一化变化时，R界不能沿用。
    for name in ('past_reward','denominator','remaining_reward_max'):
        h.update(name.encode());h.update(repr(m.metadata.get(name)).encode())
    for array in (m.c,m.integrality,m.lb,m.ub,m.row_lb,m.row_ub,m.A.indptr,m.A.indices,m.A.data):
        a=np.ascontiguousarray(array);h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()


def reuse_cached_bounds(record,previous,fingerprint):
    """同一数值模型重复求解时保留已有MIP/LP外界与可行下界。"""
    for certificate in previous.get('bound_certificates',[]):apply_bound(record,certificate)
    if previous.get('R_upper') is not None:
        apply_bound(record,{'source':'same_model_cached_upper','valid':True,'R_upper':previous['R_upper'],
            'model_fingerprint':fingerprint,'scope':'同一完整模型已有有效上界；不是固定整数LP目标'})
    if previous.get('R_lower') is not None:
        record['R_lower']=max(previous['R_lower'],record['R_lower']) if record.get('R_lower') is not None else previous['R_lower']
    if record.get('R_lower') is not None and record.get('R_upper') is not None:
        if record['R_upper']<record['R_lower']-2e-5:raise RuntimeError('同模型缓存上下界冲突，停止复用')
        record['R_upper']=max(record['R_upper'],record['R_lower'])
        record['absolute_R_gap']=record['R_upper']-record['R_lower']
    return record
