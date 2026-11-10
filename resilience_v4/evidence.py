"""由真实求解数据形成方法证据，不由绘图函数补造认证或违规时刻。"""
import json
import numpy as np
from scipy import sparse


def information_evidence(c,m,x):
    from power_model import extract
    fc=m.metadata['forecast'];plan=fc['crew_plan'];ns=len(fc['probabilities']);n=len(fc['hours']);probs=fc['probabilities']
    data=[extract(m,x,s) for s in range(ns)];nodes=[];groups_per_step=[];node_ids={};maxspread=0.;posterior=np.zeros((ns,len(fc['hidden_targets']),n))
    for t,h in enumerate(fc['hours']):
        groups={}
        for s in range(ns):groups.setdefault(plan['histories'][s][t],[]).append(s)
        groups_per_step.append(len(groups))
        for history,ids in groups.items():
            key=json.dumps(history,ensure_ascii=False);identifier=node_ids.setdefault(key,len(node_ids));mass=float(probs[ids].sum())
            for j,e in enumerate(fc['hidden_targets']):
                p=float(sum(probs[s] for s in ids if fc['severity'][s][e]=='heavy')/mass)
                posterior[ids,j,t]=p
            lead=ids[0];controls=np.array([np.r_[data[s]['pcc_p'][:,t],data[s]['pcc_q'][:,t],data[s]['served_c'][:,t],data[s]['served_g'][:,t],data[s]['target_p'][t]] for s in ids])
            spread=float(np.max(np.abs(controls-controls[0])));maxspread=max(maxspread,spread)
            command=plan['dispatch'][lead][t];trace=plan['trace'][lead][t]
            start=bool(any(slot==t and data[lead]['target_start'][k]>.5 for k,slot in enumerate(m.metadata['start_slots'])))
            diagnoses=[{'line':c['edges'][int(e)]['name'],'type':typ} for e,typ in history[1]]
            crew='待命' if trace is None else c['edges'][trace[0]]['name']+('/巡检' if trace[1]=='inspect' else '/维修')
            nodes.append({'hour':float(h),'history_id':identifier,'scenarios':ids,'probability':mass,'diagnoses':diagnoses,
                'crew_action':crew,'new_dispatch':command is not None,'start_command':start,'task_active':trace is not None,
                'same_information_control_spread':spread})
    if maxspread>2e-5:raise RuntimeError('信息证据发现同信息组控制不一致')
    return {'hours':fc['hours'],'scenario_probabilities':probs,'hidden_targets':[c['edges'][e]['name'] for e in fc['hidden_targets']],
        'group_count':groups_per_step,'posterior_heavy':posterior,'nodes':nodes,'max_same_information_control_spread':maxspread,
        'target_selection':fc.get('target_selection',[]),'scope':'当前候选预测树的有限支持条件概率与控制；不是实际未来信息，不混合不同策略的样本'}


def splice_diagnostic(local,target_p,target_q,probabilities,dt,horizon):
    """首次不能延续的前缀 + 保持硬物理约束的最小归一化L1跟踪修正。"""
    from power_model import LinearModel,solve_lp,residual
    p=np.asarray(target_p);q=np.asarray(target_q);ns,n=p.shape;idx=local.var_indices;tests=[]
    def check(k):
        equalities=[]
        for s in range(ns):
            for t in range(k):
                for key,target in [('pcc_p',p),('pcc_q',q)]:equalities.append(({int(j):1. for j in idx[key][s,:,t]},float(target[s,t])))
        r=solve_lp(local,equalities=equalities);tests.append({'fixed_steps':k,'end_hour':k*dt,'status':r['status']})
        return r
    end=check(n);answer={'status':'infeasible' if end['status']==2 else 'feasible' if end['x'] is not None else 'unknown','prefix_tests':tests,
        'scope':'固定模式/信息树下完整剩余时域的PCC跟踪；前缀证书含原终端与安全后备目标，不是瞬时功率越限'}
    if answer['status']=='unknown':return answer
    first=None
    if answer['status']=='infeasible':
        initial=check(0)
        if initial['x'] is None:raise RuntimeError('局部资源模型本身无可行计划，不能把失败归于拼接')
        lo,hi=0,n
        while hi-lo>1:
            k=(lo+hi)//2;r=check(k)
            if r['x'] is not None:lo=k
            elif r['status']==2:hi=k
            else:answer.update(prefix_status='unknown');break
        if hi-lo==1:first=hi
    answer['first_nonextendable_prefix_end_hour']=None if first is None else first*dt
    answer['prefix_extendable']=[None if first is None and answer['status']=='infeasible' else int(first is None or k<first) for k in range(n+1)]
    size=len(local.c);nslack=4*ns*n;objective=np.zeros(size+nslack);rr=[];cc=[];vv=[];rhs=[]
    for s in range(ns):
        for t in range(n):
            for a,(key,target) in enumerate([('pcc_p',p),('pcc_q',q)]):
                row=len(rhs);sp=size+4*(s*n+t)+2*a;sm=sp+1
                for col in idx[key][s,:,t]:rr.append(row);cc.append(int(col));vv.append(1.)
                rr.extend([row,row]);cc.extend([sp,sm]);vv.extend([1.,-1.]);rhs.append(float(target[s,t]))
                objective[[sp,sm]]=probabilities[s]*dt/horizon
    extra=sparse.coo_matrix((vv,(rr,cc)),shape=(len(rhs),size+nslack)).tocsr()
    A=sparse.vstack([sparse.hstack([local.A,sparse.csr_matrix((local.A.shape[0],nslack))]),extra],format='csr')
    phase=LinearModel(objective,np.zeros(size+nslack),np.r_[local.lb,np.zeros(nslack)],np.r_[local.ub,np.full(nslack,np.inf)],A,
        np.r_[local.row_lb,rhs],np.r_[local.row_ub,rhs],local.var_indices,local.metadata)
    r=solve_lp(phase)
    if r['status']!=0 or r['x'] is None:answer['tracking_status']='unknown';return answer
    slack=r['x'][size:].reshape(ns,n,4);answer.update(tracking_status='optimal',normalized_L1=r['objective'],
        P_correction_MWh=float(np.einsum('s,st->',probabilities,slack[:,:,:2].sum(axis=2))*dt),
        Q_correction_Mvarh=float(np.einsum('s,st->',probabilities,slack[:,:,2:].sum(axis=2))*dt),
        slack_scenario_0=slack[0],residual=residual(phase,r['x']),
        tracking_definition='Pbase=1MW,Qbase=1Mvar，按样本概率和时域平均的L1距离；修正分配可不唯一，不等于负荷EENS')
    return answer
