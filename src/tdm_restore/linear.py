"""稀疏MILP装配与HiGHS求解接口；同时保留可行上界、对偶下界和状态。

所有费用单位为千元。freeze_discrete仅固定已有离散日程，不能据此声称
原混合整数能力域凸或证明分布式算法全局收敛。
"""
from __future__ import annotations
from dataclasses import dataclass, field
import time
import numpy as np
from scipy import sparse
from scipy.optimize import milp, linprog, Bounds, LinearConstraint


@dataclass
class LinearModel:
    c: np.ndarray
    integrality: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    A: sparse.csr_matrix
    row_lb: np.ndarray
    row_ub: np.ndarray
    var_indices: dict
    metadata: dict=field(default_factory=dict)
    objective_constant: float=0.0


class ModelBuilder:
    """按有名变量块及稀疏行组装模型，避免手工大矩阵下标混乱。"""
    def __init__(self):
        self.c=[];self.lb=[];self.ub=[];self.integrality=[];self.var_indices={}
        self.rr=[];self.cc=[];self.vv=[];self.row_lb=[];self.row_ub=[];self.row_names=[]
        self.metadata={};self.objective_constant=0.0

    def add_var(self,name,shape,lb=0.0,ub=np.inf,integer=False,cost=0.0):
        """广播变量界与费用；返回与shape同形的全局列号。"""
        shape=(shape,) if isinstance(shape,int) else tuple(shape)
        size=int(np.prod(shape));idx=np.arange(len(self.c),len(self.c)+size).reshape(shape)
        for target,value in ((self.c,cost),(self.lb,lb),(self.ub,ub)):
            target.extend(np.broadcast_to(value,shape).ravel().tolist())
        self.integrality.extend([int(integer)]*size);self.var_indices[name]=idx
        return idx

    def add_row(self,coeff,lb=-np.inf,ub=np.inf,name=""):
        """加入lb<=Σa_j x_j<=ub；coeff为{列号:系数}。"""
        row=len(self.row_lb)
        for col,val in coeff.items():
            if abs(val)>1e-14:
                self.rr.append(row);self.cc.append(int(col));self.vv.append(float(val))
        self.row_lb.append(lb);self.row_ub.append(ub);self.row_names.append(name)

    def build(self):
        """输出SciPy CSR，保留约束名称便于定位不可行或数值偏差。"""
        A=sparse.coo_matrix((self.vv,(self.rr,self.cc)),shape=(len(self.row_lb),len(self.c))).tocsr()
        self.metadata["row_names"]=self.row_names
        return LinearModel(np.array(self.c),np.array(self.integrality),np.array(self.lb),np.array(self.ub),A,
                           np.array(self.row_lb),np.array(self.row_ub),self.var_indices,self.metadata,self.objective_constant)


def solve(model,solver_cfg=None):
    """求解MILP，限时但有incumbent仍返回可行解；未证明最优则保留间隙。"""
    cfg={"time_limit":25.0,"mip_rel_gap":0.005,"presolve":True,"disp":False}
    cfg.update(solver_cfg or {});start=time.perf_counter()
    result=milp(model.c,integrality=model.integrality,bounds=Bounds(model.lb,model.ub),
                constraints=LinearConstraint(model.A,model.row_lb,model.row_ub),options=cfg)
    x=result.x
    def finite(v,default):
        return float(v) if v is not None and np.isfinite(v) else default
    obj=finite(getattr(result,"fun",None),np.inf)+model.objective_constant
    lower=finite(getattr(result,"mip_dual_bound",None),-np.inf)+model.objective_constant
    # 纯LP或空分支有时未提供mip_dual_bound；已证最优时目标就是有效下界。
    if result.status==0 and not np.isfinite(lower):lower=obj
    return {"success":bool(result.success),"status":int(result.status),"message":result.message,"x":x,
            "objective":obj,"lower_bound":lower,"mip_gap":finite(getattr(result,"mip_gap",None),None),
            "solve_seconds":time.perf_counter()-start,"node_count":int(getattr(result,"mip_node_count",0) or 0),
            "absolute_objective_gap":float(obj-lower) if np.isfinite(obj-lower) else None,
            "relative_objective_gap_including_constant":float((obj-lower)/max(abs(obj),1e-9)) if np.isfinite(obj-lower) else None}


def freeze_discrete(model,x):
    """固定已校核解的整数变量得到条件LP；不改变连续跨时段约束。"""
    x=np.asarray(x);mask=model.integrality!=0
    if np.max(np.abs(x[mask]-np.rint(x[mask])),initial=0)>1e-5:
        raise ValueError("离散变量尚未满足整数性，不能固定生成能力域")
    lb=model.lb.copy();ub=model.ub.copy();lb[mask]=np.rint(x[mask]);ub[mask]=lb[mask]
    return LinearModel(model.c.copy(),np.zeros_like(model.integrality),lb,ub,model.A,
                       model.row_lb,model.row_ub,model.var_indices,{**model.metadata,"discrete_fixed":True},model.objective_constant)


def solve_lp(model,objective=None,extra_equalities=None):
    """解连续LP；extra_equalities=[({列:系数},右端),...]用于接口轨迹分解。"""
    if np.any(model.integrality):
        raise ValueError("solve_lp仅接受连续模型；请先freeze_discrete")
    c=model.c if objective is None else np.asarray(objective,dtype=float)
    finite_both=np.isfinite(model.row_lb)&np.isfinite(model.row_ub)
    eq=np.zeros(len(model.row_lb),dtype=bool)
    eq[finite_both]=np.abs(model.row_lb[finite_both]-model.row_ub[finite_both])<1e-10
    lo=np.isfinite(model.row_lb)&~eq;hi=np.isfinite(model.row_ub)&~eq
    Aeq=model.A[eq];beq=model.row_lb[eq]
    Aub=sparse.vstack((model.A[hi],-model.A[lo]),format="csr");bub=np.r_[model.row_ub[hi],-model.row_lb[lo]]
    if extra_equalities:
        rr=[];cc=[];vv=[];rhs=[]
        for i,(row,b) in enumerate(extra_equalities):
            rhs.append(b)
            for j,v in row.items():rr.append(i);cc.append(j);vv.append(v)
        new=sparse.coo_matrix((vv,(rr,cc)),shape=(len(rhs),len(c))).tocsr()
        Aeq=sparse.vstack((Aeq,new),format="csr");beq=np.r_[beq,rhs]
    start=time.perf_counter()
    res=linprog(c,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=list(zip(model.lb,model.ub)),method="highs")
    constant=model.objective_constant if objective is None else 0.0
    obj=float(res.fun)+constant if res.fun is not None else np.inf
    return {"success":bool(res.success),"status":int(res.status),"message":res.message,"x":res.x,
            "objective":obj,"lower_bound":obj if res.success else -np.inf,"mip_gap":0.0 if res.success else None,
            "solve_seconds":time.perf_counter()-start,"node_count":0}
