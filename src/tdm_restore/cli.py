"""命令行入口：把配置、求解、校核、图表组织为一条可重复流水线。"""
from __future__ import annotations
import argparse
from pathlib import Path
from .config import load_config


def main(argv=None):
    """解析命令；默认完整24h、八策略、两条件分支，限时参数集中可调。"""
    p=argparse.ArgumentParser(description="24h主配微韧性恢复研究原型（自设参数）")
    p.add_argument("--config",help="JSON配置文件；省略采用内置参数")
    p.add_argument("--output",default="results/demo",help="结果目录，默认results/demo")
    p.add_argument("--policies",default="all",help="all或逗号分隔ID，如A_11,B_10")
    p.add_argument("--branches",default="start,wait",help="start,wait；none只解free")
    p.add_argument("--time-limit",type=float,help="每个MILP的最大秒数，不是全实验时限")
    p.add_argument("--capability",action="store_true",help="固定离散最优方案，计算24h条件域与全轨迹分解")
    p.add_argument("--baselines",action="store_true",help="增加同菜单禁止向上反送基线（额外8次求解）")
    p.add_argument("--rolling",action="store_true",help="对选定真实场景按实际观测重求解，固定已执行前缀")
    p.add_argument("--realized-scenario",default="HH",help="滚动回放的真实损伤场景ID")
    p.add_argument("--rolling-hours",help="逗号分隔重算时刻h；默认0.5,1,1.5,2,4,8,12,18，可加密至15min")
    p.add_argument("--write-config",help="只导出完整默认/合并配置到指定JSON")
    args=p.parse_args(argv)
    cfg=load_config(args.config, {"solver":{"time_limit":args.time_limit}} if args.time_limit is not None else None)
    if args.write_config:
        from .experiments import save_json
        save_json(args.write_config,cfg);print(f"参数已写入 {args.write_config}");return 0
    branches=() if args.branches in ("none","free","") else tuple(args.branches.split(","))
    if not set(branches)<={"start","wait"}:p.error("branches仅支持start,wait或none")
    from .experiments import run_experiment
    sol,comparison=run_experiment(cfg,args.output,args.policies,branches,args.capability,args.baselines)
    if args.rolling:
        import numpy as np
        from .rolling import run_rolling_replay
        from .crew import enumerate_policies
        from .model import build_model
        policy=next(v for v in enumerate_policies(cfg) if v["id"]==sol["policy_id"])
        model=build_model(cfg,policy)
        checkpoint=Path(args.output)/"checkpoints"/f"{policy['id']}_free_coord.npz"
        result={**sol["solver"],"x":np.load(checkpoint)["x"]}
        kwargs={}
        if args.rolling_hours:
            kwargs["decision_hours"]=tuple(float(h) for h in args.rolling_hours.split(","))
        replay=run_rolling_replay(cfg,policy,model,result,args.realized_scenario, outdir=Path(args.output)/"rolling",**kwargs)
        from .plotting import plot_rolling
        plot_rolling(replay,Path(args.output)/"figures")
    print(f"完成：{Path(args.output).resolve()}；J={sol['summary']['expected_cost']:.4f}千元；{comparison.get('decision','仅自由分支')}")
    return 0
