"""参数扫描命令行：实际计算D(alpha,SOC)，或只汇总已有结果。

脚本的 --help 与 --dry-run 不调用求解器。默认实际扫描3个点；每点
会对所选菜单进行free及条件分支MILP，完整all菜单可能需要较长时间。
"""
from __future__ import annotations

import argparse
import copy
import glob
import itertools
from pathlib import Path
import sys

# 从源码目录即可运行，兼容尚未执行pip install -e .的首次使用者。
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "src"))


def _numbers(text):
    """解析逗号分隔数列，保留用户明确指定的采样位置。"""
    try:
        values=[float(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("应为逗号分隔数字，例如0,0.5,1") from exc
    if not values:
        raise argparse.ArgumentTypeError("数列不可为空")
    return values


def _pairs(text):
    """解析alpha:SOC点列表；不要求完整矩形网格。"""
    try:
        values=[tuple(float(value) for value in pair.strip().split(":"))
                for pair in text.split(",") if pair.strip()]
        if not values or any(len(pair)!=2 for pair in values):
            raise ValueError("每点须有两个值")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("应为alpha:SOC列表，例如0:0.7,1:0.7,0:0.4") from exc
    return values


def main(argv=None):
    """先解析参数；只有实际扫描分支才导入并调用完整实验流水线。"""
    parser=argparse.ArgumentParser(description="24h恢复的D(alpha,SOC)真实参数扫描；区分有效界与估计值")
    parser.add_argument("--config",help="基础JSON配置，默认内置演示参数")
    parser.add_argument("--output",default="results/sensitivity",help="扫描或汇总输出目录")
    parser.add_argument("--alphas",type=_numbers,default=[0.,.5,1.],help="外部剩余支撑比例，默认0,0.5,1")
    parser.add_argument("--socs",type=_numbers,default=[.7],help="三个微网共同初始SOC，默认0.7")
    parser.add_argument("--pairs",type=_pairs,help="明确指定点，如0:0.7,1:0.7；覆盖alphas/socs")
    parser.add_argument("--menu","--policies",dest="menu",default="all",help="all或策略ID逗号列表，如A_11,B_10")
    parser.add_argument("--time-limit",type=float,help="每个MILP限时秒数；并非整次扫描时限")
    parser.add_argument("--no-reuse",action="store_true",help="不复用相同配置与菜单的既有结果")
    parser.add_argument("--summarize",nargs="+",help="仅汇总已有结果目录，可写通配符，不进行求解")
    parser.add_argument("--dry-run",action="store_true",help="仅校核配置并导出扫描计划，不求解")
    args=parser.parse_args(argv)
    from tdm_restore.config import load_config
    from tdm_restore.experiments import save_json
    from tdm_restore.sensitivity import run_sensitivity,summarize_cases
    if args.summarize:
        paths=[]
        for pattern in args.summarize:
            paths.extend(glob.glob(pattern) or [pattern])
        summary=summarize_cases(paths,args.output)
        print(f"已汇总{len(summary['cases'])}个算例；读取错误{len(summary['errors'])}个；{Path(args.output).resolve()}")
        return 0 if summary["cases"] else 1
    cfg=load_config(args.config)
    if args.dry_run:
        points=args.pairs if args.pairs is not None else list(itertools.product(args.alphas,args.socs))
        if len(set(points))!=len(points):
            parser.error("扫描点重复；请每个(alpha,SOC)只指定一次")
        plan=[]
        for alpha,soc in points:
            patch=copy.deepcopy(cfg);patch["alpha"]=alpha
            for mg in patch["microgrids"]:
                mg["soc_initial"]=soc
            if args.time_limit is not None:
                if args.time_limit<=0:
                    parser.error("time-limit必须为正")
                patch["solver"]["time_limit"]=args.time_limit
            checked=load_config(overrides=patch)
            plan.append({"alpha":alpha,"soc":soc,"menu":args.menu,
                         "per_milp_time_limit_seconds":checked["solver"]["time_limit"],
                         "n_steps":checked["n_steps"],"horizon_hours":checked["horizon_hours"]})
        save_json(Path(args.output)/"sensitivity_plan.json",{"points":plan,"note":"仅参数校核，未求解"})
        print(f"扫描计划{len(plan)}点已校核，未调用求解器：{Path(args.output).resolve()}")
        return 0
    result=run_sensitivity(cfg,args.output,args.alphas,args.socs,args.pairs,args.menu,
                           args.time_limit,not args.no_reuse)
    print(f"扫描完成：{len(result['cases'])}点；失败{len(result['run_failures'])}点；{Path(args.output).resolve()}")
    return 0 if result["cases"] else 1


if __name__=="__main__":
    raise SystemExit(main())
