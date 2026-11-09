"""唯一入口：参数Excel→台风故障→联合滚动仿真→域与D相图→结果和图片。

运行本文件即可。保留原来的三个创新点，风灾模型只负责场景输入。
所有附加分析、求解限时和数据均在同目录“参数数据.xlsx”。
"""
from pathlib import Path
from datetime import datetime
import copy,shutil,time
from data import load_parameters,save_json,write_workbook
from network import prepare
from weather import make_world
from analysis import simulate,compute_capability,phase_map
from plots import plot_all,plot_layout
from diagnostics import audit_results

PARAMETER_FILE=Path(__file__).resolve().with_name('参数数据.xlsx')

def main(parameter_file=None,output_root=None):
    start=time.perf_counter();path=Path(parameter_file or PARAMETER_FILE).resolve();c,settings=load_parameters(path);prepare(c)
    base=Path(output_root or settings['output_dir']);base=base if base.is_absolute() else path.parent/base
    out=base/datetime.now().strftime('运行_%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    shutil.copy2(path,out/'本次参数.xlsx');save_json(out/'运行设置.json',settings)
    print(f'[1/6] r2修复版：33配电＋2主网节点，24h/{c["n_steps"]}时段，3微网；保留原三个创新点。',flush=True)
    world=make_world(c);save_json(out/'风灾实际样本_仅用于环境与事后统计.json',world)
    print(f'[2/6] 风场生成初始{sum(world["failure_hour"]<=0)}处、全天{sum(world["failure_hour"]<24)}处故障；未来真实信息不进入优化器。',flush=True)
    plot_layout(c,world,out/'figures')
    policies=settings['policies'] if settings['run_comparisons'] else ['joint']
    if 'joint' not in policies:policies=['joint']+policies
    results=[];capture=None
    print('[3/6] 执行完整24h事件驱动滚动，优化开关、启机、微网和当前抢修任务……',flush=True)
    for policy in policies:
        r,initial=simulate(c,world,policy,out/policy,capture_initial=(policy=='joint'))
        results.append(r)
        if policy=='joint':capture=initial
    if settings['run_coordination_baseline']:
        print('[对照] 开始禁止向上支撑实验；下列joint表示仍优化抢修，但主配接口禁止反送。',flush=True)
        baseline=copy.deepcopy(c);baseline['allow_upward']=0
        r,_=simulate(baseline,world,'joint',out/'no_upward')
        r['policy']='no_upward';save_json(out/'no_upward/executed_result.json',r);results.append(r)
    primary=next(r for r in results if r['policy']=='joint');cap=phase=None
    if capture is None:raise RuntimeError('没有可用的初始条件模型')
    model,solver,state=capture
    print('[4/6] 多时段条件域与整轨迹分解……',flush=True)
    if settings['compute_capability']:cap=compute_capability(model,solver['x'],c,out/'capability')
    print('[5/6] 初始D_R相图：每格重新优化两分支……',flush=True)
    if settings['compute_phase_map']:phase=phase_map(c,state,out/'phase')
    quality=audit_results(c,results,cap,phase);save_json(out/'结果质量检查.json',quality)
    write_workbook(primary,phase,out/'结果汇总.xlsx',world=world,comparisons=results,capability=cap,diagnostics=quality)
    print('[6/6] 输出真实结果图、比较和校核记录……',flush=True)
    plot_all(c,world,primary,out/'figures',cap,phase,results)
    summary={'policies':{r['policy']:r['summary'] for r in results},'elapsed_seconds':time.perf_counter()-start,
             'capability_decomposition':None if cap is None else cap['decomposition_success'],
             'quality_warning_count':len(quality['warnings']),
             'note':'r2：多模式候选、滚动MGO能量包络、资源/网络域分离；有限样本限时策略，ADMM尚未实现'}
    save_json(out/'运行汇总.json',summary)
    print(f'完成：{out.resolve()}\n实际加权恢复面积比：{100*primary["summary"]["actual_weighted_recovery"]:.2f}%\n实际EENS：{primary["summary"]["actual_EENS_MWh"]:.4f} MWh',flush=True)
    print('先打开结果汇总.xlsx和figures；未判相图不得解释为已证明最优。',flush=True)
    for warning in quality['warnings']:print('质量检查：'+warning,flush=True)
    return out

if __name__=='__main__':main()
