"""入口：Excel→因果预测与滚动控制→后向校核→结果→原版宋体期刊图。"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,copy,shutil,time,json
from data import load_parameters,save_json,write_workbook
from network import prepare
from weather import make_world
from analysis import simulate,compute_capability,phase_map
from diagnostics import audit_results
from information import option

PARAMETER_FILE=Path(__file__).resolve().with_name('参数数据.xlsx')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else None


def redraw(folder):
    """重画已保存结果；不为旧运行包伪造其没有的信息树或后向证据。"""
    from plots import plot_all
    folder=Path(folder).resolve();c,_=load_parameters(folder/'本次参数.xlsx');prepare(c)
    world=_read(folder/'world.json') or _read(folder/'风灾实际样本_仅用于环境与事后统计.json')
    if world is None:raise FileNotFoundError('缺少保存的风灾样本，不能从不同随机样本补图')
    # JSON用null表示未发生事件；转换回正无穷，不能当作0h故障。
    import numpy as np
    world['failure_hour']=np.array([np.inf if h is None else h for h in world['failure_hour']])
    for key in ('wind','hazard','interval_probability_if_intact','cumulative_first_failure_probability','first_failure_mass'):
        if key in world:world[key]=np.asarray(world[key])
    if 'probability_time_hours' not in world:
        # 只补概率定义，保留原始实际故障样本和粗粒度发生时间。
        probabilities=make_world(c)
        for key in ('probability_time_hours','interval_probability_if_intact','cumulative_first_failure_probability','first_failure_mass'):world[key]=probabilities[key]
    results=[]
    for name in ('joint','critical_first','nearest','no_upward'):
        r=_read(folder/name/'executed_result.json')
        if r is not None:results.append(r)
    if not results:raise FileNotFoundError('没有执行结果，不以预测计划冒充实际轨迹')
    primary=next(r for r in results if r['policy']=='joint')
    cap=_read(folder/'capability/capability.json');phase=_read(folder/'phase/phase_map.json');info=_read(folder/'initial_information.json')
    files=plot_all(c,world,primary,folder/'figures',cap,phase,results,information=info)
    print(f'重画完成：{len(files)}个图文件；没有原始证据的图不会补造。')
    return files


def main(parameter_file=None,output_root=None):
    start=time.perf_counter();path=Path(parameter_file or PARAMETER_FILE).resolve();c,settings=load_parameters(path);prepare(c)
    base=Path(output_root or settings['output_dir']);base=base if base.is_absolute() else path.parent/base
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('运行_%Y%m%d_%H%M%S_%f')
    out=base/stamp;out.mkdir(parents=True);shutil.copy2(path,out/'本次参数.xlsx');save_json(out/'运行设置.json',settings)
    save_json(out/'版本与范围.json',{'version':'r3','controller':'每步连续MPC+周期/事件联合MILP','bound_scope':'有限因果作业策略类','ADMM':False,'timezone':'Asia/Shanghai'})
    print(f'[1/6] R3：{len(c["nodes"])}节点、{len(c["edges"])}线路、{len(c["microgrids"])}微网；{c["horizon_hours"]:g}h/{c["n_steps"]}时段。',flush=True)
    world=make_world(c);save_json(out/'world.json',world)
    print(f'[2/6] 风速驱动首次故障；概率模型={world["probability_model"]}。合成参数须另行标定。',flush=True)
    policies=settings['policies'] if settings['run_comparisons'] else ['joint'];results=[];capture=None
    if 'joint' not in policies:policies=['joint']+policies
    print('[3/6] 观测→更新能力与信息树→求解→后向校核→只执行一步→更新状态。',flush=True)
    for policy in policies:
        r,initial=simulate(c,world,policy,out/policy,capture_initial=(policy=='joint'));results.append(r)
        if policy=='joint':capture=initial
    if settings['run_coordination_baseline']:
        print('[对照] 禁止主配接口向上支撑，其他资源及灾害保持相同。',flush=True)
        baseline=copy.deepcopy(c);baseline['allow_upward']=0
        r,_=simulate(baseline,world,'joint',out/'no_upward');r['policy']='no_upward'
        save_json(out/'no_upward/executed_result.json',r);results.append(r)
    primary=next(r for r in results if r['policy']=='joint');cap=phase=None
    if capture is None:raise RuntimeError('没有可核查的初始候选，停止出图')
    model,solver,state=capture
    information={'hidden_targets':model.metadata['forecast']['hidden_targets'],'crew_plan':model.metadata['forecast']['crew_plan'],
        'backward_certificate':model.metadata.get('backward_certificate'),'reserve_tube':model.metadata.get('reserve_tube'),
        'reference_energy':solver['x'][model.var_indices['energy']],
        'horizon_hours':c['horizon_hours'],'scope':'初始预测条件策略证据；不是实际未来观测'}
    save_json(out/'initial_information.json',information)
    import numpy as np
    np.savez_compressed(out/'initial_model_solution.npz',x=solver['x'])
    print('[4/6] 多时段条件资源域与全接口轨迹分解。',flush=True)
    if settings['compute_capability']:cap=compute_capability(model,solver['x'],c,out/'capability')
    print('[5/6] 按需计算当前状态D网格，保留求解界和未判点。',flush=True)
    if settings['compute_phase_map']:phase=phase_map(c,state,out/'phase')
    quality=audit_results(c,results,cap,phase);save_json(out/'结果质量检查.json',quality)
    write_workbook(primary,phase,out/'结果汇总.xlsx',world=world,comparisons=results,capability=cap,diagnostics=quality)
    plot_status='disabled'
    if option(c,'plots.enabled',True):
        print('[6/6] 宋体校验与IEEE列宽图形输出。',flush=True)
        try:redraw(out);plot_status='completed'
        except (FileNotFoundError,ValueError,RuntimeError) as error:
            # 保留数值结果，明确没有完成绘图，不用其他字体代替。
            plot_status='blocked';(out/'绘图未完成说明.txt').write_text(str(error),encoding='utf-8');print('绘图未完成：'+str(error),flush=True)
    summary={'policies':{r['policy']:r['summary'] for r in results},'elapsed_seconds':time.perf_counter()-start,
        'plots':plot_status,'quality_warnings':quality['warnings'],'scope':'有限因果策略类与有限预测样本；集中MILP/LP；不是任意POMDP最优或样本外安全证明'}
    save_json(out/'运行汇总.json',summary)
    print(f'完成：{out}\n实际恢复面积比：{100*primary["summary"]["actual_weighted_recovery"]:.2f}%\n实际EENS：{primary["summary"]["actual_EENS_MWh"]:.4f} MWh\n绘图状态：{plot_status}',flush=True)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parameters',default=None);parser.add_argument('--redraw',default=None)
    args=parser.parse_args()
    if args.redraw:redraw(args.redraw)
    else:main(args.parameters)
