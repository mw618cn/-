"""结果质量检查：区分约束可行、求解质量和创新点证据。"""
import numpy as np


def audit_results(c,results,capability=None,phase=None):
    report={'version':'2026-11-10-r3','warnings':[],'policies':{},'comparisons':[]}
    for r in results:
        branches=[b for q in r['rounds'] for b in q['branches'].values()]
        discharge=np.asarray(r['discharge']);active=np.flatnonzero(discharge.sum(axis=1)>1e-7)
        served_c=np.asarray(r['served_c']);served_g=np.asarray(r['served_g'])
        computed=c['dt']*(c['critical_weight']*served_c.sum()+c['general_weight']*served_g.sum())/c['denominator']
        total=(c['profiles']['critical']+c['profiles']['general']).T
        eens=c['dt']*(total-served_c-served_g).sum()
        error=max(abs(computed-r['summary']['actual_weighted_recovery']),abs(eens-r['summary']['actual_EENS_MWh']))
        if error>2e-5:raise RuntimeError('结果统计与实际供电轨迹不一致，停止交付图表')
        info={'branch_solves':len(branches),'unclosed_bound_count':sum(b['status']==1 for b in branches),
              'max_R_gap':max((b.get('absolute_R_gap',max(0.,b['R_upper']-b['R_lower'])) for b in branches if b.get('R_lower') is not None),default=0.),
              'discharge_MWh':float(discharge.sum()*c['dt']),
              'first_discharge_hour':None if not len(active) else float(r['time_hours'][active[0]]),
              'statistics_error':float(error),
              'interface_envelopes_per_round':[q.get('interface_envelope_count',0) for q in r['rounds']]}
        report['policies'][r['policy']]=info
        if info['unclosed_bound_count']:report['warnings'].append(f"{r['policy']}有{info['unclosed_bound_count']}/{len(branches)}次分支未证明最优；实际指标不等于最优韧性。")
        if info['first_discharge_hour'] is not None and info['first_discharge_hour']>=20:
            report['warnings'].append(f"{r['policy']}储能首次放电较晚，需结合候选比较、负荷和预测解释，不能直接作为储能无效的结论。")
    joint=next(r for r in results if r['policy']=='joint')
    for r in results:
        if r is joint:continue
        same=all(np.allclose(joint[k],r[k],atol=1e-7,rtol=0) for k in ('recovery','closed','discharge','served_c','served_g'))
        gain=joint['summary']['actual_weighted_recovery']-r['summary']['actual_weighted_recovery']
        scope='允许主配反送的单路径贡献；非三层完全分离基线' if r['policy']=='no_upward' else '抢修日程规则对照，双方均保留电力协同'
        report['comparisons'].append({'baseline':r['policy'],'same_trajectory':same,'delta_R':float(gain),
                                      'delta_EENS_MWh':r['summary']['actual_EENS_MWh']-joint['summary']['actual_EENS_MWh'],'scope':scope})
        if same:report['warnings'].append(f"joint与{r['policy']}执行轨迹相同，本样本未体现该项增益；不得人为拉开曲线。")
    if phase:
        certified=sum(bool(q['certified']) for q in phase['records'])
        report['phase']={'grid_points':len(phase['records']),'certified_points':certified}
        if certified<len(phase['records']):report['warnings'].append(f"相图仅{certified}/{len(phase['records'])}格取得判据；未判点不能用来认定临界线。")
    if capability:
        if not capability['decomposition_success']:raise RuntimeError('整轨迹分解失败，停止输出成功证书')
        zero=[q['hour'] for q in capability['slices'] if q['points'] and np.max(np.abs(q['points']))<1e-7]
        report['capability']={'decomposition_success':True,'resource_zero_hours':zero,
                              'tracking_error':max(capability['tracking_p_max'],capability['tracking_q_max'])}
    report['scope']='有限因果策略类、有限预测样本的数值校核；不证明一般POMDP最优、样本外递归可行或暂态安全；ADMM未实现。'
    primary=joint
    steps=primary.get('controller_steps',[])
    if len(steps)!=c['n_steps']:raise RuntimeError('控制器没有对每个实际时段执行求解/校核记录')
    if any(abs(q['horizon_end_hour']-c['horizon_hours'])>1e-8 or abs(q['execute_step_minutes']-c['dt']*60)>1e-8 for q in steps):raise RuntimeError('研究终点漂移或一次执行了多于首步的控制')
    report['rolling']={'controller_steps':len(steps),'joint_MILP':sum(q['solve']=='joint_MILP' for q in steps),
                       'continuous_LP':sum(q['solve']=='fixed_plan_LP' for q in steps),'backward_passed_steps':sum(bool(q['backward_passed']) for q in steps)}
    max_pq=0.;max_soc=0.
    for r in results:
        load=np.asarray(r['served_c'])+np.asarray(r['served_g']);pr=-load.copy();qr=-load*c['load']['reactive_ratio'][None,:]
        pr[:,0]+=np.asarray(r['grid_p'])+np.asarray(r['target_p'])-np.asarray(r['aux'])
        qr[:,0]+=np.asarray(r['grid_q'])+np.asarray(r['target_q'])-np.asarray(r['aux'])*np.tan(np.arccos(c['generator']['aux_power_factor']))
        for j,node in enumerate(c['mg_nodes']):
            pr[:,node]+=np.asarray(r['pv'])[:,j]+np.asarray(r['dg'])[:,j]+np.asarray(r['discharge'])[:,j]-np.asarray(r['charge'])[:,j]
            qr[:,node]+=np.asarray(r['mg_q'])[:,j]
            mg=c['microgrids'][j];error=np.diff(np.asarray(r['energy'])[:,j])-c['dt']*(mg['eta_charge']*np.asarray(r['charge'])[:,j]-np.asarray(r['discharge'])[:,j]/mg['eta_discharge'])
            max_soc=max(max_soc,float(np.max(np.abs(error))))
        for e,line in enumerate(c['edges']):
            pr[:,line['from']]-=np.asarray(r['flow_p'])[:,e];pr[:,line['to']]+=np.asarray(r['flow_p'])[:,e]
            qr[:,line['from']]-=np.asarray(r['flow_q'])[:,e];qr[:,line['to']]+=np.asarray(r['flow_q'])[:,e]
        max_pq=max(max_pq,float(np.max(np.abs(pr))),float(np.max(np.abs(qr))))
        if np.any((np.asarray(r['closed'])>.5)&(np.asarray(r['damaged'])>.5)):raise RuntimeError('实际闭合了尚未修复的线路')
        if np.max(np.minimum(np.asarray(r['charge']),np.asarray(r['discharge'])))>2e-5:raise RuntimeError('实际存在同时充放电')
    if max_pq>2e-5 or max_soc>2e-5:raise RuntimeError('实际执行轨迹的独立功率/能量核查失败')
    report['actual_physics']={'nodal_PQ_max_residual':max_pq,'SOC_max_residual':max_soc,'damaged_line_closed':False}
    ratio=max(c['generator']['startup_profile_mw'])/c['generator']['max_mw']
    report['parameter_scale']={'aux_peak_ratio':float(ratio),'aux_energy_MWh':float(c['dt']*sum(c['generator']['startup_profile_mw'])),'calibration':'MATPOWER接线/RX/PQ可追溯；气象/道路/机组与损伤先验仍为合成参数'}
    if ratio>.15:report['warnings'].append('等值机组辅机峰值比例偏高，须以真实启动曲线校准；不能直接当成大型火电/燃机实证。')
    return report
