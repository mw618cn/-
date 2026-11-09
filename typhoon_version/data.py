"""一份Excel的数据入口。行2是程序字段，行3中文解释，数据从行4开始。

读取顺序：基础参数→35母线/39支路→3微网→独立地理坐标→台风路径→96点
曲线→运行开关。所有默认算例数值在Excel中，不使用旧版A/B故障或32接线菜单。
"""
from pathlib import Path
import json,math
import numpy as np
from openpyxl import load_workbook,Workbook
from openpyxl.styles import Font,PatternFill,Alignment


def rows(book,name):
    if name not in book.sheetnames:raise ValueError(f'缺少工作表：{name}')
    ws=book[name];keys=[x.value for x in ws[2]]
    if any(k is None for k in keys) or len(keys)!=len(set(keys)):raise ValueError(f'{name}第2行字段不合法')
    return [{k:r[i] for i,k in enumerate(keys)} for r in ws.iter_rows(min_row=4,values_only=True) if r[0] is not None]

def boolean(v):
    if str(v).lower() in ('true','是','1','yes'):return True
    if str(v).lower() in ('false','否','0','no'):return False
    raise ValueError(f'应填是/否：{v}')

def number(v,name):
    try:x=float(v)
    except (TypeError,ValueError):raise ValueError(f'{name}须为数值（公式先在Excel计算保存）') from None
    if not math.isfinite(x):raise ValueError(f'{name}不是有限数')
    return x

def assign(c,k,v):
    path=k.split('.')
    for a in path[:-1]:c=c.setdefault(a,{})
    c[path[-1]]=v

def load_parameters(path):
    book=load_workbook(path,data_only=True)
    try:
        c={};seen=set()
        for r in rows(book,'基础参数'):
            k=str(r['parameter'])
            if k in seen:raise ValueError(f'参数重复：{k}')
            seen.add(k);assign(c,k,str(r['value']) if k=='case_name' else number(r['value'],k))
        required=['horizon_hours','step_minutes','base_mva','alpha','grid_reference_mw','grid_price','voltage_min','voltage_max',
                  'polygon_sides','battery_mode_minutes','switch_minutes','switch_min_hold_hours','switch_max_per_line','switch_max_total',
                  'switch_cost','critical_weight','general_weight','restore_tolerance','generator','crew','solver','weather']
        if set(required)-set(c):raise ValueError(f'缺少基础参数{set(required)-set(c)}')
        n=c['horizon_hours']/(c['step_minutes']/60)
        if n<1 or not np.isclose(n,round(n)):raise ValueError('时域必须是步长整数倍')
        c['n_steps']=int(round(n));c['dt']=c['step_minutes']/60
        ns=rows(book,'节点负荷');c['nodes']=[str(r['node']) for r in ns]
        if c['nodes']!=['T0','T1']+[f'D{i}' for i in range(1,34)]:raise ValueError('本版为T0/T1加D1—D33，共35节点')
        c['load']={k:np.array([number(r[k],k) for r in ns]) for k in ('critical_base_mw','general_base_mw','reactive_ratio')}
        c['protected_nodes']=[i for i,r in enumerate(ns) if boolean(r['protected'])]
        c['plot_xy']=np.array([[number(r['plot_x'],'plot_x'),number(r['plot_y'],'plot_y')] for r in ns])
        geo={r['node']:r for r in rows(book,'地理坐标')}
        c['geo_xy']=np.array([[number(geo[k]['x_km'],'x_km'),number(geo[k]['y_km'],'y_km')] for k in c['nodes']])
        c['edges']=[]
        for r in rows(book,'线路参数'):
            e={'name':str(r['name']),'from':c['nodes'].index(r['from_node']),'to':c['nodes'].index(r['to_node']),
               'kind':str(r['kind']),'switch_id':str(r['switch_id'] or ''),'initial_closed':boolean(r['initial_closed']),
               'switchable':boolean(r['switchable']),'exposed':boolean(r['exposed'])}
            for k in ('s_mva','r_pu','x_pu','hazard_scale','wind_threshold_ms','hazard_exponent','inspection_hours','repair_light_hours','repair_heavy_hours'):
                e[k]=number(r[k],k)
            e['plot_route']=[[float(x) for x in segment.split(',')] for segment in str(r.get('plot_route') or '').split(';') if segment]
            c['edges'].append(e)
        if len(c['edges'])!=39 or len({e['name'] for e in c['edges']})!=39:raise ValueError('需要39条名称唯一的线路')
        c['mg_nodes']=[];c['microgrids']=[]
        for r in rows(book,'微网参数'):
            mg={'name':str(r['name']),'bus':str(r['bus'])}
            for k,v in r.items():
                if k not in ('name','bus'):mg[k]=number(v,k)
            c['microgrids'].append(mg);c['mg_nodes'].append(c['nodes'].index(mg['bus']))
        if len(c['microgrids'])!=3 or len(set(c['mg_nodes']))!=3:raise ValueError('需要三个不同接入点的微网')
        c['track']=rows(book,'台风路径')
        for r in c['track']:
            for k,v in r.items():r[k]=number(v,k)
        if any(a['time_hour']>=b['time_hour'] for a,b in zip(c['track'],c['track'][1:])):raise ValueError('台风路径时刻须递增')
        if c['track'][0]['time_hour']>c['weather']['history_hours']*-1 or c['track'][-1]['time_hour']<c['horizon_hours']:raise ValueError('路径须覆盖灾前历史至24h终点')
        curve=rows(book,'24小时曲线')
        if len(curve)!=c['n_steps']:raise ValueError(f'曲线应有{c["n_steps"]}行')
        if not np.allclose([r['time_hour'] for r in curve],np.arange(c['n_steps'])*c['dt']):raise ValueError('曲线时间不匹配')
        c['profiles']={'pv':np.array([[number(r[f'pv_MG{m+1}'],'PV') for r in curve] for m in range(3)])*
                           np.array([m['pv_peak_mw'] for m in c['microgrids']])[:,None],
                       'critical':np.array([[number(r[f'critical_{k}'],'关键负荷') for r in curve] for k in c['nodes']])*c['load']['critical_base_mw'][:,None],
                       'general':np.array([[number(r[f'general_{k}'],'一般负荷') for r in curve] for k in c['nodes']])*c['load']['general_base_mw'][:,None],
                       'price':np.array([number(r['grid_price'],'价格') for r in curve])*c['grid_price']}
        auxrows=rows(book,'启动曲线')
        if not auxrows or not np.allclose([number(r['time_hour'],'启动时刻') for r in auxrows],np.arange(len(auxrows))*c['dt']):raise ValueError('启动轮廓必须按调度步长从0开始，禁止跳过峰值')
        c['generator']['startup_profile_mw']=np.array([number(r['aux_power_mw'],'启动轮廓') for r in auxrows])
        settings={r['parameter']:r['value'] for r in rows(book,'运行设置')}
        settings.setdefault('run_coordination_baseline','否')
        for k in ('compute_capability','compute_phase_map','run_comparisons','run_coordination_baseline'):settings[k]=boolean(settings[k])
        for k in ('forecast_scenarios','capability_directions'):
            value=number(settings[k],k)
            if not value.is_integer():raise ValueError(f'{k}须为整数')
            settings[k]=int(value)
        for k in ('phase_alphas','phase_socs'):
            settings[k]=[number(x,k) for x in str(settings[k]).replace('，',',').split(',')]
        settings['policies']=[x.strip() for x in str(settings['policies']).split(',')]
        if not set(settings['policies'])<= {'joint','critical_first','nearest'}:raise ValueError('策略为joint,critical_first,nearest')
        settings['output_dir']=str(settings['output_dir'])
        c['settings']=settings;validate_config(c)
        return c,settings
    finally:book.close()

def validate_config(c):
    if not 0<=c['alpha']<=1:raise ValueError('alpha须在[0,1]')
    if not 0<c['voltage_min']<=1<=c['voltage_max']:raise ValueError('电压界须包含1pu')
    if c['polygon_sides']<4 or not float(c['polygon_sides']).is_integer():raise ValueError('容量多边形边数须为至少4的整数')
    if c['generator']['max_attempts']<1 or not float(c['generator']['max_attempts']).is_integer():raise ValueError('启动尝试次数须为正整数')
    if c['settings']['capability_directions']<4:raise ValueError('能力投影方向至少4个')
    if c['critical_weight']<=0 or c['general_weight']<=0:raise ValueError('恢复权重须正')
    for k in ('switch_minutes','battery_mode_minutes'):
        if c[k]<=0 or not np.isclose(c[k]/c['step_minutes'],round(c[k]/c['step_minutes'])):raise ValueError(f'{k}须为步长整数倍')
    for e in c['edges']:
        if e['s_mva']<=0 or min(e['r_pu'],e['x_pu'],e['hazard_scale'])<0:raise ValueError('线路容量/阻抗/危险率不合法')
        if e['wind_threshold_ms']<=0 or e['hazard_exponent']<=0:raise ValueError('风速阈值和指数须正')
        for k in ('inspection_hours','repair_light_hours','repair_heavy_hours'):
            if e[k]<=0 or not np.isclose(e[k]/c['dt'],round(e[k]/c['dt'])):raise ValueError('巡检/维修工时须为步长正整数倍')
        if e['repair_heavy_hours']<e['repair_light_hours']:raise ValueError('重伤工时不能小于轻伤')
    for mg in c['microgrids']:
        if not 0<=mg['soc_min']<=mg['soc_initial']<=mg['soc_max']<=1:raise ValueError('SOC界不合法')
        if not mg['soc_min']<=mg['soc_terminal']<=mg['soc_max']:raise ValueError('末端SOC不合法')
        for k in ('energy_mwh','charge_mw','discharge_mw','inverter_mva'):
            if mg[k]<=0:raise ValueError('微网容量须正')
        if not 0<mg['eta_charge']<=1 or not 0<mg['eta_discharge']<=1:raise ValueError('效率须在(0,1]')
    for v in c['profiles'].values():
        if not np.isfinite(v).all() or (v<0).any():raise ValueError('曲线须为非负有限数')
    if c['crew']['speed_kmh']<=0 or c['crew']['decision_hours']<=0 or c['crew']['safe_wind_ms']<=0:raise ValueError('队伍速度/决策间隔/安全阈值须正')
    if not np.isclose(c['weather']['fault_interval_hours']/c['dt'],round(c['weather']['fault_interval_hours']/c['dt'])):raise ValueError('故障报告间隔须为调度步长整数倍')
    if c['settings']['forecast_scenarios']<1:raise ValueError('预测场景数须正')
    return c

def jsonable(v):
    if isinstance(v,dict):return {str(k):jsonable(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [jsonable(x) for x in v]
    if isinstance(v,np.ndarray):return jsonable(v.tolist())
    if isinstance(v,(np.floating,float)):return float(v) if np.isfinite(v) else None
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,np.bool_):return bool(v)
    return v

def save_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(jsonable(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def write_workbook(result,phase,path,world=None):
    """实际执行轨迹和预测D分开保存，避免把预测最优值当成实际收益。"""
    b=Workbook();w=b.active;w.title='核心指标';w.append(['指标','实际执行结果'])
    labels={'actual_weighted_recovery':'实际加权恢复面积比','actual_EENS_MWh':'实际失供电量/MWh','actual_operation_cost_thousand':'实际运行成本/千元',
        'manual_switch_operations':'主动开关操作次数','reported_fault_count':'已报告故障总数','unrepaired_end_count':'24h末未修复数','startup_attempts':'启动尝试次数',
        'reoptimizations':'滚动重算次数','emergency_rounds':'应急回退次数','terminal_reserve_satisfied':'终端SOC目标是否满足','scope':'结果解释范围'}
    for k,v in result['summary'].items():
        if isinstance(v,(str,int,float)) or v is None:
            w.append([labels.get(k,k),v])
            if k=='actual_weighted_recovery':w.cell(w.max_row,2).number_format='0.00%'
    w=b.create_sheet('24小时功率');w.append(['时间h','主配向上MW','机组MW','辅机MW','关键供电MW','一般供电MW','加权恢复率','在线状态'])
    for i,h in enumerate(result['time_hours']):w.append([h,result['upward_p'][i],result['target_p'][i],result['aux'][i],sum(result['served_c'][i]),sum(result['served_g'][i]),result['recovery'][i],result['unit_online'][i]])
    w=b.create_sheet('SOC');w.append(['时间h','MG1','MG2','MG3'])
    for i,e in enumerate(result['energy']):w.append([i*result['dt'],*[e[j]/result['energy_capacity'][j] for j in range(3)]])
    w=b.create_sheet('故障与巡检修复');w.append(['事件序号','时刻h','事件','线路','损伤','说明'])
    for i,r in enumerate(result['events']):w.append([i,r['hour'],r['kind'],r.get('line',''),r.get('damage',''),r.get('note','')])
    w=b.create_sheet('开关与节点');w.append(['时间h','线路','物理闭合','带电','有功MW','无功Mvar'])
    for i,h in enumerate(result['time_hours']):
        for e,name in enumerate(result['edge_names']):w.append([h,name,int(result['closed'][i][e]),int(result['energized_edges'][i][e]),result['flow_p'][i][e],result['flow_q'][i][e]])
    w=b.create_sheet('滚动D与证据');w.append(['时刻','所选行动','判定','D_R估计','D_R下界','D_R上界','是否证实','抢修候选数'])
    for r in result['rounds']:w.append([r['hour'],r['action'],r['decision'],r.get('D_estimate'),r.get('D_lower'),r.get('D_upper'),r.get('certified'),r['job_count']])
    if world is not None:
        w=b.create_sheet('风速与条件故障概率');w.append(['时间h','线路','沿线最大风速m/s','气象危险率1/h','尚未首损条件15min故障概率','实际首次故障时刻h'])
        for i,h in enumerate(result['time_hours']):
            for e,name in enumerate(result['edge_names']):
                fail=float(world['failure_hour'][e]);risk=1-np.exp(-float(world['hazard'][e,i])*result['dt'])
                w.append([h,name,float(world['wind'][e,i]),float(world['hazard'][e,i]),risk,fail if np.isfinite(fail) else None])
    if phase:
        w=b.create_sheet('D相图');w.append(['alpha','SOC','D_R估计','D_R下界','D_R上界','类别'])
        for r in phase['records']:w.append([r[k] for k in ('alpha','soc','D_estimate','D_lower','D_upper','classification')])
    for w in b:
        w.freeze_panes='B2';w.auto_filter.ref=w.dimensions
        for row in w:
            for cell in row:
                cell.font=Font(name='微软雅黑',size=10);cell.alignment=Alignment(vertical='center')
                if cell.row==1:cell.fill=PatternFill('solid',fgColor='18374D');cell.font=Font(name='微软雅黑',size=10,color='FFFFFF',bold=True)
        for col in w.columns:w.column_dimensions[col[0].column_letter].width=23
    b.save(path)
