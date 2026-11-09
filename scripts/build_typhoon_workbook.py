"""开发脚本：生成单一台风参数工作簿；交付用户只需main.py。"""
from pathlib import Path
import math
from openpyxl import load_workbook,Workbook
from openpyxl.styles import Font,PatternFill,Alignment
ROOT=Path(__file__).resolve().parents[1];old=load_workbook(ROOT/'ieee33_version/参数数据.xlsx',data_only=True)
def getrows(name):
 w=old[name];keys=[c.value for c in w[2]]
 return [{k:r[i] for i,k in enumerate(keys)} for r in w.iter_rows(min_row=4,values_only=True) if r[0] is not None]
b=Workbook();b.remove(b.active)
def sheet(name,headers,cn,data,title):
 w=b.create_sheet(name);w.append([title]);w.append(headers);w.append(cn)
 for r in data:w.append(r)
 return w
base={r['parameter']:r['value'] for r in getrows('基础参数')}
keep=['horizon_hours','step_minutes','base_mva','alpha','grid_reference_mw','grid_price','voltage_min','voltage_max','polygon_sides','battery_mode_minutes']
p={k:base[k] for k in keep};p.update({'case_name':'typhoon_dynamic_33','switch_minutes':60,'switch_min_hold_hours':1,'switch_max_per_line':4,'switch_max_total':24,
 'switch_cost':.02,'allow_upward':1,'critical_weight':5,'general_weight':1,'restore_tolerance':.0002,
 'crew.speed_kmh':30,'crew.road_detour_factor':1.35,'crew.safe_wind_ms':18,'crew.hourly_cost':.12,'crew.decision_hours':1,
 'crew.depot_x_km':8,'crew.depot_y_km':-8,'solver.primary_seconds':8,'solver.secondary_seconds':1,'solver.phase_seconds':8,'solver.rel_gap':.02,
 'weather.seed':5,'weather.history_hours':4,'weather.fault_interval_hours':1,'weather.outer_exponent':.8,'weather.surface_factor':.9,'weather.pv_wind_derating':.45,
 'replan_hours':2,'capability_interval_hours':1,'phase_restore_tolerance':.001})
for k in ['max_mw','min_mw','ramp_mw_per_hour','q_max_mvar','s_mva','aux_power_factor','last_start_hour','start_cost','fuel_cost']:
 p['generator.'+k]=base['generator.'+k]
p.update({'generator.max_attempts':2,'generator.restart_delay_hours':1,'generator.emergency_ramp_down_mw':2.4})
sheet('基础参数',['parameter','value','description'],['参数','值','说明'],[[k,v,'恢复率优先；时间h，功率MW，能量MWh，费用千元；风速m/s'] for k,v in p.items()],
 '保留原三创新：D决策边界／多时段可分解支撑域／信息—抢修—重构—启动协同；台风为场景输入')
ns=getrows('节点负荷');sheet('节点负荷',list(ns[0]),list(ns[0]),[[r[k] for k in ns[0]] for r in ns], '原33节点Pd/Qd，增加T0/T1；绘图坐标不作风场坐标')
geo={r['node']:(8+1.5*r['plot_x'],3*r['plot_y']) for r in ns}
sheet('地理坐标',['node','x_km','y_km','source'],['节点','地理X/km','地理Y/km','来源'],[[k,*v,'独立合成示例位置，非真实IEEE33地理数据'] for k,v in geo.items()],
 '合成地理位置供风场与道路近似；后续可替换真实坐标。不同于节点图排版坐标。')
edges=getrows('线路参数');fields=['name','from_node','to_node','s_mva','r_pu','x_pu','initial_closed','switchable','switch_id','kind','exposed','hazard_scale','wind_threshold_ms','hazard_exponent','inspection_hours','repair_light_hours','repair_heavy_hours','plot_route']
data=[]
for i,e in enumerate(edges):
 length=math.dist(geo[e['from_node']],geo[e['to_node']]);light=math.ceil((.75+.025*length)*4)/4;heavy=math.ceil((1.5+.05*length)*4)/4
 r={**e,'initial_closed':int(e['kind']!='tie'),'switchable':int(i>=2),'switch_id':f'S{i-1:02d}' if i>=2 else '',
    'exposed':int(i>=2),'hazard_scale':.09*(1+.03*length)*(1+.12*(i%3)),'wind_threshold_ms':20,'hazard_exponent':2,
    'inspection_hours':.25,'repair_light_hours':light,'repair_heavy_hours':heavy}
 data.append([r[k] for k in fields])
sheet('线路参数',fields,fields,data,'37条配电线全部参与风灾（含常开联络）；容量、危险率、工时均为可替换合成参数，非标定数据')
mgold={r['parameter']:r for r in getrows('微网参数')};mgfields=list(mgold)
sheet('微网参数',['name','bus',*mgfields],['微网','接入点',*mgfields],[[f'MG{i}',f'D{node}',*[mgold[k][f'MG{i}'] for k in mgfields]] for i,node in enumerate((18,25,33),1)],
 '微网为单母线聚合资源；DG/PV/储能，硬保护本地关键负荷，保留全时域SOC')
track=[[-4,-8,6,42,8],[0,6,6,44,8],[4,20,6,46,8],[8,38,6,40,8],[12,58,6,32,8],[24,120,6,22,8]]
sheet('台风路径',['time_hour','center_x_km','center_y_km','max_wind_ms','radius_max_km'],['时间h','中心X/km','中心Y/km','参考最大风速m/s','最大风速半径km'],track,'移动Rankine型简化风场；全部参数为合成示例，不声称Holland完整物理标定')
curve=getrows('24小时曲线');sheet('24小时曲线',list(curve[0]),list(curve[0]),[[r[k] for k in curve[0]] for r in curve], '96个15分钟PV/负荷/价格系数；台风进一步影响PV预测与实际出力')
aux=getrows('启动曲线');sheet('启动曲线',['time_hour','aux_power_mw'],['启动历时h','辅助功率MW'],[[r['time_hour'],r['aux_power_mw']] for r in aux], '完整启动辅助功率持续曲线；模型和实际执行共用')
settings={'forecast_scenarios':2,'compute_capability':'是','capability_directions':8,'compute_phase_map':'是','phase_alphas':'0,0.5,1',
          'phase_socs':'0.35,0.65,0.85','run_comparisons':'是','run_coordination_baseline':'是','policies':'joint,critical_first','output_dir':'results'}
sheet('运行设置',['parameter','value','description'],['项目','值','说明'],[[k,v,'joint联合优化；critical_first重要负荷规则；nearest就近规则；可关闭附加分析加快首跑'] for k,v in settings.items()],
 '首次运行完整24h。相图与域耗时较多，可先改为否；所有比较共用同一真实风灾样本')
w=b.create_sheet('说明',0)
for row in [['台风动态故障33节点仿真','只运行main.py，参数只修改这一份Excel'],
 ['三个创新点','①启动/等待决策边界；②多时段可分解支撑能力；③信息驱动的抢修—重构—启动。台风仅为场景输入。'],
 ['初始条件','台风灾前4h历史产生t=0故障；微网已成网。新故障按小时发生，电力调度15分钟、SOC97点。'],
 ['主目标','最大化24h加权恢复面积；在已找到恢复水平容差内降低运行/维修/主动开关成本。'],
 ['故障模型','每线路每次台风最多首次损坏一次；常开联络线同样暴露；已故障线路必须维修才恢复。'],
 ['信息约束','未来真实故障和损伤不传入优化器。未来故障独立预测采样；相同故障报告历史下动作绑定。'],
 ['抢修原型','一队伍；只分派当前已报告故障。未知工时前瞻用重伤上界；先执行巡检，获知轻重后再优化。'],
 ['天气作业','作业遇危险风速暂停，保留进度；道路距离用独立合成坐标×绕行系数，不用电气阻抗当路程。'],
 ['ADMM','本版仍是集中式MILP＋条件LP证书，未实现ADMM。'],
 ['结果边界','预测场景中的R界不是实际全天执行效果的全局最优界；实际恢复曲线单独统计。'],
 ['求解限时','默认短限时用于先跑通；未判区如实显示。论文实验应提高限时并检验R界收紧。'],
 ['Excel打开','用Excel/WPS打开；PyCharm No loader仅表示缺少预览插件。'],
 ['基础数据','标准接线/阻抗/Pd/Qd来自MATPOWER case33bw；其余为合成研究参数，可替换。']]:w.append(row)
for w in b:
 w.freeze_panes='B4' if w.title!='说明' else 'B2'
 for row in w:
  for cell in row:
   cell.font=Font(name='微软雅黑',size=10);cell.alignment=Alignment(vertical='center')
   if cell.row<=3:cell.fill=PatternFill('solid',fgColor='18374D');cell.font=Font(name='微软雅黑',color='FFFFFF',bold=True,size=10)
 for i in range(1,w.max_column+1):
  from openpyxl.utils import get_column_letter
  w.column_dimensions[get_column_letter(i)].width=22
 w.column_dimensions['A'].width=33
 if w.title=='说明':w.column_dimensions['B'].width=100
b.save(ROOT/'typhoon_version/参数数据.xlsx');print('台风工作簿已生成')
