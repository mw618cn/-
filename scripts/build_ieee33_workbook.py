"""开发用：从官方case33bw数据生成交付Excel；用户运行main.py无需本脚本。"""
from pathlib import Path
import re, itertools, sys
import numpy as np
from openpyxl import load_workbook
from openpyxl.styles import Font,PatternFill,Alignment
from openpyxl.comments import Comment
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ieee33_version'))
from network import forest_components
source=Path(sys.argv[1]).read_text()
def matrix(name):
    text=source.split('mpc.'+name+' = [',1)[1].split('];',1)[0]
    return [[float(v) for v in line.split(';')[0].split()] for line in text.splitlines() if line.strip() and not line.strip().startswith('%')]
bus=matrix('bus');branches=matrix('branch')
book=load_workbook(ROOT/'simple_version/参数数据.xlsx')
# 保留清晰一致的基础参数/微网/抢修/场景表，换掉节点和线路。
for row in book['基础参数'].iter_rows(min_row=4):
    values={'case_name':'modified_ieee33_24h','solver.time_limit':90,'solver.mip_rel_gap':.01,'voltage_min':.95,'voltage_max':1.05}
    if row[0].value in values:row[1].value=values[row[0].value]
for row in book['运行设置'].iter_rows(min_row=4):
    values={'policy_ids':'B_11','compute_capability':'否','compute_cost_surface':'否','run_rolling':'否','compute_baseline':'否','compare_start_wait':'是'}
    if row[0].value in values:row[1].value=values[row[0].value]
node_names=['T0','T1']+[f'D{i}' for i in range(1,34)]
coords=[[-2.8,0],[-1.4,0]]+[[i-1,0] for i in range(1,19)]
coords += [[x,2.8] for x in (1,2.3,3.6,4.9)]
coords += [[x,-2] for x in (2,3.4,4.8)]
coords += [[x,1.4] for x in (5,6.4,7.8,9.2,10.6,12,13.4,14.8)]
def sheet(name,headers,chinese,rows,title):
    pos=book.sheetnames.index(name) if name in book.sheetnames else len(book.sheetnames)
    if name in book.sheetnames:del book[name]
    ws=book.create_sheet(name,pos);ws.append([title]);ws.append(headers);ws.append(chinese)
    for row in rows:ws.append(row)
    return ws
critical=[0,.3]+[b[2]/1000*.3 for b in bus]
general=[0,.55]+[b[2]/1000*.7 for b in bus]
ratios=[0,np.tan(np.arccos(.97))]+[b[3]/b[2] if b[2]>0 else 0 for b in bus]
sheet('节点负荷',['node','critical_base_mw','general_base_mw','protected','reactive_ratio','role','plot_x','plot_y'],
      ['节点','关键负荷MW','一般负荷MW','关键负荷硬保护','原始Q/P比例','角色','示意横坐标','示意纵坐标'],
      [[name,critical[i],general[i],'是' if name in ('D18','D25','D33') else '否',ratios[i],
        'TSO机组与剩余源' if i==0 else 'TSO负荷与变电接口' if i==1 else 'DSO与MGO接入' if name in ('D18','D25','D33') else 'DSO',*coords[i]] for i,name in enumerate(node_names)],
      '33个配电节点＋T0/T1；D1—D33有功/无功基于case33bw，关键/一般拆分和TSO负荷为自设')
sheet('微网位置',['name','bus','description'],['微网','接入母线','模型含义'],
      [[f'MG{i}',f'D{b}','单母线聚合：PV+储能+DG+所在母线负荷，DSO管理所有相邻线路'] for i,b in enumerate((18,25,33),1)],
      '微网为单母线聚合，不把附近所有配电节点宣称为微网内部')
edges=[dict(name='TS',fr='T0',to='T1',s=5.,r=.012,x=.018,kind='transmission',sid='',fault='',closed=1),
       dict(name='TX',fr='T1',to='D1',s=4.,r=.003,x=.015,kind='transformer',sid='',fault='',closed=1)]
zbase=12.66**2/10
routes=['6.5,3.5;7,3.5','8,-1.1;14,-1.1','11,4.2;4.9,4.2','17,2.3;14.8,2.3','9.2,-2;9.2,1.0']
for i,b in enumerate(branches):
    fr,to=int(b[0]),int(b[1]);fault='A' if (fr,to)==(12,13) else 'B' if (fr,to)==(27,28) else ''
    edges.append(dict(name=f'L{fr:02d}_{to:02d}' if i<32 else f'TIE{i-31}',fr=f'D{fr}',to=f'D{to}',
        s=3. if i<32 else 2.,r=b[2]/zbase,x=b[3]/zbase,kind='feeder' if i<32 else 'tie',
        sid=f'S{i+1:02d}',fault=fault,closed=int(i<32 and not fault),route='' if i<32 else routes[i-32]))
# 生成清楚的有限菜单：4种故障支路闭合组合 × 基础/单联络/两联络。
# 每次闭合导致成环时，打开环中一条健康分段线；不打开固定TS/TX。
indexed=[{'name':e['name'],'from':node_names.index(e['fr']),'to':node_names.index(e['to'])} for e in edges]
def path_edges(status,start,end):
    seen={start};stack=[(start,[])]
    while stack:
        node,path=stack.pop()
        if node==end:return path
        for j,e in enumerate(indexed):
            if not status[j]:continue
            other=e['to'] if e['from']==node else e['from'] if e['to']==node else None
            if other is not None and other not in seen:seen.add(other);stack.append((other,path+[j]))
    return None
profiles=[];signatures=set()
for a,b in itertools.product((0,1),repeat=2):
    for ties in [(),(34,),(35,),(36,),(37,),(38,),(35,38),(34,37)]:
        status=[1]*34+[0]*5
        status[13]=a  # original branch12, plus TS/TX => index13
        status[28]=b  # original branch27 => index28
        for tie in ties:
            e=indexed[tie];path=path_edges(status,e['from'],e['to'])
            if path is not None:
                candidates=[j for j in path if 2<=j<34 and not edges[j]['fault']]
                if not candidates:break
                # 稳定地优先选择环中编号最小的健康分段开关。
                status[min(candidates)]=0
            status[tie]=1
        else:
            signature=tuple(status)
            if signature not in signatures:
                comps=forest_components(node_names,indexed,status)
                roots={node_names.index(f'D{v}') for v in (18,25,33)}
                if all(set(c)&roots for c in comps):
                    signatures.add(signature)
                    profiles.append([f'R{len(profiles):02d}',f'A={a},B={b}；联络'+('/'.join(edges[j]['sid'] for j in ties) or '无'),
                                     ','.join(e['name'] for e,z in zip(edges,status) if not z)])
# 仅变化过或故障/联络线路需要可操作开关，其余线路固定闭合。
opened={v for p in profiles for v in p[2].split(',')}
sheet('线路参数',['name','from_node','to_node','s_mva','r_pu','x_pu','initial_closed','switchable','switch_id','kind','fault_site','plot_route'],
 ['线路名','起点','终点','容量MVA(自设)','Rpu(10MVA)','Xpu(10MVA)','初始闭合','可操作开关','开关编号','类型','故障地点','绘图折点非地理坐标'],
 [[e['name'],e['fr'],e['to'],e['s'],e['r'],e['x'],e['closed'],int(e['name'] in opened),e['sid'] if e['name'] in opened else '',e['kind'],e['fault'],e.get('route','')] for e in edges],
 '37条标准配电支路（32常闭＋5常开联络）＋TS/TX；容量为自设，原数据未给热额定')
sheet('接线方案',['id','label','open_edges'],['方案ID','说明','断开线路名称（英文逗号）'],profiles,
      '有限径向森林候选菜单：每一分量含成网微网；真实优化在此菜单内选择，可修改并接受无环/成网检查')
old=list(book['24小时曲线'].iter_rows(min_row=4,values_only=True))
# 使用此前相同的24h形状，不重复改变负荷时变假设。
headers=['time_hour']+[f'pv_MG{i}' for i in (1,2,3)]+[f'critical_{v}' for v in node_names]+[f'general_{v}' for v in node_names]+['grid_price']
curve=[]
for r in old:
    h=r[0];c=r[6];g=r[12]
    curve.append([h,*r[1:4],*[c]*35,*[g]*35,r[-1]])
sheet('24小时曲线',headers,['灾后时间h']+['光伏系数']*3+['关键负荷系数']*35+['一般负荷系数']*35+['购电价格系数'],curve,
      '完整96时段：各节点负荷系数乘节点负荷基值；Q随该节点恢复的P按Q/P同比恢复')
ws=book['说明'];ws.delete_rows(1,ws.max_row)
for row in [
 ['33节点主配微协同韧性恢复参数表','使用说明'],
 ['运行方式','只运行同目录main.py；先用Excel/WPS打开本文件编辑，IDE的No loader不表示损坏'],
 ['网络范围','D1—D33为配电网，T0/T1为新增主网等值；MG1/2/3位于D18/D25/D33'],
 ['首跑设置','B_11、24h/96时段/4损伤场景；比较启动等待；能力域、代价面和滚动默认关闭，可在运行设置开启'],
 ['规模说明','有限接线菜单＋有限因果抢修策略；不是全部开关组合/任意车辆路径的全局优化'],
 ['开关编号','S01—S32按原32条馈线顺序；S33—S37为原五条常开联络，实际可操作位置见线路表switchable'],
 ['故障','A=12—13，B=27—28，初始均故障并隔离，位置已知、严重程度待巡检。修复可用性随时间变，不会无修复自愈'],
 ['成网假设','所有候选分量至少含一成网微网，初始送电瞬态已完成；不模拟励磁、频率或保护暂态'],
 ['数据来源','MATPOWER case33bw，Baran & Wu 1989，DOI10.1109/61.25627'],
 ['标准数据范围','保留33节点接线、R/X、原Pd/Qd；负荷拆分、微网、主网等值、容量、故障、时间、概率为自设'],
 ['阻抗基准','12.66kV / 10MVA：Zbase=16.02756Ω，Rpu=RΩ/Zbase，X同理；线性电压公式以MW除baseMVA'],
 ['有功与无功','节点负荷表reactive_ratio保留原Q/P；基础参数load.power_factor仅供兼容和TSO等值初始填表参考'],
 ['图件','00参数接线图在求解前生成；14实际接线快照和15开关热图仅用通过校核的真实结果'],
 ['ADMM','本版采用集中式HiGHS MILP，未实现ADMM；能力域为固定离散方案后的全时域条件投影'],
 ['D值','D0=等待最优费用−当前启动最优费用；范围仅限所选抢修策略与接线菜单，求解界跨零不得宣称判定'],
 ['表格格式','保留各表第2行英文字段、第3行中文解释；数据从第4行开始。先保存Excel再运行Python']]:ws.append(row)
for ws in book:
    ws.freeze_panes='B4' if ws.title!='说明' else 'B2'
    if ws.title!='说明':ws.auto_filter.ref=f'A3:{ws.cell(ws.max_row,ws.max_column).coordinate}'
    for row in ws:
        for cell in row:
            cell.font=Font(name='微软雅黑',size=10);cell.alignment=Alignment(vertical='center',wrap_text=False)
            if cell.row<=3:
                cell.fill=PatternFill('solid',fgColor='18374D' if cell.row!=3 else 'E7F0F4')
                cell.font=Font(name='微软雅黑',size=10,color='FFFFFF' if cell.row!=3 else '18374D',bold=True)
    from openpyxl.utils import get_column_letter
    for i in range(1,ws.max_column+1):ws.column_dimensions[get_column_letter(i)].width=18
    ws.column_dimensions['A'].width=30
    if ws.title in ('说明','接线方案'):ws.column_dimensions['B'].width=68;ws.column_dimensions['C'].width=100
    if ws.title=='基础参数' or ws.title=='运行设置':ws.column_dimensions['C'].width=48
book.save(ROOT/'ieee33_version/参数数据.xlsx')
print('Generated',len(node_names),'nodes',len(edges),'edges',len(profiles),'topologies; switchable',len(opened))
