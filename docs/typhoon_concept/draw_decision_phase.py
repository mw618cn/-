"""D_R相图表达示意：使用公开说明的解析函数，不调用电网优化或故障仿真。

运行本文件输出PNG/SVG/PDF。数值和边界不能作为33节点算例结论。
D_R=启动分支最大加权恢复率−等待分支最大加权恢复率；费用只用于主目标近似平手后的比较。
"""
from pathlib import Path
import os,tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'tdm_matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(tempfile.gettempdir())/'tdm_font_cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
from matplotlib.lines import Line2D

font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
else:plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
plt.rcParams.update({'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42,
    'font.size':10,'text.color':'#17334a','axes.labelcolor':'#17334a','axes.titleweight':'bold',
    'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
alpha=np.linspace(0,1,101);soc=np.linspace(.2,.9,101);x,y=np.meshgrid(alpha,soc)
# 仅供画图的任意解析函数；不是优化结果、代理预测或拟合到算例的数据。
z=100*(.10*(y-.55)-.035*(x-.4)+.006*np.sin(2*np.pi*x)*np.cos(np.pi*(y-.2)/.7))
cmap=LinearSegmentedColormap.from_list('decision',['#28569c','#e9f1f7','#fbfcfa','#d6ecdf','#19795b'])
norm=TwoSlopeNorm(vmin=-6,vcenter=0,vmax=6)
fig=plt.figure(figsize=(15.8,7.7));gs=fig.add_gridspec(1,2,left=.045,right=.94,bottom=.22,top=.80,wspace=.25)
ax=fig.add_subplot(gs[0,0],projection='3d');top=fig.add_subplot(gs[0,1])
ax.plot_surface(x,y,z,cmap=cmap,norm=norm,linewidth=0,alpha=.96,antialiased=True,rstride=3,cstride=3)
# 同一零等值线分别画在曲面及俯视图中。
contours=top.contour(x,y,z,levels=[0],colors=['#17334a'],linewidths=2.1)
for segment in contours.allsegs[0]:ax.plot(segment[:,0],segment[:,1],np.zeros(len(segment)),color='#17334a',lw=2.2,zorder=7)
ax.set(xlabel='主网剩余支撑比例 α',ylabel='统一初始SOC',zlabel='D_R / 百分点（示意）',xlim=(0,1),ylim=(.2,.9),zlim=(-6,6))
ax.view_init(elev=26,azim=-128);ax.set_title('(a) 启动—等待的恢复收益差曲面',loc='left',pad=15,fontsize=13)
mesh=top.pcolormesh(x,y,z,cmap=cmap,norm=norm,shading='auto',rasterized=True)
# 重画在网格之上，避免零线被色块盖住。
top.contour(x,y,z,levels=[0],colors=['#17334a'],linewidths=2.1)
top.text(.12,.80,'当前启动更优\nD_R > 0',ha='center',va='center',fontsize=14,fontweight='bold',color='#12513e',
         bbox=dict(boxstyle='round,pad=.45',facecolor='#ffffffdd',edgecolor='none'))
top.text(.82,.30,'等待更优\nD_R < 0',ha='center',va='center',fontsize=14,fontweight='bold',color='#244878',
         bbox=dict(boxstyle='round,pad=.45',facecolor='#ffffffdd',edgecolor='none'))
zero_points=np.vstack(contours.allsegs[0]);target=zero_points[np.argmin(np.abs(zero_points[:,0]-.57))]
top.annotate('示意零边界 D_R = 0',xy=target,xytext=(.68,.79),ha='center',fontsize=11,
             arrowprops=dict(arrowstyle='->',color='#17334a',lw=1.3),color='#17334a')
top.set(xlabel='主网剩余支撑比例 α（0：无外部注入；1：参考能力）',ylabel='统一初始SOC',xlim=(0,1),ylim=(.2,.9),
        xticks=[0,.25,.5,.75,1],yticks=[.2,.35,.5,.65,.8,.9])
top.set_title('(b) 同一曲面的俯视决策分区',loc='left',pad=15,fontsize=13)
cb=fig.colorbar(mesh,ax=top,pad=.025,fraction=.048);cb.set_label('D_R / 百分点（示意）')
fig.text(.055,.925,'恢复率优先：D_R曲面与决策相图应如何表达',fontsize=22,fontweight='bold')
fig.text(.055,.867,'正式计算口径：固定台风情景集与当前信息，分别优化“当前启动”和“当前等待”的完整恢复过程。',fontsize=11,color='#526d82')
fig.text(.055,.125,'D_R = 最大加权恢复率（当前启动）− 最大加权恢复率（当前等待）',fontsize=13,fontweight='bold')
fig.text(.055,.077,'示意图｜使用任意解析函数，非33节点计算结果；颜色、数值和零边界均不构成研究结论。',fontsize=10.5,color='#a15e34')
fig.text(.055,.033,'正式相图还应区分“求解界跨零、尚待判定”与“分支已证不可行”；边界可能弯曲或分离，不预设唯一临界点。',fontsize=10.5,color='#526d82')
folder=Path(__file__).resolve().parent
for suffix in ('png','svg','pdf'):fig.savefig(folder/f'decision_phase_concept.{suffix}',dpi=180,bbox_inches='tight')
plt.close(fig)
