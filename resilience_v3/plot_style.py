"""IEEE列宽、矢量输出和原版宋体硬校验；不静默替换字体。"""
from pathlib import Path
import os,tempfile,json
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'resilience_v3_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,ft2font
from matplotlib.text import Text

CONFIG={};FONT=None
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9']


def setup(c):
    global CONFIG,FONT
    CONFIG=c.get('options',{}).copy();paths=[]
    for value in [os.environ.get('TDM_SIMSUN_PATH'),CONFIG.get('plots.font_file')]:
        if value:paths.append(Path(value).expanduser())
    paths += [Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/simsun.ttc',Path('fonts/simsun.ttc')]
    try:paths.append(Path(font_manager.findfont('SimSun',fallback_to_default=False)))
    except ValueError:pass
    path=next((p.resolve() for p in paths if p.is_file()),None)
    if path is None:raise FileNotFoundError('未发现原版宋体。请安装系统宋体，或在算法与绘图设置填写plots.font_file。数值结果可先保存，在Windows用 --redraw 重画。')
    face=ft2font.FT2Font(str(path));family=face.family_name
    if 'simsun' not in family.lower() and family not in ('宋体','新宋体'):raise ValueError(f'指定文件的字体为{family}，不是宋体')
    FONT={'path':str(path),'family':family};font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({'font.family':[family],'font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,
        'axes.titleweight':'normal','axes.linewidth':.65,'lines.linewidth':1.1,'lines.markersize':3,
        'xtick.labelsize':7,'ytick.labelsize':7,'xtick.direction':'in','ytick.direction':'in',
        'legend.fontsize':7,'legend.frameon':False,'axes.spines.top':False,'axes.spines.right':False,
        'axes.unicode_minus':False,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
        'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'mathtext.default':'regular'})
    return FONT


def figure(height=3.,**kw):
    width=float(CONFIG.get('plots.width_mm',180))/25.4
    return plt.figure(figsize=(width,height),layout='constrained',**kw)


def save(fig,out,stem):
    if FONT is None:raise RuntimeError('绘图必须先完成宋体校验')
    # 生成刻度后逐对象绑定同一原版字体文件；不只依赖系统家族名匹配。
    fig.canvas.draw()
    face=ft2font.FT2Font(FONT['path']);missing=set()
    for obj in fig.findobj(Text):
        props=obj.get_fontproperties().copy();props.set_file(FONT['path']);obj.set_fontproperties(props)
        for char in obj.get_text():
            if char.isprintable() and not char.isspace() and face.get_char_index(ord(char))==0:missing.add(char)
    if missing:raise RuntimeError('宋体文件缺少图中文字形：'+''.join(sorted(missing))+'；请使用完整系统simsun.ttc，禁止自动代换。')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);files=[]
    for ext in ('png','pdf','svg'):
        p=out/f'{stem}.{ext}';fig.savefig(p,dpi=int(CONFIG.get('plots.dpi',600)));files.append(str(p))
    plt.close(fig)
    (out/'字体与图形设置.json').write_text(json.dumps({'font':FONT,'width_mm':CONFIG.get('plots.width_mm',180),'dpi':CONFIG.get('plots.dpi',600),'vector_formats':['pdf','svg']},ensure_ascii=False,indent=2),encoding='utf-8')
    return files


def panel(ax,label,title):
    ax.set_title(f'({label}) {title}',loc='left',pad=4)


def timeaxis(ax,horizon):
    ax.set_xlim(0,horizon);ax.set_xticks([0,horizon/4,horizon/2,3*horizon/4,horizon]);ax.set_xlabel('时间 / h')
    ax.grid(axis='y',color='.9',linewidth=.4,zorder=0)


def stairs(ax,t,y,dt,**kw):
    import numpy as np
    return ax.step(np.r_[t,t[-1]+dt],np.r_[y,y[-1]],where='post',**kw)
