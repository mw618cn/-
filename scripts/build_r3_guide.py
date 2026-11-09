"""构建R3可编辑Word与对应LaTeX；只读最终说明/源码，不重新优化。"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json
from copy import deepcopy
import build_code_guide as guide
import build_paper as math

ROOT=Path(__file__).resolve().parents[1]/'resilience_v3'


def equation_specs():
    s,u,f,S=math.sub,math.sup,math.frac,math.summ
    return [
      ('危险率与区间概率',r'\lambda_{e,k}=a_e[\max(v_{e,k}/v_e^0-1,0)]^{b_e},\quad p_{e,k}=1-e^{-\lambda_{e,k}\Delta t}',
       [s('λ','e,k'),' = ',s('a','e'),u(['[max(',f(s('v','e,k'),math.ss('v','e','0')),' − 1,0)]'],s('b','e')),',  ',s('p','e,k'),' = 1 − ',u('e',['−',s('λ','e,k'),'Δt'])]),
      ('累计首次故障概率',r'F_e(t)=1-\exp[-\sum_{k:t_k<t}\lambda_{e,k}\Delta t]',
       [s('F','e'),'(t) = 1 − exp[−',S('k: tₖ<t',[s('λ','e,k'),'Δt']),']']),
      ('预测加权恢复面积',r'R={A_{\rm past}+\sum_s\pi_s\sum_{k=k_0}^{N-1}\Delta t\sum_i(w_c\ell^c_{i,k,s}+w_g\ell^g_{i,k,s})\over A_{\rm demand}}',
       ['R = ',f([s('A','past'),'+',S('s',[s('π','s'),S('k=k₀',['Δt',S('i',[s('w','c'),math.ss('ℓ','i,k,s','c'),'+',s('w','g'),math.ss('ℓ','i,k,s','g')])],'N−1')])],s('A','demand'))]),
      ('实际失供电量',r'\mathrm{EENS}=\sum_{k=0}^{N-1}\Delta t\sum_i(d^c_{i,k}+d^g_{i,k}-\ell^c_{i,k}-\ell^g_{i,k})',
       ['EENS = ',S('k=0',['Δt',S('i',[math.ss('d','i,k','c'),'+',math.ss('d','i,k','g'),'−',math.ss('ℓ','i,k','c'),'−',math.ss('ℓ','i,k','g')])],'N−1')]),
      ('SOC跨期动态',r'E_{j,k+1}=E_{j,k}+\Delta t(\eta^c_jP^{\rm ch}_{j,k}-P^{\rm dis}_{j,k}/\eta^d_j)',
       [s('E','j,k+1'),' = ',s('E','j,k'),'+Δt(',math.ss('η','j','c'),math.ss('P','j,k','ch'),' − ',f(math.ss('P','j,k','dis'),math.ss('η','j','d')),')']),
      ('MGO接口净功率',r'p_{j,k}=P^{\rm PV}_{j,k}+P^{\rm DG}_{j,k}+P^{\rm dis}_{j,k}-P^{\rm ch}_{j,k}-\ell^c_{j,k}-\ell^g_{j,k}',
       [s('p','j,k'),' = ',math.ss('P','j,k','PV'),'+',math.ss('P','j,k','DG'),'+',math.ss('P','j,k','dis'),'−',math.ss('P','j,k','ch'),'−',math.ss('ℓ','j,k','c'),'−',math.ss('ℓ','j,k','g')]),
      ('条件非预期性',r'\mathcal I_{s,k}=\mathcal I_{r,k}\ \Longrightarrow\ u_{s,k}=u_{r,k}',
       [s('I','s,k'),' = ',s('I','r,k'),' ⇒ ',s('u','s,k'),' = ',s('u','r,k')]),
      ('实体任务连续性',r'q_k\ne\varnothing,\ r_k>0\ \Longrightarrow\ q_{k+1}=q_k\quad\text{until completion}',
       [s('q','k'),' ≠ ∅, ',s('r','k'),' > 0 ⇒ ',s('q','k+1'),' = ',s('q','k'),'  (until completion)']),
      ('有限规则类的最优值界',r'L_b=\max_{p\in\mathcal P}L_{b,p},\quad U_b=\max_{p\in\mathcal P}U_{b,p}',
       [s('L','b'),' = ',s('max','p∈P'),s('L','b,p'),',  ',s('U','b'),' = ',s('max','p∈P'),s('U','b,p')]),
      ('启动等待差值与区间',r'D_R=R^*_{\rm start}-R^*_{\rm wait}\in[L_{\rm start}-U_{\rm wait},\ U_{\rm start}-L_{\rm wait}]',
       [s('D','R'),' = ',math.ss('R','start','*'),'−',math.ss('R','wait','*'),' ∈ [',s('L','start'),'−',s('U','wait'),', ',s('U','start'),'−',s('L','wait'),']']),
      ('同模型追加求解界合并',r'L^{m+1}=\max(L^m,\widehat L^{m+1}),\quad U^{m+1}=\min(U^m,\widehat U^{m+1})',
       [u('L','m+1'),' = max(',u('L','m'),', ',u('L̂','m+1'),'),  ',u('U','m+1'),' = min(',u('U','m'),', ',u('Û','m+1'),')']),
      ('后向管终端集合',r'\mathcal B_N=[E_{\rm terminal},E_{\max}]',
       [s('B','N'),' = [',s('E','terminal'),', ',s('E','max'),']']),
      ('区间前驱递推',r'\mathcal B_k=[E_{\min},E_{\max}]\cap[\underline B_{k+1}-\delta_{\max},\overline B_{k+1}-\delta_{\min}]',
       [s('B','k'),' = [',s('E','min'),', ',s('E','max'),'] ∩ [',s('B̲','k+1'),'−',s('δ','max'),', ',s('B̅','k+1'),'−',s('δ','min'),']']),
      ('观测前先交可能子分支',r'\mathcal J_{g,k+1}=\bigcap_{h\in\operatorname{child}(g)}\mathcal B_{h,k+1},\quad\mathcal B_{g,k}=\mathcal X\cap\operatorname{Pre}(\mathcal J_{g,k+1})',
       [s('J','g,k+1'),' = ',s('⋂','h∈child(g)'),s('B','h,k+1'),',  ',s('B','g,k'),' = X ∩ Pre(',s('J','g,k+1'),')']),
      ('共同系数的全轨迹内近似',r'v=\sum_r\lambda_rv^r,\quad x=\sum_r\lambda_rx^r,\quad\lambda_r\ge0,\ \sum_r\lambda_r=1',
       ['v = ',S('r',[s('λ','r'),u('v','r')]),', x = ',S('r',[s('λ','r'),u('x','r')]),', ',s('λ','r'),' ≥ 0, ',S('r',s('λ','r')),' = 1']),
      ('主配反送路径贡献的独立对照',r'\Delta R_{\rm path}=R_{\rm allow\ upward}^{\rm actual}-R_{\rm no\ upward}^{\rm actual}',
       [s('ΔR','path'),' = ',math.ss('R','allow upward','actual'),'−',math.ss('R','no upward','actual')]),
    ]


def build():
    guide.FONT='宋体';doc=guide.setup_document()
    doc.core_properties.title='R3主配微韧性恢复：模型与代码说明'
    doc.add_heading('主配微协同韧性恢复 R3',0)
    doc.add_paragraph('参数—信息—能力与后向递推—联合求解—首步执行—韧性与图形')
    doc.add_paragraph('正文、表格与公式均可编辑。公式采用Word原生OMML，并附同编号LaTeX源；不是原生MathType OLE对象。未获完整宋体文件，云端不提供替代字体预览。')
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M（北京时间）')
    doc.add_paragraph('构建时间：'+stamp)
    chapters=['README.md','模型与求解逻辑.md','模块与参数逐段说明.md','三个创新点与代码对应.md','图形与文献依据.md','验证说明.md']
    for chapter in chapters:
        doc.add_page_break();doc.add_heading(chapter.removesuffix('.md'),1);guide.markdown(doc,ROOT/chapter)
    doc.add_page_break();doc.add_heading('附录：可编辑结构公式',1)
    doc.add_paragraph('下列公式与LaTeX源同编号。式中的规则类、观测支持与单微网凸线性假设见模型正文；不可将条件界扩大解释为任意POMDP或样本外全局界。')
    tex=[r'\documentclass{article}',r'\usepackage{amsmath,amssymb}',r'\begin{document}'];catalog=[]
    for number,(title,latex,body) in enumerate(equation_specs(),1):
        doc.add_heading(f'({number}) {title}',3)
        p=doc.add_paragraph();om=math.node('m:oMath')
        for node in math.items(*body):om.append(deepcopy(node))
        p._p.append(om)
        catalog.append({'number':number,'title':title,'latex':latex})
        tex += [f'% ({number}) {title}',r'\begin{equation}',latex,rf'\tag{{{number}}}',r'\end{equation}']
    tex += [r'\end{document}']
    (ROOT/'模型公式.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    (ROOT/'公式索引.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
    doc.save(ROOT/'模型与代码说明_可编辑公式.docx')
    print('Word paragraphs',len(doc.paragraphs),'tables',len(doc.tables),'OMML',len(catalog))


if __name__=='__main__':build()
