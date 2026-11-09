# 主—配—微协同韧性恢复：24小时可运行研究原型

> 下载源码：点击仓库上方 **Code → Download ZIP**。也可打开 `downloads/tdm_source_code.zip`，点击文件页面的 **Download raw file**。
>
> 本仓库包含完整Python源码、参数、测试及文档生成脚本。算例图和Word文档可在运行后生成；首次使用请按下面顺序操作。

## 最快运行方法

1. 下载并解压源码ZIP，进入含有 `requirements.txt` 的文件夹。
2. 在该文件夹打开终端，依次执行：

```bash
python -m pip install -r requirements.txt
python scripts/run_demo.py --config configs/demo_24h.json --policies A_11 --branches none --output results/my_first_run --time-limit 60
```

完整8策略、启动/等待对比、能力域、禁反送对照及滚动回放：

```bash
python scripts/run_demo.py --config configs/demo_24h.json --policies all --capability --baselines --rolling --output results/demo --time-limit 25
```

计算完成后生成可编辑Word（公式为OMML，LaTeX源在 `equations/`）：

```bash
python scripts/build_paper.py --results results/demo --output results/模型与算例分析_可编辑.docx
python scripts/build_code_guide.py --output results/Python代码说明书_可编辑.docx
```

本代码把“是否现在启动”“微网群能否连续支撑”“巡检—抢修—拓扑如何改变决策”放进同一套24小时恢复模型。输出从参数、场景和优化模型一直延伸到物理校核、结果表和论文图。所有示例参数均为**自设参数**，集中存放，方便替换；结果只能解释这套示例系统。

## 先理解本版本计算什么

默认连续调度分辨率为15分钟，即24小时、96个时段；储能能量有97个状态点。网络包含主网层2个节点、配网层1个节点和3个微网节点。两处待巡检线路A、B各有轻损/重损，构成4个情景。一个巡检抢修队执行旅行、巡检和不可中断的维修任务；只有巡检结束后，调度器才知道该处损伤类型。

为使代码可以直接运行并便于解释，本版本先枚举8个**因果抢修策略**，再对每个策略求解四情景混合整数线性规划。最优性的范围是这8个候选策略及给定时间网格；它不是任意抢修车辆路径问题的全局最优解。充放电模式、机组启动和开关动作使用各自明确的离散网格，连续功率与SOC仍覆盖完整96时段。

网络采用无损LinDistFlow类线性近似、有功/无功平衡、平方电压约束和视在功率内接多边形。机组启动阶段的连续辅助功率必须由网络实际支撑，启动后的出力受容量和爬坡约束。本版本不把某一时刻“主网自动恢复”当作既定条件；主网初始可用电源与被启动机组分别建模。微网自启动及初始带电通道已具备是初始条件，合闸涌流、频率、电磁暂态与保护配合需在后续工程验证中补充。

## 三个研究问题怎样连接

1. **现在启动还是继续等待？** 在同样的24小时资源、信息规则和终端储能要求下，分别约束“此刻启动”和“此刻不启动、以后仍可启动”，比较最优续期代价。定义 `D = J_wait − J_start`。正值支持现在启动，负值支持先等待；求解未达到足够精度时，使用上下界区间判断，不把两个可行解的代价差直接当作已证实的最优判据。
2. **聚合支撑是否真的可分解？** 固定已求出的离散决策，保留全时域SOC、网络和情景信息约束，计算接口功率投影切片，并恢复各微网的可行调度见证。每个二维截面仍来自完整24小时模型；不能把不同时刻截面的任意点拼成一条可行轨迹。
3. **巡检和抢修怎样改变启动价值？** 巡检决定什么时候能分辨情景，抢修决定什么时候线路可用，拓扑决定资源能否送达。巡检之前的相同信息历史必须使用相同决策；巡检后才允许按已观察到的损伤分支。这既影响能力域，也影响启动与等待的比较。

闭环关系是：`可分解能力 → 启动/等待价值 → 巡检抢修与重构 → 新拓扑、新信息 → 能力更新`。主实验实现有限策略集下的完整时域优化；可选滚动回放按实际巡检观测更新后验，固定已执行控制、原队伍策略与动作网格后重求解。它保持同一个24小时终点，不重置SOC，也不允许在已开工后任意改写路线。

## 主网、配网、微网分别负责什么

| 角色 | 主要决策 | 向其他层提供的内容 |
|---|---|---|
| TSO 主网层 | 非黑启动机组的启动时刻、在线出力、主网负荷恢复、初始外部电源调用 | 启动阶段功率需求、主配接口约束与恢复价值 |
| DSO 配网层 | 巡检抢修策略、A/B恢复开关、配网负荷恢复、网络有功/无功与电压 | 可送达的上送功率、通道可用时间、损伤观测 |
| MGO 微网层 | 本地发电、光伏利用、储能充放电、关键负荷保护和接口支撑 | 24小时可行功率轨迹及其内部可分解调度 |

三级协同体现为同一个节点功率平衡、接口流量、跨时段储能和因果决策约束，不是仅在图上画三层。当前主求解器是SciPy调用的HiGHS MILP，方便无需商业许可证直接复现。ADMM不是本版本求解含启动、开关、抢修选择的非凸模型的默认方法；若后续研究分布式求解，应先固定离散决策，再对凸连续子问题进行接口一致性迭代，并与本版本集中求解结果对照。

## 文件和阅读顺序

| 阅读次序 | 文件 | 负责的事情 |
|---:|---|---|
| 1 | `configs/` | 示例系统、设备、损伤、队伍、费用和求解参数 |
| 2 | `src/tdm_restore/config.py` | 读取/校验参数，生成96时段负荷、光伏和费用序列 |
| 3 | `src/tdm_restore/crew.py` | 构造4情景、8因果策略、巡检与维修时间线及信息历史 |
| 4 | `src/tdm_restore/linear.py` | 稀疏线性模型、MILP/LP求解、离散决策固定 |
| 5 | `src/tdm_restore/model.py` | 建立全部变量、目标函数与物理/信息约束 |
| 6 | `src/tdm_restore/solution.py` | 将求解向量还原为有物理单位的时序和评价指标 |
| 7 | `src/tdm_restore/experiments.py` | 策略枚举、条件分支比较与结果组织 |
| 8 | `src/tdm_restore/validation.py` | 数值残差、设备边界、因果性等可执行校核 |
| 9 | `src/tdm_restore/capability.py` | 全时域约束下的条件能力域切片、轨迹与可行分解 |
| 10 | `src/tdm_restore/plotting.py` | 从计算结果生成24小时论文图 |
| 11 | `src/tdm_restore/rolling.py` | 按实际观测更新后验，固定已执行前缀进行滚动回放 |
| 12 | `docs/code_walkthrough.md` | 逐模块、逐约束块解释输入、方法和输出 |
| 13 | `docs/data_replacement.md` | 换成真实算例时的参数映射和验证步骤 |
| 14 | `scripts/build_paper.py`、`equations/` | 生成论文式Word与对应LaTeX公式源 |

`scripts/import_profiles.py`与 `configs/profile_template_96.csv`提供真实PV/负荷曲线的导入入口；按模板改96行即可生成模型可直接使用的JSON，步骤见数据替换指南。

`DESIGN.md`记录实现边界和数据接口，供扩展代码时查阅。数学模型、算例解释见配套Word；公式为Word原生可编辑OMML，并附LaTeX源。OMML不等于原生MathType OLE对象，可按需要在安装MathType的Word中转换。

## 安装

建议使用Python 3.11或3.12以及独立虚拟环境。SciPy自带HiGHS接口，无需购买Gurobi/CPLEX许可证。

Linux/macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

若PowerShell未允许激活脚本，可直接调用 `.\.venv\Scripts\python.exe` 执行后续命令。绘制中文图需要系统中文字体；代码会优先搜索常见中文字体，Windows通常可用微软雅黑，Linux示例环境使用Noto Sans CJK。

## 运行

先切换到解压后的代码包根目录。下面使用源码启动脚本，无需修改 `PYTHONPATH`。

查看参数说明，或只导出配置（均不求解MILP）：

```bash
python scripts/run_demo.py --help
python scripts/run_demo.py --config configs/demo_24h.json --write-config results/my_config.json
```

先运行单策略自由动作，检查完整24小时链条：

```bash
python scripts/run_demo.py --config configs/demo_24h.json --policies A_11 --branches none --time-limit 60 --output results/single_policy
```

再运行完整8策略、启动/等待比较、能力域和禁止上送基线：

```bash
python scripts/run_demo.py --config configs/demo_24h.json --policies all --branches start,wait --time-limit 25 --capability --baselines --output results/my_study
```

若同时回放实际HH情景的观测更新，在上面命令末尾加 `--rolling --realized-scenario HH`。`--rolling`会在主实验之后增加若干次续期求解；它不是只把原方案画成动画。

`--time-limit`是**每个MILP**的秒数，不是整次实验时限。完整菜单、两个动作分支和基线会运行多次MILP，能力域又需要若干LP。若短时限没有找到可行候选，应增加时限并查看状态；时限内有候选但界未收紧时，输出会保留上下界，而不会自动保证D符号可判定。

也可将项目安装成可编辑包后使用模块或命令行入口：

```bash
python -m pip install -e .
python -m tdm_restore --help
tdm-restore --config configs/demo_24h.json --output results/installed_run
```

策略ID为 `A_11,A_10,A_01,A_00,B_11,B_10,B_01,B_00`。首字母表示先巡检地点；两个二进制数字分别表示该点轻伤、重伤观测后是否先修它。`--policies A_11,B_10`只比较指定子菜单，结论范围也随之缩小。

可选的D参数扫描与已有算例汇总：

```bash
python scripts/run_sensitivity.py --alphas 0,0.5,1 --socs 0.7 --dry-run --output results/scan_plan
python scripts/run_sensitivity.py --summarize results/demo results/external_support --output results/decision_summary
```

去掉第一条中的 `--dry-run`才实际重求解各参数点；默认三微网使用共同SOC0.7，可能与主示例的SOC向量不同。扫描应保持其他条件和策略菜单一致，未知界或缺失结果不会自动当作D=0。完整扫描需要多次实验，不包含在一次单策略运行中。

重新生成两份Word（只读源码/已有结果，不求解）：

```bash
python scripts/build_paper.py --results results/demo --output results/论文式模型与算例分析_可编辑.docx
python scripts/build_code_guide.py --output results/Python代码说明书_可编辑.docx
```

## 输出怎样阅读

| 文件 | 应当查看的内容 |
|---|---|
| `config_resolved.json` | 本次运行最终参数，替换参数时以它核对 |
| `policy_screening.csv` | 每个候选策略的可行代价、下界、间隙与状态 |
| `solution.json` | 被选方案的完整情景时序、费用分项、指标和求解状态 |
| `branch_comparison.json` | 当前启动/等待的分支可行性、D值和可信区间 |
| `validation.json` | 约束、变量边界、整数性、能量与信息因果性的校核 |
| 时序CSV | 便于Excel重画的情景功率、储能、恢复与机组进程 |
| `figures/` | 同一组计算结果导出的PNG、SVG和PDF |
| `capability/` | 固定离散决策后的投影点、完整轨迹与分解校验 |
| `rolling/` | 实际观测、后验、已执行前缀和续期重求解记录（启用时） |

交付目录内 `results/demo/`保存本次实际计算的参考结果，便于先看图和理解格式。新实验请使用新的输出目录，避免不同配置覆盖同一组结果。完成上面的可编辑包安装后，检查代码测试可运行 `python -m unittest discover -s tests -v`。

查看结果时先看求解状态和校核，再看恢复曲线。带时限的可行解仍有研究价值，但应报告代价和下界；不能把未收敛时的候选最优方案写成已证全局最优。

## 韧性怎样体现在结果里

恢复不是只看机组是否启动。代码同时观察关键负荷硬保护、其余负荷的供电恢复曲线、未供电量EENS、带权未供电量、储能轨迹与末端储备，以及机组接替供能的时间。恢复曲线下面积对应已恢复供电电量；需求曲线与恢复曲线之间的面积对应未供电量。启动辅助功率若挤占关键恢复资源，代价函数会反映这种机会成本。

“启动代价高于等待”与“所有协同都无效”是不同命题。若允许协同的模型包含不启动方案，理想最优解不会因为多了一个可选动作而变差。本文的D只比较两个条件动作分支；是否存在符号切换，应由不同初始电源、SOC、损伤先验和修复时间的实际扫描结果决定。

## 复现实验的边界

- 默认示例是6节点、3微网、4损伤情景，不冒充IEEE标准系统或真实灾害样本。
- 巡检抢修策略仅在声明的8个候选策略中选择；扩展为多队伍和任意路径需要增加车辆/任务模型。
- 线性网络不提供频率稳定、暂态启动、励磁或保护配合证明。
- 光伏和负荷是示例预测，损伤是显式离散不确定性；可扩展联合时序情景，但不能把三个人工“极端/恶劣/常规”标签自动视为统计保证。
- 能力域为固定离散方案下的条件可行域；单时刻P–Q投影不等于任意24小时轨迹都可行。
- 数据替换或新增约束后，应重新运行校核及有针对性的测试，并重新生成Word算例结果。
