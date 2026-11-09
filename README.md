# 主配微协同韧性恢复 R3

**[下载R3完整代码包](https://github.com/mw618cn/-/raw/refs/heads/main/downloads/tdm_resilience_r3.zip)** · [详细使用说明](resilience_v3/README.md) · [模型与求解逻辑](resilience_v3/模型与求解逻辑.md)

完整解压到新目录，只运行 **main.py**，只维护 **参数数据.xlsx**。首次安装依赖：

```bash
python -m pip install -r requirements.txt
python main.py
```

仓库根目录main.py调用 [resilience_v3](resilience_v3/)。独立ZIP中main.py直接位于包内顶层，无需安装项目为Python包。

保留三个研究点：①启动/等待恢复价值判据；②多时段可实施能力与条件后向可达；③检查信息和实体承诺分离的抢修—重构—启动滚动协同。集中HiGHS MILP/LP及有限因果派工策略类已实现，ADMM未实现。

- 默认33配网节点＋T0/T1主网等值，三个微网；24h/96时段，故障/修复/信息事件动态更新。
- 后向递推从终端SOC集合计算安全后备集和给定指令的条件管；同信息先交可能子分支，观测后仅保留兼容后继。
- 每15min连续LP、周期/事件联合MILP，只执行首步；继承已执行历史、SOC、开关保持与机组/任务承诺。
- 全时域PCC支持见证、共同系数轨迹凸包、整轨迹分解及真实拼接检验；MGO资源能力与网络可送能力分开。
- D差值、有效上下界和未判分开保存；机组已启动后不编造新的自由启动D点，不平滑生成未经证明的临界面。
- 事件甘特图、信息分组、SOC后向管、节点服务状态和滚动控制器图；所有交付图严格使用原版宋体。

云端尚缺完整原版宋体，**未提供替代字体预览**。ZIP内数值示例可在Windows使用系统宋体直接重画：

```bash
python main.py --redraw 示例结果_24小时
```

参数/源码导读见[逐段说明](resilience_v3/模块与参数逐段说明.md)，研究定位见[三点对应](resilience_v3/三个创新点与代码对应.md)，图与文献边界见[图形说明](resilience_v3/图形与文献依据.md)，实际验证与限制见[验证说明](resilience_v3/验证说明.md)。Word附可编辑OMML公式，另附LaTeX源。

标准33节点基础电气数据可追溯；风场、脆弱性、GIS、工时及机组等为待校准合成参数。有限样本和有限策略类不构成一般POMDP最优、样本外递归可行或暂态安全证明。默认短求解预算可能使整个D网格未判，不能据此宣称已获得临界边界或明显协同增益。

旧版保留：[R2](typhoon_version/) · [R2完整包](https://github.com/mw618cn/-/raw/refs/heads/main/downloads/tdm_typhoon_33_r2.zip) · [固定故障33节点版](ieee33_version/) · [6节点版](simple_version/)。旧版结果不作为R3验证结果。
