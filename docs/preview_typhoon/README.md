# 台风版实际示例

layout.png是台风产生初始故障后的动作前0−接线，标明主配微和全部开关。weather.png来自合成台风输入和首次故障样本。phase.png来自每格两分支重新优化；默认9格均未判。

默认两种24h实际执行指标见run_summary.json；条件轨迹分解见capability_verification.json；seed12_initial_decision只验证了该样本初始启动不可行，非完整24h结果。

联合策略本次恢复面积77.4633%，规则基线78.0188%；没有隐去基线更好的结果。不能据此宣称算法优势、临界D边界或真实台风工程效果。全部参数、有限预测与限时边界见[代码说明](../../typhoon_version/README.md)。
