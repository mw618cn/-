"""唯一运行入口：运行本文件即可读取Excel、建模、求解、校核并出图。

参数与开关均在同目录的“参数数据.xlsx”，不用修改Python代码。
可以在PyCharm/VS Code中打开本文件点运行，也可以执行 python main.py。
"""
from pathlib import Path
from datetime import datetime
import shutil
import numpy as np
from data import load_parameters, write_result_workbook
from network import draw_network
from repair_model import enumerate_policies
from power_model import build_model
from analysis import run_experiment, save_json
from rolling import run_rolling_replay
from plots import plot_rolling

# 唯一文件位置设置：默认从main.py所在文件夹读取Excel。
PARAMETER_FILE = Path(__file__).resolve().with_name("参数数据.xlsx")


def main(parameter_file=None, output_root=None):
    """以下六步就是整个程序的运行逻辑；其余文件分别实现相应模型。"""
    # 第1步：读取Excel参数表。所有物理参数、96点曲线和运行开关集中于此。
    parameter_file=Path(parameter_file or PARAMETER_FILE).resolve()
    cfg,settings=load_parameters(parameter_file)
    base=Path(output_root) if output_root is not None else Path(str(settings["output_dir"]))
    if not base.is_absolute():base=parameter_file.parent/base
    # 每次运行单独保存，避免把不同参数的旧图和新结果混在一起。
    output=base/ datetime.now().strftime("运行_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    shutil.copy2(parameter_file,output/"本次参数.xlsx")
    save_json(output/"运行开关.json",settings)
    print(f"[1/6] 参数已读取：{cfg['horizon_hours']:g}小时 / {cfg['n_steps']}时段 / {len(cfg['scenarios'])}情景",flush=True)

    # 即使后续求解没有可行解，参数接线图仍然保留，先确认节点/故障/开关。
    draw_network(cfg,output/'figures')
    print(f"33配电节点＋2主网等值 / {len(cfg['edges'])}支路 / {len(cfg['topologies'])}个径向候选接线",flush=True)

    # 第2—4步：建立因果抢修策略→调用恢复模型→比较启动/等待→物理校核。
    # run_experiment内部逐个调用repair_model与power_model，保存真实求解界。
    print("[2—4/6] 建立抢修与电力模型，求解并校核……",flush=True)
    solution,comparison=run_experiment(
        cfg,output,policy_ids=settings["policy_ids"],
        branches=("start","wait") if settings["compare_start_wait"] else (),
        capability=settings["compute_capability"],baselines=settings["compute_baseline"])

    # 第5步：可选的实际滚动。继承SOC与已执行命令，终点始终为原24小时。
    if settings["run_rolling"]:
        print("[5/6] 根据真实巡检观测进行滚动回放……",flush=True)
        policy=next(p for p in enumerate_policies(cfg) if p["id"]==solution["policy_id"])
        model=build_model(cfg,policy)
        checkpoint=output/"checkpoints"/f"{policy['id']}_free_coord.npz"
        with np.load(checkpoint) as data:result={**solution["solver"],"x":data["x"].copy()}
        replay=run_rolling_replay(cfg,policy,model,result,settings["realized_scenario"],
            decision_hours=settings["rolling_hours"],outdir=output/"rolling")
        plot_rolling(replay,output/"figures")
    else:
        print("[5/6] Excel已关闭滚动回放，跳过本步。",flush=True)

    # 第6步：汇总为一份易读Excel；详细CSV与全部图片在同一结果文件夹。
    write_result_workbook(solution,comparison,output/"结果汇总.xlsx")
    print("[6/6] 完成。",flush=True)
    print(f"结果文件夹：{output.resolve()}")
    print(f"期望恢复费用：{solution['summary']['expected_cost']:.4f} 千元")
    print(f"期望失供电量：{solution['summary']['expected_eens_mwh']:.4f} MWh")
    if settings["compare_start_wait"]:print(f"初始D0决策：{comparison['decision']}")
    print("请先打开“结果汇总.xlsx”，图片在figures文件夹。")
    return output


if __name__ == "__main__":
    main()
