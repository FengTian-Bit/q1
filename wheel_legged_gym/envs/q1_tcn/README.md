# Q1 TCN v3 r11r11000

注册任务`q1_tcn`现使用v3环境、网络和PPO，实现来自`r11r_g10000_parking03_consolidate`运行根。默认配置保存在`v3_config.json`，与选中11000轮检查点的环境及训练配置一致。配置JSON不含模型权重。

## 训练和续训

在已安装Isaac Gym的Linux环境，从仓库根目录执行：

```bash
python wheel_legged_gym/scripts/train.py --task=q1_tcn --headless
# 也可使用模块入口
python -m wheel_legged_gym.envs.q1_tcn.train --task=q1_tcn --headless
# 续训保留模型、Adam、RNG和累计轮次；max_iterations是绝对目标轮次
python wheel_legged_gym/scripts/train.py --task=q1_tcn --resume --load_run=/absolute/path/to/run --checkpoint=11000 --max_iterations=12000 --headless
```

默认2048环境、训练种子2、每更新48步、绝对预算12000。新训练从随机权重和空Adam开始，使用v3最终阶段配置；这不是对原历史训练链的从0复现保证。原11000模型经过r9i、覆盖修复、r11g及停车收敛分支。重现该检查点的后续行为应提供实际checkpoint并续训，配置本身不能恢复原策略。仿真、episode和地形课程状态不在checkpoint中，续训会重新创建。

默认奖励含Huber停车残差阈值0.25m、停车权重-0.3；entropy_coef=0.002、desired_kl=0.004。环境为20×20地形、0.50m踏面、3m平台及地形/命令交错分配。保留接触引导教师及累计时钟；选中的11000轮模型教师已退场，fresh训练仍按保存的教师日程运行。不要把本任务当作旧V46、旧停车分支或r10c定版v2配方。

每更新的CSV包含奖励、终止、KL、动作std/限幅及采样/PPO耗时。检查点包含配置、模型、Adam、RNG、累计更新/步数。公共入口不依赖旧实验spec、旧基线日志、ONNX或MuJoCo部署。公共入口当前使用CSV与控制台日志；不接旧v7的SwanLab/停车双策略训练器。

## 网络与控制

继承关系为`Q1TCN → Q1Flat → LeggedRobot`。当前观测30维，存储61帧（1830维），快TCN读取最近16帧，慢TCN读取全部61帧。Actor输入包含快/慢latent、坡度和净空预测；显式速度、接触和滑移槽在A0中关闭。Critic使用146维特权观测。Actor层宽128/64/32，Critic256/128/64。

物理步长0.005s、控制decimation2，即100Hz控制。六动作顺序为左thigh/calf/foot、右thigh/calf/foot。腿动作尺度[0.5,1.5,0.5,1.5]，限幅[0.7,0.95,0.7,0.95]为动作单位；轮速scale10，启用保存的轮速前馈。`q1_flat.py`是v3控制链的必要共享依赖。

## 回放与评估

```bash
python wheel_legged_gym/scripts/play.py --task=q1_tcn --load_run=/absolute/path/to/run --checkpoint=11000 --num_envs=1
python -m wheel_legged_gym.envs.q1_tcn.target_evaluate --checkpoint_path=/absolute/path/model_11000.pt --output_dir=logs/v3_eval/flat --stage=turning --seeds 2301 2302 --num_envs=32 --headless
python -m wheel_legged_gym.envs.q1_tcn.target_evaluate --checkpoint_path=/absolute/path/model_11000.pt --output_dir=logs/v3_eval/slope --stage=slope --seeds 2301 2302 --num_envs=32 --headless
Q1_STAIR_WIDTH=0.50 python -m wheel_legged_gym.envs.q1_tcn.target_evaluate --checkpoint_path=/absolute/path/model_11000.pt --output_dir=logs/v3_eval/stairs --stage=stairs --seeds 2301 2302 --num_envs=32 --headless
```

回放W/S前后、A/D转向、空格停车。评估用同一检查点、固定2301/2302各32台，不能用存活代替登阶。完整统一验收仍未通过：原11000轮上8/12/16cm成功95.3125/95.3125/96.875%，上20cm为0；站立漂移均值/P95约2.498/5.441cm，坡T1-2/T1-7未过。冻结优秀检查点不等于所有能力达标或从0稳定复现。

## 来源和验证

原checkpoint SHA256：`586c4454ab58ae6c98bda2ac4d62b5f62afbe0740a6cf3e2617644c068cc96f8`。核心环境、TCN、PPO、runner和控制源码按冻结v3同步；`v3_profile.py`与`v3_task.py`只负责配置装载及公共train/play适配。测试使用真实模块数值计算和tensor-only仿真桩，不证明物理登阶效果：

```bash
python -m unittest discover -s tests -p test_q1_tcn_v3.py -v
python -m unittest discover -s tests -p test_q1_tcn.py -v
```
