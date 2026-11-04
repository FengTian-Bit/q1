# Q1 TCN＋PPO：奖励 v2 与状态估计消融

目标：平地站稳、零速减少漂移、前后速度跟踪与抗扰。基于 `q1_flat`，保留 27 维本体观测、143 维特权观测和六维动作（四个腿关节位置目标＋两个轮速目标）。策略为 100 Hz，历史为从旧到新的 16 帧，跨度 0.15 s。本目录集中维护环境、TCN、PPO、训练和评估，原 `q1_flat` 等任务保持原奖励。

使用本目录专用入口，不使用通用 `scripts/train.py` / `scripts/play.py`。

## 奖励 v2

所有组使用同一奖励。表中权重统一乘控制周期 dt=0.01 s；普通分项加权后裁剪到 ±0.1，终止项单独加入；允许总奖励为负。

| 项目 | 配置权重 | 原始奖励/惩罚 |
|---|---:|---|
| tracking_lin_vel | +3.0 | exp(-(vx-vx_cmd)^2 / 0.01) |
| tracking_ang_vel | +0.3 | exp(-(wz-wz_cmd)^2 / 0.04) |
| parking_drift | -2.0 | 停车位移惩罚，见下文 |
| orientation | -2.0 | projected_gravity 的 XY 分量平方和 |
| base_height | +0.5 | exp(-(h-h_cmd)^2 / 0.0025)，默认 h_cmd=0.397 m |
| lateral_velocity | -0.5 | vy^2 |
| lin_vel_z | -0.2 | vz^2 |
| ang_vel_xy | -0.05 | wx^2+wy^2 |
| action_rate | -0.01 | 六维执行动作的一阶差分平方和 |
| torques | -0.0001 | 六个电机力矩平方和 |
| collision | -1.0 | 机身/大腿/小腿接触力 >0.1 N 的部件数 |
| dof_pos_limits | -2.0 | 四个腿关节超出软限位的角度总和 |
| termination | -200.0 | 非正常跌倒；100 Hz 下每次实际扣 2 分，正常超时不扣 |

删除 alive、nominal_state、dof_vel、dof_acc、action_smooth，配置中设为零以便读取旧模型配置。未增加轮速静止、腿部对称、接触力均分或滑移奖励。跌倒与超时同时发生时仍按实际跌倒扣分。

停车状态为 |vx_cmd|<0.02 m/s 且 |wz_cmd|<0.05 rad/s。从运动切换到停车，先减速 0.5 s，再锁定当前世界 XY 位置；随后 0.5 s 内把惩罚权重从零渐入到一。初始站立使用重置位置作为参考，同样保留渐入前的 0.5 s 缓冲。锁定不依赖实际速度，防止一直不减速就绕过停车惩罚；参考点锁定后不滚动更新，直到再次运动或重置。

原始停车惩罚为 `alpha * min((max(position_error - 0.02, 0) / 0.10)^2, 1)`，再乘权重 -2 和 dt。允许 2 cm 平衡调整，运动期间关闭。减速期间的位移与锁定之后的漂移分别统计。

**停车真位置只用于奖励和评估，不输入 Actor。** 当前短历史本体策略可学习抑制漂移，但不能据此承诺长期返回某个世界坐标停车点。若需要可部署的位置闭环，应另外接入里程计/融合定位的相对停车位移，并作为独立接口版本重新验证。

## 网络与 A0～A4

TCN：64 通道，四个因果残差卷积块，kernel=2，dilation=[1,2,4,8]，感受野 16 帧。输出 z[16]、机身速度[3]、左右轮接触 logits[2]、左右轮有符号纵向滑移[2]。Actor 为 [128,64,32]，Critic 为 [256,128,64]。

| 组 | 隐特征 z | 速度 v | 接触 c | 滑移 s |
|---|---|---|---|---|
| A0 | 开 | 关 | 关 | 关 |
| A1 | 开 | 开 | 关 | 关 |
| A2 | 开 | 开 | 开 | 关 |
| A3 | 开 | 开 | 关 | 开 |
| A4 | 开 | 开 | 开 | 开 |

Actor 始终保留七个显式槽位，关闭的槽位置零，关闭对应监督损失；参数量相同。PPO 通过 z 更新 TCN，显式输出送入 Actor 前 detach，各辅助损失也更新 TCN。Critic 不拼接 z。熵系数和网络容量沿用原版，奖励实验中不同时修改。

标签固定为 `[v_body(3), contact(2), slip(2), frame_valid(1), slip_valid(2)]`。执行动作前克隆历史及标签，重置帧的接触监督无效，历史以同一首帧重复填充。速度和滑移使用归一化 Smooth L1（默认尺度 1 m/s），接触使用 BCEWithLogits，默认辅助权重均为 1。

接触标签采用世界 Z 承载力：5 N 开、3 N 关，连续两个策略帧确认切换。滑移为几何轮底点沿机身水平前向的有符号速度，考虑绝对角速度、圆柱碰撞中心和 COM 偏移；即时承载力 >3 N 才有滑移监督。仅适用于当前圆柱轮和平地，不用于台阶。

## 先验证 A0

在仓库根目录、已安装 Isaac Gym 的训练环境中运行。新版默认目录为 `logs/q1_tcn_reward_v2`，旧版结果和模型保留在原处。

```bash
python wheel_legged_gym/envs/q1_tcn/batch.py --num_envs 4096 --ablations A0 --seeds 1
```

这会从头训练奖励 v2 的 A0，3000 轮后自动评估并生成图表。先查看站立、零指令速度、锁定后漂移、启停跟踪、动作裁剪率以及各奖励项贡献，再决定是否启动完整消融。

短训练检查：

```bash
python -m wheel_legged_gym.envs.q1_tcn.train --ablation A0 --seed 1 --num_envs 64 --max_iterations 10 --headless --sim_device cuda:0 --rl_device cuda:0 --output_dir logs/q1_tcn_reward_v2/smoke_A0
```

正式训练单组：

```bash
python -m wheel_legged_gym.envs.q1_tcn.train --ablation A0 --seed 1 --num_envs 4096 --max_iterations 3000 --headless --sim_device cuda:0 --rl_device cuda:0 --output_dir logs/q1_tcn_reward_v2/A0_seed1
```

## 全部消融与恢复

```bash
python wheel_legged_gym/envs/q1_tcn/batch.py --num_envs 4096
```

按 seed1 的 A0→A1→A2→A3→A4，再 seed2、seed3 顺序完成 15 次训练，每次结束自动评估，全部结束生成 comparison.csv、comparison.json、comparison.png。每组 4096×48×3000=589824000 个交互步。之前完成的新版 A0 会跳过训练。

可选：`--seeds 1` 只做一个训练种子；`--train_only` 只训练；`--dry_run` 只打印计划；`--output_root PATH` 选择其他目录。当前若有训练进程占用 GPU，等该进程退出后再启动批量入口。

中断后重跑相同命令，脚本从最近可读检查点补足到总目标轮数。恢复写入 resume 子目录，保留已有文件；尚无检查点则从头开始。失败立即停止队列，状态存入 ablation_batch.json。恢复优化器和计数器，仿真从新回合开始，不是逐物理步恢复。

**新版训练拒绝复用奖励 v1 的检查点。** 奖励已经改变，新版消融应全部从头训练，不能把旧 A0 或旧 A1 与新奖励结果混在一起。

## 评估与旧模型复评

评估协议 v3 包含高度初始化修复：在 reset 填充历史前设置名义高度 0.397 m，并记录 protocol_version、command_height_m、reward_version 和停车参数。旧入口开启外部指令后未初始化高度，实际给出了 0 m 指令，旧评估不能作为名义工况的消融排名。

单模型评估：

```bash
python -m wheel_legged_gym.envs.q1_tcn.evaluate --checkpoint_path logs/q1_tcn_reward_v2/A0_seed1/model_3000.pt --num_envs 64 --eval_seeds 101 102 103 --duration 12 --headless --sim_device cuda:0 --rl_device cuda:0 --output_dir logs/q1_tcn_reward_v2/A0_seed1/evaluation
```

已有奖励 v1 的 A0～A3 模型无需因高度缺陷而重训，下面只补评估，不启动训练：

```bash
python wheel_legged_gym/envs/q1_tcn/batch.py --evaluate_only --num_envs 4096 --seeds 1 --ablations A0 A1 A2 A3 --output_root logs/q1_tcn_4096
```

旧模型按保存的原奖励配置读取；新停车奖励不会被悄悄打开。批量入口不复用旧协议评估，并选择新的 attempt 输出目录保留旧结果。该命令要求相应模型已达到目标 3000 轮。

七种场景，每种 3 个评估种子×64 个机器人、最长 12 s：

| 场景 | 条件 |
|---|---|
| stand | 零速，摩擦 0.8 |
| tracking | 0–2 s 停，2–5 s +0.25 m/s，5–7 s 停，7–10 s -0.25 m/s，之后停 |
| push_front / push_back / push_side | 零速，3–3.15 s 分别沿 +X / -X / +Y 对 trunk COM 施加 8 N |
| low_friction | 跟踪序列，摩擦 0.2 |
| payload | 跟踪序列，trunk 增重 0.75 kg，COM 沿 X 偏置 +0.02 m |

评估保留观测噪声，关闭随机参数及随机推扰，使用 Actor 均值动作。训练则保留共同的摩擦、质量、COM 随机化和有限推力。评估可用 `--scenarios stand tracking` 选择子集，用 `--push_force 12` 更改固定推力；所有比较组须协议一致。

## 输出与判断

- training.csv：环境原始 reward、逐项 reward_*（实际乘 dt 和裁剪后的贡献）、跌倒/超时数、PPO 与辅助损失、KL、熵、学习率、action_clipping_fraction、policy_std_max。
- 检查点与 resolved_config.json：完整配置与优化器状态；source/ 保存本目录及继承的平地、基类奖励源码；同时保存 q1.urdf。
- summary.json / summary.csv：各指标均值、标准差、有效样本数与协议；trials.csv 保留逐试验统计；trace.csv 保留每时刻仍存活试验的响应均值。
- 场景 PNG：速度、姿态、支撑滑移、支撑力、存活试验数、锁定后停车误差、动作裁剪比例。

停车指标为 locked_parking_rms_m、locked_parking_max_m、locked_parking_end_m、locked_parking_observed_s；减速位移单独记录为 braking_displacement_max_m。zero_command_speed_rmse_m_s 包含零指令下的减速阶段。原 end_displacement_m 是相对试验起点的位移，跟踪场景中不能解释为停车漂移。

其他指标包括速度 RMSE、Pitch/Roll RMS 与峰值、存活比例、动作裁剪比例、执行器力矩饱和、承载期间滑移、异常空转、估计误差、接触分类与检测延迟。动作裁剪与力矩饱和是不同指标。

推后恢复要求在推力结束后 3 s 内，速度误差 <0.05 m/s、Pitch/Roll 绝对值 <0.12 rad 持续 0.5 s，且随后未跌倒；同时报告实际活到推力开始的比例。跟踪稳定时间要求速度误差进入 ±0.05 m/s 并持续 0.5 s。

先看跌倒与观测时长，再看误差。失败前最后状态参与统计，重置后的状态不重新计入试验；提前跌倒可能造成较低 RMSE。空值表示无有效样本，不是零误差。3 个评估种子不等于 3 个独立训练种子。

比较工具可在无 Isaac Gym 的机器上按文件路径运行：

```bash
python wheel_legged_gym/envs/q1_tcn/compare.py --inputs logs/q1_tcn_reward_v2/A0_seed1/evaluation/summary.json logs/q1_tcn_reward_v2/A1_seed1/evaluation/summary.json --output_dir logs/q1_tcn_reward_v2/comparison
```

可加入所有组与所有训练种子的 summary.json。工具要求相同预算、训练条件、网络和评估协议，禁止重复训练种子；先按训练种子汇总，再计算跨训练种子标准差。

## 验证

```bash
python -m unittest discover -s tests -p "test_q1_tcn*.py" -v
```

CPU 测试覆盖奖励权重/缩放/终止、停车切换/渐入/重置、分项日志与总奖励对账、因果性、梯度、标签、GAE、五组 PPO 更新、检查点、批量恢复和评估高度。张量替身不验证真实接触或训练收敛；先做 A0 仿真验收。
