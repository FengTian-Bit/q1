# Q1 TCN＋PPO：奖励 v2 与状态估计消融

**固定的从零共同训练任务已整理为 `q1_tcn_v7_joint_v1`。** 使用公共入口 `python wheel_legged_gym/scripts/train.py --task=q1_tcn --headless`，停车和运动两个分支同时训练，不读取旧模型。固定参数、训练/续训/播放命令、源码包及逐文件修改清单见 [共同训练任务说明](../../../docs/q1_tcn_joint.md)。下文保留原实验与续训记录。

目标：平地站稳、零速减少漂移、前后速度跟踪与抗扰。基于 `q1_flat`，默认 27 维本体观测、143 维特权观测；开启下述停车位移反馈后为 30/146 维。六维动作包含四个腿关节位置目标和两个轮速目标。策略为 100 Hz，历史为从旧到新的 16 帧，跨度 0.15 s。本目录集中维护环境、TCN、PPO、训练和评估，原 `q1_flat` 等任务保持原奖励。

使用本目录专用入口，不使用通用 `scripts/train.py` / `scripts/play.py`。

## loco 基础奖励与 ±0.6 m/s 候选结果（历史配置）

原基础版保留在 `logs/loco_reward_search/locked_base_v1/model.pt`，SHA256：`774e52bea0bdf1f945bbb57bf927491cad065cddb383df06fbc1fdc0d9401c11`。用户已通过该±.3 m/s模型，奖励v5、配置和原文件均不覆盖。

本轮待复核候选：`logs/q1_tcn_speed_v4/review_speed060_v1/model.pt`，来源`midrange_polish_seed1/model200`，SHA256：`b147b5fd4faaa948f5308a258b5cc490d4e9cdf9eac5d3b6e3b435643c70f2e0`。按原定偏差≤.025m/s标准，训练第100/200轮三档均通过；额外尝试的≤.020m/s加严筛选未达成，原记录保留。

固定model200后，用未参与选择的701/702/703种子、每档192试验复测：

|速度范围|存活|整段RMSE（含启停）|稳态RMSE|停车末段RMSE|
|---|---:|---:|---:|---:|
|±.25|192/192|.05932|.01837|.01521|
|±.45|192/192|.12197|.02262|.01780|
|±.60|192/192|.17596|.01707|.02347|

单位均为m/s。60秒站立seed701为64/64存活，末端位移均值.946cm/P95 1.967cm，Pitch/Roll RMS 2.761°/1.800°。另在seed704完成正反运动后继续约50秒零指令，64/64存活，锁定后末端位移均值1.073cm/P95 2.281cm。

**严格全项标记仍为false**：仅seed701的±.6停车末段RMSE=.02509277，略高于.025；其余八组均通过。该检查点按`review_candidate`保存，未标记为全部严格验收通过，也未替换原基础版。完整结果、逐机器人CSV和响应图位于候选`evaluation`目录。V3/model600曾在独立seed502出现2/64跌倒，作为失败证据保留。

已确认的失败环节及证据：

|环节|证据|本轮处理|
|---|---|---|
|旧版微调开局更新过大|只继承权重、重建Adam；首轮平均KL=12.87，随后奖励转负|继承Adam，保守固定学习率，KL>.02时停止剩余更新|
|固定轮数课程过早升档|前档仍有跌倒就继续升速|检查通过才升级；当前三档必须共同达标|
|扩速损害低速|同seed211，原基础版±.25稳态RMSE=.02077，V3早期model400=.05358m/s；两者64/64存活|训练指令覆盖全范围，联合学习腿轮平衡；当前再冻结已训练好的骨干|
|零指令制动余量丢失|撤去前馈后，轮速均值退回9.5rad/s，停车分支接管前已倾倒|只延展近饱和的零指令轮输出，均值上限13.5rad/s|
|反向起步余量不足|失败轨迹中两轮目标持续顶在−14rad/s，后倾角及倒退速度仍增大|用现有IMU倾角逐渐开放运动恢复余量，最高20rad/s|

后两项有固定权重对照支持：零指令扩展使seed111的±.5/.55跌倒由4/9次降为0/0；IMU条件运动扩展使V3/model600在seed502的±.6跌倒由2降为0，稳态RMSE=.01786m/s，原前进偏差也得以保留。无条件运动扩展虽能消除跌倒，却增加稳态前进偏差，已作为失败试验保留。旧版错误方向跟踪与后续制动/起步饱和是不同阶段观测到的问题，不能全部归于一个原因。

当前仍使用A0：关闭机身线速度预测头及其监督损失，Actor不输入真实或预测机身线速度。真实速度仅用于奖励、特权Critic和评估。停车位移观测来自编码器、腿部FK和IMU里程计。

运动分支采用期望命令导出的轮速前馈：超出±.375的部分按轮径转换，同时平移策略内部轮角速度和历史动作；实际奖励、评估指令不裁小。网络内嵌冻结的原基础停车TCN，已有停车锁定标志的16帧历史用于平滑交接；两个分支都随检查点保存，play不依赖另一个模型文件。该 ±0.6 m/s 候选使用`freeze_motion_backbone=True`冻结已训练的运动TCN/Actor，只学习两轮小修正头及Critic，原停车网络始终冻结。下文完整范围 v7 配置重新训练运动骨干，不能将此历史配置当作当前配置。

轮输出映射保留小幅控制。`braking_wheel_extension`的start=.85、slope=.3，零指令limit=1.35；运动moving_limit=2，且仅在16帧IMU gravity_x均值的绝对值从.08到.14时渐进开放，并要求原始轮需求与倾斜方向一致。约对应4.6°–8°的前后倾角，不使用机身线速度。腿动作仍限±1，轮执行动作限±2，原控制增益、动作比例及力矩限值不变。无条件扩展、延后交接、传感器平稳条件等旧开关默认关闭，保留用于复现失败对照。

训练使用名义平地、摩擦.8，关闭质量/COM/摩擦随机化及推扰，保留原观测噪声。75%环境为镜像启停往返，持续时间变化20%，先以幅值15%–100%覆盖全范围，最后校准阶段为60%–100%，其余使用原随机指令。`sequence_fraction`默认0，`sequence_min_fraction`默认.85；最后校准配置分别为.75/.60，外部评估关闭训练序列。Adam完整继承，运动骨干阶段学习率1e-5，最后修正头校准3e-5；KL阈值停止后续优化步，不回滚此前更新。

每100轮检查三档，每档64环境，必须共同达标：全存活，稳态RMSE≤.03m/s，正反向偏差≤.025m/s，持续到位比例≥.98，末段停车速度RMSE≤.025m/s；12秒站立位移均值≤2.5cm、Pitch/Roll RMS≤3.5°。原定标准连续两次通过后进入701/702/703独立复测和60秒站立，后者还要求末端位移P95≤5cm。旧501–503及601–603已用于发现问题，不作为最后候选的留出种子。加严.020偏差目标单独保留，未改写原定正式阈值。

```bash
# 查看本轮候选（图形桌面终端，不加headless）
cd ~/q1
bash tools/loco_run_q1.sh -m wheel_legged_gym.envs.q1_tcn.play \
  --checkpoint_path logs/q1_tcn_speed_v4/review_speed060_v1/model.pt \
  --mode tracking --tracking_speed .6 --num_envs 1 \
  --sim_device cuda:0 --rl_device cuda:0

# 手动控制：将mode改为keyboard，W/S每次±.05m/s，空格归零；偏航保持零。
# 原基础版仍用同一入口，checkpoint_path改为logs/loco_reward_search/locked_base_v1/model.pt。

# 从候选继续微调；新输出目录必须为空，仍保留原奖励
bash tools/loco_run_q1.sh tools/q1_reward_iteration.py --mode train \
  --inherit_checkpoint_config --inherit_optimizer \
  --spec tools/reward_pilots/speed_060_midrange_polish.json \
  --warm_start logs/q1_tcn_speed_v4/review_speed060_v1/model.pt \
  --output_dir logs/q1_tcn_speed_v4/continue_review_seed1 \
  --num_envs 2048 --max_iterations 600 --seed 1 --headless \
  --sim_device cuda:0 --rl_device cuda:0

python tools/q1_training_status.py logs/q1_tcn_speed_v4/midrange_polish_seed1

# 独立复测与作图；输出目录必须为新的绝对路径
bash tools/loco_validate_speed_q1.sh CHECKPOINT ABSOLUTE_OUTPUT_DIR 701
bash tools/loco_run_q1.sh tools/q1_summarize_speed.py \
  --root ABSOLUTE_OUTPUT_DIR \
  --baseline_config logs/loco_reward_search/locked_base_v1/resolved_config.json \
  --candidate_config logs/q1_tcn_speed_v4/review_speed060_v1/resolved_config.json \
  --eval_seeds 701 702 703
```

锁定奖励v5的非零权重：tracking_lin_vel=3、tracking_ang_vel=.3、parking_drift=−3、parking_velocity=−.5、orientation=−20、base_height=.5、lateral_velocity=−.5、lin_vel_z=−.2、ang_vel_xy=−.1、action_rate=−.02、torques=−.0001、collision=−1、dof_pos_limits=−2、termination=−200。权重统一乘dt=.01s；运动/零速跟踪sigma=.0025/.0004，停车位置死区.005m、尺度.03m，速度死区.001m/s、尺度.02m/s，高度.397m。所有正式扩速配置逐项核对奖励相等。默认训练入口仍使用下文奖励v2，重现v5必须继承锁定检查点。

原基础版60秒seed201为64/64存活，末端位移均值.97cm/P95 2.27cm，Pitch/Roll RMS 2.75°/1.81°；seed401均值1.00cm。V3/model600的60秒seed501为64/64存活、均值.85cm/P95 1.52cm，但高速独立复测有跌倒，不能只凭站立或训练奖励宣布成功。

`diagnostic.json`、`trials.csv`、`joint_trace.csv`保留原始数据；`q1_reward_report.py --root DIR --plots`输出统计和PNG。整段RMSE含启停；稳态取4–5s及9–10s，末段停车为11–12s。持续到位要求误差<.05m/s连续.5秒，报告的到位时间包含这.5秒。失败机器人不重新加入统计；必须同时查看存活率及有效窗口数。`--trace_failures`与`q1_inspect_failure.py`保存、展开跌倒前两秒的输入和输出。

该候选阶段77项相关CPU测试通过；物理结论以实际仿真复测为准。只有训练种子1，多个评估种子不代表多训练种子复现。该候选有两个TCN分支，不能作为A0～A4等容量消融排名。环境q1_gym（Python3.8、Torch2.4.1+cu121、Isaac Gym Preview4），`loco_run_q1.sh`使用GPU0。该历史阶段尚未启动±1.2及转向，后续实验见完整范围目标训练部分。

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

## 完整速度范围目标训练（奖励 v7，保留 v6 对照）

本轮先训练完整 `vx ∈ [-1.2, 1.2] m/s`，通过独立评估后，再从合格模型训练 `vx ∈ [-1.2, 1.2] m/s、wz ∈ [-1, 1] rad/s`。每次训练从第一轮开始覆盖该阶段的完整范围，不按迭代逐档扩速。采用 A0，不向 Actor 输入真实或显式估计的机身线速度；真实线速度用于奖励、Critic 和评估。

保留原基础版 `logs/loco_reward_search/locked_base_v1/model.pt` 和当前停车候选 `logs/q1_tcn_speed_v4/review_speed060_v1/model.pt`。以下入口使用新目录，不覆盖它们。`review_speed060_v1` 是停车与中速候选，不表示已经通过本轮 1.2 m/s 或转向验收。

当前 v7 的 `parking_policy=true`、`parking_blend=true`、`freeze_motion_backbone=false`：运动 TCN＋Actor、两轮动作修正头及 Critic 参与训练；停车 TCN＋Actor 参数冻结，仍根据每步观测输出动作。停车分支权重来自零指令历史与停车锁定标志，最终动作均值是两分支动作均值的加权和；没有用估计机身线速度决定交接。两个完整 Actor 分支均保存在同一个检查点中，修正头只输出两个轮动作修正值。此结构用于扩速迭代，不能直接与单 Actor 的 A0～A4 作等容量消融比较。

非零奖励权重如下，其他项继承初始化配置中的零权重；实际贡献还经过公共的 `dt` 缩放和单项裁剪。

| 奖励项 | 直线阶段 | 加入转向阶段 |
|---|---:|---:|
| tracking_lin_vel | 4 | 4 |
| tracking_ang_vel | 2 | 2 |
| tracking_wheel_diff | 0 | 0.5 |
| parking_drift | -1 | -1 |
| parking_velocity | -0.2 | -0.2 |
| orientation | -4 | -4 |
| base_height | 0.5 | 0.5 |
| lateral_velocity | -0.5 | -0.5 |
| lin_vel_z | -0.2 | -0.2 |
| ang_vel_xy | -0.05 | -0.05 |
| action_rate | -0.005 | -0.005 |
| torques | -0.0001 | -0.0001 |
| collision | -1 | -1 |
| dof_pos_limits | -2 | -2 |
| termination | -200 | -200 |

运动时的跟踪项为 `exp(-x²) - 0.2 * Huber(x)`，其中 `x = error / width`，Huber 转折点为 1。v6 使用固定宽度 sqrt(0.04)=0.2 m/s 和 sqrt(0.09)=0.3 rad/s。v7 按目标幅值归一化：线速度 width=max(0.1×|vx_cmd|,0.03) m/s；非零偏航 width=max(0.1×|wz_cmd|,0.05) rad/s；运动中的零偏航 width=0.1 rad/s。这样不同速度档位的相对跟踪精度具有可比较的奖励，同时保留远误差 Huber 信号。静止指令的线速度高斯 sigma=0.0004、偏航高斯 sigma=0.09 保持 v6 原值。奖励项权重、停车分支冻结方式和运动网络不变。名义平地摩擦系数 0.8，关闭质量、质心、摩擦随机化和推扰，保留观测噪声。

轮速动作仍使用原单位，每单位对应 10 rad/s；新阶段轮速动作范围 ±3，腿部仍为 ±1。指令参考和高速轮动作增益按保存配置运行，使 1.2 m/s 指令下仍能输出反向轮动作来启动平衡；这不是新增线速度估计器。两轮前馈及转向几何参数写入模型配置，播放必须读取对应模型的完整配置。

这些动作范围是无量纲控制输入的裁剪范围：`normalization.clip_actions=3`，`normalization.leg_clip_actions=1`。控制器用 `vel_action_scale=10` 换算轮速目标，所以轮速目标上限为 ±30 rad/s；腿部用 `pos_action_scale=0.35` 换算默认角度附近的角度目标，所以最大偏移为 ±0.35 rad。默认大腿 0.6 rad、小腿 -1.2 rad，对应目标区间 [0.25,0.95] 和 [-1.55,-0.85] rad。策略内部 `mean_limit=0.95` 约束基础均值，轮动作另有前馈、残差和大输出扩展；腿部确定性均值范围实际为 ±0.95，训练噪声后仍由 ±1 裁剪保护。URDF 的机械限位和腿部 17 N·m/轮部 6 N·m 力矩上限独立保留。

在 loco 的 `tt1` 中执行两次训练及中间验收：

```bash
bash tools/loco_target_plan_q1.sh /home/ubuntu/q1/logs/q1_target_v7/production_v7_relative
```

每阶段 4096 环境，最多 4000 轮，每 250 轮筛查一次；筛查通过才进行 3 个未用于训练的评估种子、每种子 64 个机器人验证。只有直线阶段验证通过才自动开始转向阶段；达到预算仍未通过会停止并保留结果，不以训练奖励宣称达标。运行中的同名目录不能再次启动。

无对称损失的配置为 `target_linear_v6_accel2.json` 和 `target_turning_v6_accel2.json`，相对上一版 stable 配置仅增加用户指定的线速度指令加速度限制 2 m/s²，奖励、网络和优化器不变。stable 配置仍保留：相对最初 v6 配置，大轮速输出区间 slope 从 0.3 调为 1.0，直线阶段零偏航跟踪权重从 0.5 调为 3。依据是同一 model_300 固定权重对照：1.2 m/s 启停中 slope=0.3/0.6/1.0/1.5 分别存活 0/8、5/8、8/8、3/8；slope=1.0 时停车漂移均值 1.85 cm，中速与冷启动站立回归也各存活 8/8。这是参数诊断，不能代替正式训练模型的独立验收。最初配置文件和 production_v2 的模型、图表全部保留。

加入对称损失后，批量入口默认选择 `target_linear_v6_accel2_sym.json` 和 `target_turning_v6_accel2_sym.json`。原 accel2 配置保留作无对称损失对照。

最初 v7 使用 `target_linear_v7_accel2_sym.json` 和 `target_turning_v7_accel2_sym.json`，角速度权重为3，这两个历史配置保留。v6 对称损失训练第1500轮的筛查仍有约0.14/0.195 rad/s正向偏航误差，以及中速反向偏差；最初 v7 仅收紧运动跟踪的奖励误差宽度，保持评估门槛、奖励权重、停车奖励、网络、对称损失0.1及2m/s²指令限制不变。`rewards.relative_tracking` 保存上述四个宽度参数。

用户后续要求角速度权重为线速度的一半，并增加差速转向引导。当前批量入口默认直线配置为 `target_linear_v7_yaw_half.json`，转向配置为 `target_turning_v7_yaw_half_wheel_diff.json`：线速度4、机身角速度2、轮速差引导0.5（仅转向阶段）。原来的机身角速度指数项已经存在，保留其与线速度相同的指数＋Huber结构，不重复添加角速度奖励，也不修改误差宽度。

轮速差引导采用 `wheel_yaw=(omega_right-omega_left)*r/d`，奖励为 `exp(-(wz_command-wheel_yaw)^2/sigma)`。r=0.07496 m、轮距d=0.299 m、sigma=1.0 (rad/s)^2；因此1 rad/s转向对应约3.99 rad/s的右减左轮速差。只在 `abs(wz_command)>=0.05 rad/s` 且双轮接触力均高于3 N时启用，零转向指令或任一轮离地时为零。输入使用实测轮关节速度，辅助奖励鼓励正确轮速差；实际机身角速度仍由主奖励和全部最终评估约束，不能用轮转动代替机身转向。此项不向Actor新增观测或估计器。

旧转向模型保留到第600轮检查点后，从它恢复网络、Adam、迭代计数和RNG，应用用户要求的两个奖励更新。GPU上4096环境、两轮PPO更新验证通过：初始化网络与Adam完全一致，停车35个张量更新后仍不变；dt缩放后的线速度/角速度/差速奖励权重分别为0.04/0.02/0.005。99项CPU测试通过。该检查仅验证实现，尚不表示转向性能达标。

当前实际续训命令如下（重跑需另选新输出目录）：

```bash
bash tools/loco_run_q1.sh -m wheel_legged_gym.envs.q1_tcn.target_train \
  --stage turning --spec tools/reward_pilots/target_turning_v7_yaw_half_wheel_diff.json \
  --warm_start logs/q1_target_v7/production_v7_relative/turning_seed1/model_600.pt \
  --continue_training --update_yaw_reward_weight --update_wheel_diff_reward \
  --output_dir logs/q1_target_v7/production_v7_relative/turning_yaw2_diff_seed1 \
  --num_envs 4096 --seed 1 --max_iterations 4000 --screen_interval 250 \
  --headless --sim_device cuda:0 --rl_device cuda:0
```

两个更新开关必须与续训同时使用：前者仅允许角速度权重变为未变线速度权重的一半，后者仅允许增加差速奖励权重及其几何/宽度设置；其他奖励和控制条件仍严格匹配。对旧直线模型批量续训时，可用 `Q1_UPDATE_YAW_WEIGHT=1` 显式启用半权重更新。运动探索标准差仍为0.08，未采用固定权重诊断中的增益3/6或偏航参考变换。增益6原地转向16/16跌倒，两个参考表达也未有效解决转向，相关失败证据保留在 `turning100_yaw_gain_probe` 和 `turning100_yaw_reference_probe`。不同奖励配置的训练回报不能直接当作性能提升，仍使用统一的机身速度/角速度评估。

2026-11-05 开发记录：固定 `production_v7_relative/linear_seed1/model_1750.pt`（SHA256 `22363f74f6fc855c46d5a9e367955108fae3a8240936fc65142fd2e54bc07955`）在 1451/1452/1453 种子完成四档速度复核，768/768 次运动试验存活。0.3/0.6/0.9/1.2 m/s 的最差种子、最差正反阶段线速度得分分别为 91.54%/94.94%/92.14%/93.53%；姿态、停车、指令加速度项通过。60 s 站立 64/64 存活，末端位移均值 0.850 cm、P95 2.080 cm，Pitch/Roll RMS 2.752°/1.799°。全部原始结果位于 `logs/q1_target_v7/production_v7_relative/linear1750_full_audit`。

该模型的严格直线评估仍为 false：中高速前进的零偏航 RMSE 超过 0.1 rad/s。根据先完成约 90% 线速度跟踪、再加入转向的训练顺序，保存 `linear_to_turning_handoff.json` 后，从这个经过完整复核的模型启动转向训练；该文件明确 `final_goal_passed=false` 并保留九个未通过的偏航检查。原直线进程停止时保留至 model_2200，未将未经完整复核的更晚模型替换为初始化模型。此为人工开发阶段切换，批量脚本的严格自动门槛仍保留；最终模型必须通过原直线、原地转向和组合指令的全部检查，包括直行零偏航要求。

本次转向训练的实际启动命令如下（目录已被运行占用，重跑须另选新目录）：

```bash
bash tools/loco_run_q1.sh -m wheel_legged_gym.envs.q1_tcn.target_train \
  --stage turning --spec tools/reward_pilots/target_turning_v7_accel2_sym.json \
  --warm_start logs/q1_target_v7/production_v7_relative/linear_seed1/model_1750.pt \
  --output_dir logs/q1_target_v7/production_v7_relative/turning_seed1 \
  --num_envs 4096 --seed 1 --max_iterations 4000 --screen_interval 250 \
  --headless --sim_device cuda:0 --rl_device cuda:0
```

`production_v3` 在启动检查时发现入口遗漏训练前的环境重置；修正后在 `production_v4` 从原候选重新初始化，参数不再变化。新入口在 PPO 首次采样前调用 `env.reset()`，填充正确关节状态、指令和历史；初始化清单记录该步骤，并有入口生命周期回归测试。

`production_v4` 的阶跃指令实验保留至第 750 轮筛查，因用户新增加速度限制要求，在 `production_v5_accel2` 从原 ±0.6 m/s 候选重新训练。`commands.max_linear_acceleration=2.0` 限制的是前后速度指令变化率，不使用机身线速度估计：100 Hz 下每步最多变化 0.02 m/s，0→1.2 m/s 和 1.2→0 均约 0.6 s，+1.2→-1.2 约 1.2 s。随机指令、序列指令、评估和 play 使用同一限幅函数；新回合从零指令开始。该限制属于回合内指令整形，训练目标范围从第一轮即为完整 ±1.2 m/s。偏航指令保持原设置。

评估 CSV 和响应图同时记录目标速度、限幅后的速度指令、指令加速度和机身实际前向加速度。额外检查指令加速度不超过 2 m/s²；实际加速度作为统计结果报告，不声称机器人瞬时实际加速度被硬限制为 2 m/s²。`max_linear_acceleration=0` 保留旧阶跃行为，旧模型按其原配置播放。

`target_result.json` 记录阶段状态和被评估模型路径。`screens/`、`holdouts/` 下包含 `summary.json`、逐场景 `result.json`、`trials.csv`、`trace.csv` 和 `response.png`。训练的 `initialization.json` 记录初始化模型哈希、完整范围及停车分支是否冻结。

本轮“90%”定义为稳态分试验 RMSE 相对指令幅值不超过 10%，分别检查正、反方向。小指令使用绝对误差下限：线速度 0.03 m/s、偏航 0.05 rad/s。零线速度轴容差 0.05 m/s，零偏航轴容差 0.1 rad/s。评估指令为停 0–2 s、正向 2–6 s、停 6–9 s、反向 9–13 s、停 13–16 s，稳态窗口取 5–6 s 和 12–13 s；这不代表瞬时阶跃全程误差都小于 10%。

直线验收覆盖 0.3/0.6/0.9/1.2 m/s；转向验收另外覆盖 0.5/1 rad/s 原地转向及 0.6+0.5、1.2+1、1.2-1 组合指令。要求全部试验完整存活，运动姿态 RMS≤10°，冷启动站立姿态 RMS≤4°。停车仅做回归检查：末秒速度 RMSE≤0.08 m/s、偏航≤0.15 rad/s，锁定后最终漂移均值≤5 cm/P95≤10 cm；冷启动站立为均值≤2.5 cm/P95≤5 cm。本协议不复用旧版 0.025 m/s 的严格速度门槛。

筛查的冷启动站立为 20 s，独立正式验收为 60 s；`summary.json` 同时记录实际站立时长。响应 CSV 另含轮速目标、前一步实测轮速、轮动作原始输出与停车分支权重，便于区分稳态跟踪误差和高速制动失败。

无需 Isaac Gym 即可导出各场景关键统计和汇总图，将 `EVAL_DIR` 替换为一次完整评估的目录：

```bash
python tools/q1_target_report.py --summary EVAL_DIR/summary.json --output_dir logs/q1_target_v6/target_report
# 加 --training_dir RUN_DIR 可一并保存训练 CSV 快照和 training_progress.png（含对称损失曲线）。
```

输出 `metrics.csv`、`comparison.png`、`source.json`。图中跟踪得分取所有评估种子和正反阶段中的最差值，零指令轴按绝对容差单独显示。缺失完整窗口的指标留空，存活图仍保留失败试验；报告不会重新判定模型合格。`source.json` 保留评估文件、模型和哈希的对应关系。

手动复评模型（将 MODEL 替换为 `target_result.json` 中的 checkpoint）：

```bash
bash tools/loco_run_q1.sh -m wheel_legged_gym.envs.q1_tcn.target_evaluate --checkpoint_path MODEL --stage linear --seeds 1401 1402 1403 --num_envs 64 --headless --sim_device cuda:0 --rl_device cuda:0 --output_dir logs/q1_target_v6/recheck_linear
# 转向模型使用 --stage turning，另选新的 output_dir。
```

有显示设备的机器上播放，首次仍需安装相同仓库和 Isaac Gym 环境：

```bash
python -m wheel_legged_gym.envs.q1_tcn.play --checkpoint_path MODEL --mode tracking --tracking_speed 1.2 --num_envs 1 --sim_device cuda:0 --rl_device cuda:0
python -m wheel_legged_gym.envs.q1_tcn.play --checkpoint_path MODEL --mode combined --tracking_speed 1.2 --yaw_speed 1 --num_envs 1 --sim_device cuda:0 --rl_device cuda:0
python -m wheel_legged_gym.envs.q1_tcn.play --checkpoint_path MODEL --mode keyboard --num_envs 1 --sim_device cuda:0 --rl_device cuda:0
```

键盘 W/S 调整前后速度，A/D 调整转向，空格同时置零。`stand` 为静止，`turning` 为原地转向；超出模型保存范围的自动播放参数会被拒绝。播放的启停周期为 12 s，正式定量验收以 `target_evaluate` 的 16 s 协议为准。

### 左右对称损失

`symmetry.py` 沿用四轮足 DreamWaQ 的镜像动作均值 MSE，重新定义 Q1 映射。Q1 的六个关节轴均为 +Y，动作镜像为 `[3,4,5,0,1,2]`，不改变关节动作符号。角速度的 X/Z 分量、重力的 Y 分量、偏航指令和停车横向误差取反，位置误差、关节速度、上一帧动作交换左右。16 帧历史逐帧映射，保留原时间顺序；支持 27/30 维 Actor 观测。该模块不处理特权 Critic，不增加机身线速度输入。

`L_sym = mean((mu(o,h) - mirror_action(mu(mirror(o),mirror(h))))²)`，两个分支均保留梯度，使用确定性动作均值。PPO 总损失增加 `algorithm.symmetry_loss_coef * L_sym`；默认值 0 兼容旧实验，本轮设置 0.1。`symmetry_batch_size=4096` 限制每个已随机打乱的 PPO 小批次中用于镜像前向的样本数，以控制额外显存。完全接管的冻结停车分支权重为零，过渡阶段按运动分支权重加权；停车网络参数继续冻结。

`training.csv` 增加原始 `symmetry_loss` 和乘系数后的 `symmetry_loss_weighted`。将 `algorithm.symmetry_loss_coef` 设为 0 可关闭该项，进行同配置对照。对称约束允许镜像转向状态产生镜像轮速差，不要求所有状态下左右轮动作相等。

可以从已经覆盖完整速度范围的检查点继续当前阶段，恢复优化器、RNG 和累计迭代计数，仅允许显式修改对称损失系数；控制、奖励、观测和指令范围仍严格匹配。仿真从新回合开始：

```bash
Q1_LINEAR_WARM_START=MODEL Q1_CONTINUE_LINEAR=1 bash tools/loco_target_plan_q1.sh /home/ubuntu/q1/logs/q1_target_v6/production_v6_sym
```

`initialization.json` 记录 `continued_from_iteration` 和 `continued_from_total_steps`。这不是逐档扩速，续训前后目标均为完整 ±1.2 m/s；直线阶段通过后，转向阶段仍从首轮覆盖完整 ±1 rad/s。未设置上述环境变量时，从保留的 ±0.6 m/s 候选初始化并从首轮开启对称损失。

从 v6 的完整范围检查点继续 v7，须显式增加 `Q1_UPDATE_TRACKING_REWARD=1`。该标志只允许 `reward_version` 与 `relative_tracking` 这两项变化，其他奖励项和控制配置仍严格匹配，权重、Adam 状态、RNG、累计迭代计数继续保留：

```bash
Q1_LINEAR_WARM_START=MODEL Q1_CONTINUE_LINEAR=1 Q1_UPDATE_TRACKING_REWARD=1 bash tools/loco_target_plan_q1.sh /home/ubuntu/q1/logs/q1_target_v7/production_v7_relative
```
