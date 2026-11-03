# Q1 TCN＋PPO：平地状态估计消融

基于 `q1_flat`，保留 27 维本体观测、143 维特权观测、4 个腿关节位置目标＋2 个轮速目标。策略频率 100 Hz，历史为从旧到新的 16 帧，首尾跨度 0.15 s。环境、标签、网络、PPO、训练入口和评估均在本目录。现有任务的网络和 PPO 不受影响。

第一阶段只研究站立、前后速度跟踪和抗扰，偏航指令固定为零。使用本目录的专用训练入口，**不要使用通用 `scripts/train.py` / `scripts/play.py`**，它们的旧编码器接口不适用于本任务。

## 文件与网络

| 文件 | 用途 |
|---|---|
| `q1_tcn_config.py` | 共同环境、随机化、网络和 PPO 配置，A0～A4 开关表 |
| `q1_tcn.py` / `labels.py` | 平地环境、接触迟滞、有符号接触点滑移、终止前状态 |
| `actor_critic.py` | 因果 TCN、z 与三个估计头、固定 Actor 输入槽、监督损失 |
| `ppo.py` / `runner.py` | PPO、GAE、采样、超时 bootstrap、CSV 和检查点 |
| `train.py` / `evaluate.py` / `compare.py` | 训练、单模型评估、跨训练种子比较 |

TCN：输入投影到 64 通道，4 个残差因果卷积块，kernel=2，dilation=[1,2,4,8]，感受野 16 帧。最终时刻特征输出 z[16]、v[3]、contact_logits[2] 和 slip[2]。Actor 为 `[128,64,32]`，Critic 为 `[256,128,64]`。

Actor 输入为当前观测＋z＋7 个固定显式槽位。关闭的槽位置零且关闭对应损失，所有组参数量相同。PPO 通过 z 更新 TCN；显式输出送入 Actor 前 detach；各辅助损失也更新 TCN。Critic 仅接收固定的 143 维特权观测，不拼接 z。

| 组 | z | v：三维机身速度 | c：左右轮有效承载 | s：左右轮纵向滑移 |
|---|---|---|---|---|
| A0 | 开 | 关 | 关 | 关 |
| A1 | 开 | 开 | 关 | 关 |
| A2 | 开 | 开 | 开 | 关 |
| A3 | 开 | 开 | 关 | 开 |
| A4 | 开 | 开 | 开 | 开 |

## 训练命令

在仓库根目录执行，使用已安装 Isaac Gym、PyTorch 的训练环境。下面的命令在 PowerShell 和 Bash 中均可逐行运行。

### 一条命令跑完所有消融

```bash
python wheel_legged_gym/envs/q1_tcn/batch.py
```

默认顺序为 seed1 的 A0→A1→A2→A3→A4，再运行 seed2、seed3，共 15 次训练。每次训练为 1024 个环境、3000 轮，使用 `cuda:0`，训练完成后自动评估；全部完成后生成总对比表和图。各进程串行执行，上一项正常退出后才启动下一项。

默认输出为 `logs/q1_tcn/A0_seed1` 等目录，与下方单次训练命令一致。因此，手动训练的 `A1_seed1` 完成后可以直接运行批量入口，脚本会验证并复用它。**如果单次 A1 仍在运行，请等它退出后再启动批量入口，避免两个训练进程同时使用 GPU。**

只跑 seed1 的五组消融：

```bash
python wheel_legged_gym/envs/q1_tcn/batch.py --seeds 1
```

只训练、暂不评估：

```bash
python wheel_legged_gym/envs/q1_tcn/batch.py --train_only
```

先预览完整命令清单（不启动训练，不写入文件）：

```bash
python wheel_legged_gym/envs/q1_tcn/batch.py --dry_run
```

自定义预算和设备示例：

```bash
python wheel_legged_gym/envs/q1_tcn/batch.py --seeds 1 2 3 --num_envs 1024 --max_iterations 3000 --sim_device cuda:0 --rl_device cuda:0 --output_root logs/q1_tcn
```

批量入口中的 `max_iterations` 表示每组**总目标轮数**。重跑同一命令时，已达到目标的检查点跳过训练；未完成的模型从最近可读检查点恢复，仅补足剩余轮数。例如已有 model_500.pt 时，追加 2500 轮到 model_3000.pt。恢复产出存入该实验目录下的 `resume_500_to_3000/`，保留旧文件；训练失败且尚无检查点时，会使用新的 restart 子目录重新开始。

已完成且协议匹配、结果文件齐全的评估会跳过。中途失败立即停止队列，状态保存在 `logs/q1_tcn/ablation_batch.json`；解决问题后重跑同一命令即可继续。脚本不覆盖已有非空的训练或评估目录。环境数量、种子、组别或预算不兼容时会报错，不能把短测试目录直接作为正式实验继续使用。

默认结果位置：各实验目录下的 `evaluation/`（恢复后位于对应恢复目录下），以及 `logs/q1_tcn/comparison/comparison.csv`、`comparison.json`、`comparison.png`。先用 `--train_only` 跑完后，再去掉该选项重跑，就会补评估和总对比。

先运行一次短训练，检查设备、接触标签和输出文件：

```bash
python -m wheel_legged_gym.envs.q1_tcn.train --ablation A1 --seed 1 --num_envs 64 --max_iterations 10 --headless --sim_device cuda:0 --rl_device cuda:0 --output_dir logs/q1_tcn/smoke_A1
```

正式实验示例，每组保持相同的环境数和训练轮数：

```bash
python -m wheel_legged_gym.envs.q1_tcn.train --ablation A0 --seed 1 --num_envs 1024 --max_iterations 3000 --headless --sim_device cuda:0 --rl_device cuda:0 --output_dir logs/q1_tcn/A0_seed1
python -m wheel_legged_gym.envs.q1_tcn.train --ablation A1 --seed 1 --num_envs 1024 --max_iterations 3000 --headless --sim_device cuda:0 --rl_device cuda:0 --output_dir logs/q1_tcn/A1_seed1
python -m wheel_legged_gym.envs.q1_tcn.train --ablation A2 --seed 1 --num_envs 1024 --max_iterations 3000 --headless --sim_device cuda:0 --rl_device cuda:0 --output_dir logs/q1_tcn/A2_seed1
python -m wheel_legged_gym.envs.q1_tcn.train --ablation A3 --seed 1 --num_envs 1024 --max_iterations 3000 --headless --sim_device cuda:0 --rl_device cuda:0 --output_dir logs/q1_tcn/A3_seed1
python -m wheel_legged_gym.envs.q1_tcn.train --ablation A4 --seed 1 --num_envs 1024 --max_iterations 3000 --headless --sim_device cuda:0 --rl_device cuda:0 --output_dir logs/q1_tcn/A4_seed1
```

正式实验对 seed=2、3 重复上面的五组命令，同时修改输出目录。每组默认预算为 `1024 × 48 × 3000 = 147456000` 个环境交互步。

训练输出：`training.csv`、`resolved_config.json`、`model_100.pt` 等检查点、`q1.urdf`、`source/` 源码快照。CSV 包含环境原始 reward、跌倒/超时数、策略/价值损失、各估计损失、熵、KL 和学习率。检查点自带完整配置，评估按该配置建立网络，不需要另外填写 ablation。

继续训练示例（配置须与原训练一致，`max_iterations` 表示本次追加轮数）：

```bash
python -m wheel_legged_gym.envs.q1_tcn.train --ablation A1 --seed 1 --num_envs 1024 --max_iterations 3000 --resume_checkpoint logs/q1_tcn/A1_seed1/model_3000.pt --headless --sim_device cuda:0 --rl_device cuda:0 --output_dir logs/q1_tcn/A1_seed1_continued
```

恢复模型、优化器、学习率、计数器和随机数状态，但仿真从新回合开始，并非物理轨迹的逐步恢复。不同训练预算的模型不要直接放进同一正式消融表。

## 标签、损失与时间对齐

每个观测时刻的标签固定为 10 维：`[v_body(3), contact(2), slip(2), frame_valid(1), slip_valid(2)]`。采样器在执行动作前克隆历史和标签；仿真重置后的接触张量尚未刷新，对应帧的监督有效位为零。历史重置时以同一个首帧重复填充。噪声、裁剪先于历史入队。

- 速度：机身坐标系 m/s，除以 `velocity_scale`，使用 Smooth L1。
- 接触：平地世界 Z 法向力，5 N 开、3 N 关，连续 2 个策略帧确认状态切换，使用 BCEWithLogits。0.5 概率阈值用于评估。
- 滑移：几何轮底点速度沿机身水平前向投影，带符号，单位 m/s，除以 `slip_scale`，使用 Masked Smooth L1。考虑轮子绝对角速度、圆柱碰撞中心和 link COM 偏移；不是 `r*qdot-v_base`。即时法向力 >3 N 才有滑移监督，避免接触迟滞对离地帧产生错误标签。
- 三项回归/分类损失在各自有效元素上取平均，再按配置权重求和。默认速度/滑移归一化尺度均为 1 m/s，辅助权重均为 1。

此几何标签仅针对 Q1 圆柱轮与平地，不等同于 PhysX 输出的真实接触位置，不用于台阶或复杂地形。接触标签表达承载状态，不表达轮侧碰撞。训练默认存在轻量本体噪声、摩擦随机化 [0.35,1.0]、机身附加质量 [-0.2,0.5] kg、COM 随机化和有限推力；配置对所有组相同。

推力为世界水平 XY 方向、作用于 trunk COM 的外力，每轴随机 [-5,5] N，持续 0.15 s，每 5 s 一次，首次约在 5 s。每个物理步都显式施加或清零力，没有瞬间修改根速度。

## 评估命令

完整评估：7 类场景 × 3 个评估种子 × 每种子 64 个机器人，每个试验最长 12 s。

```bash
python -m wheel_legged_gym.envs.q1_tcn.evaluate --checkpoint_path logs/q1_tcn/A1_seed1/model_3000.pt --num_envs 64 --eval_seeds 101 102 103 --duration 12 --headless --sim_device cuda:0 --rl_device cuda:0 --output_dir logs/q1_tcn/A1_seed1/evaluation
```

只评估站立、跟踪和前向推力：

```bash
python -m wheel_legged_gym.envs.q1_tcn.evaluate --checkpoint_path logs/q1_tcn/A1_seed1/model_3000.pt --scenarios stand tracking push_front --num_envs 64 --eval_seeds 101 --headless --sim_device cuda:0 --rl_device cuda:0 --output_dir logs/q1_tcn/A1_seed1/quick_evaluation
```

更强推力测试使用 `--push_force 12`；所有参与比较的模型须使用相同值。`--duration` 最小 7 s，完整启停序列应使用默认的 12 s。检查点应来自自己可信的训练输出。

| 场景 | 固定条件 |
|---|---|
| stand | 零速度指令，名义摩擦 0.8 |
| tracking | 0–2 s 停；2–5 s +0.25 m/s；5–7 s 停；7–10 s -0.25 m/s；随后停 |
| push_front / push_back / push_side | 零速度指令，3.0–3.15 s 分别沿世界 +X / -X / +Y 施加 8 N |
| low_friction | 相同跟踪序列，摩擦 0.2 |
| payload | 相同跟踪序列，trunk 增加 0.75 kg，trunk COM 沿 X 偏置 +0.02 m |

评估保留训练的观测噪声，关闭训练随机推扰及参数随机化，使用明确的场景参数、固定评估种子和 Actor 均值动作。payload 中质量与 COM 偏置作为一组负载压力测试，不能用于分别归因这两个因素。

输出：

- `summary.json` / `summary.csv`：各场景各指标的均值、标准差和有效样本数，包含训练与评估协议。
- `trials.csv`：每个机器人第一回合的统计，包含 seed、trial、是否跌倒、有效观测时长。
- `<scenario>_seed<seed>_trace.csv`：每时刻仍存活试验的均值响应、速度标准差、样本数。
- `<scenario>.png`：速度与指令、Pitch、Roll、接触滑移、轮支撑力及存活试验数曲线。

关键指标：速度 RMSE，Pitch/Roll RMS 与峰值，位移，支撑期间滑移 MAE、高滑移占比、异常空转时间，执行器饱和比例；启停后速度进入 ±0.05 m/s 并持续 0.5 s 的稳定时间与成功比例；推后恢复比例和恢复时间；速度估计 RMSE、接触 F1/负类召回/状态切换检测延迟、滑移估计 RMSE。

推后恢复要求在推力结束后 3 s 内，速度误差 <0.05 m/s、Pitch/Roll 绝对值 <0.12 rad 持续 0.5 s，且整个试验之后未跌倒。`recovered` 以全部试验为分母，同时提供 `push_exposed` 以识别推前已跌倒的试验。该指标衡量回到稳定带，不要求推力一定造成离开稳定带。

高滑移阈值为 0.2 m/s；异常空转定义为有效承载下至少一个轮子高滑移且机身纵向速度绝对值 <0.1 m/s。接触检测延迟相对迟滞后的标签状态切换计算，仅对检测到的切换平均，同时报告切换检测比例。没有负类时负类召回为空，没有有效估计头/接触样本时相应指标为空；不能把空值当成零误差。

**先看跌倒比例和观测时长，再看误差。** 跌倒前的最后状态参与统计，自动重置后的状态不重新计入试验。误差统计覆盖到跌倒或规定时长，提前跌倒的模型可能有较低 RMSE。图中的响应均值随存活试验集合变化，要同时看存活数。位移在零指令场景解释为停车漂移，在跟踪场景不应如此解释。

## A0～A4 汇总图

先为每个正式模型运行相同评估，再汇总；比较工具不需要 Isaac Gym，可直接按文件路径运行。

```bash
python wheel_legged_gym/envs/q1_tcn/compare.py --inputs logs/q1_tcn/A0_seed1/evaluation/summary.json logs/q1_tcn/A1_seed1/evaluation/summary.json logs/q1_tcn/A2_seed1/evaluation/summary.json logs/q1_tcn/A3_seed1/evaluation/summary.json logs/q1_tcn/A4_seed1/evaluation/summary.json --output_dir logs/q1_tcn/comparison
```

把 seed2、seed3 的 summary.json 也加入 `--inputs`，输出 `comparison.csv`、`comparison.json` 和 `comparison.png`。先在每个训练种子的评估试验上求均值，再计算训练种子之间的均值/标准差。单个训练种子不显示不确定性误差棒。工具拒绝预算、环境条件、网络配置、评估协议不一致或同组同训练种子重复的输入。

## 验证

```bash
python -m unittest discover -s tests -p test_q1_tcn.py -v
```

测试覆盖因果性、参数量与开关、梯度路径、标签与重置对齐、推力清零、GAE 边界、五组实际 PPO 更新、评估统计和配置恢复。CPU 张量仿真替身不验证真实接触或能否平衡；正式批量训练前先完成短训练及短评估。
