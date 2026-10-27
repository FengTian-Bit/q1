# Q1 接触触发盲上楼梯任务

`q1_ctbc` 是独立任务，沿用 Q1 六关节机器人与本体观测，在训练中增加接触触发、单腿摆动指导、前馈退火和方向明确的上台阶课程。保留 `q1`、`q1_flat` 作原任务对照。本实现是依据 [CTBC 论文 v3](https://arxiv.org/html/2509.02986v3) 为 Q1 编写的适配，作者官网目前没有公开训练代码和权重，不能视为作者实现的逐参数复制。

论文的八自由度 Tron1 与 Q1 的六自由度硬件不同。这里先验收 2.5–8 cm 单阶和 4–12 cm 连续三阶；20 cm 不作为已具备的能力。初始接触阈值、动作幅值、奖励权重和退火时间均是 Q1 的起始配置，需由真实接触仿真结果调整。

## 启动与课程

需要已安装 Isaac Gym Preview 4、匹配的 CUDA/PyTorch 与项目依赖。从项目根目录运行：

```bash
pip install -e .
python wheel_legged_gym/scripts/train.py --task=q1_ctbc --ctbc_stage=single --headless
```

默认 `single`：20% 平地、80% 单阶，单级高度 0.025–0.08 m，踏面 0.5 m，正向速度 0.15–0.3 m/s，不采样转向/后退。出生在第一个台阶前的低处，课程升降依据实际到达目标平台的结果，避免在出生平台上运动就升级。

每个楼梯环境累计 5 个有效回合后判定课程：成功率 ≥80% 升一级，<30% 降一级，其余保持；判定后清零该组计数。平地回合不用于楼梯升级。

先用固定 5 cm 单阶评估指导开启与纯策略表现；具备稳定落台能力后再续训连续楼梯：

```bash
# RUN_NAME 为 logs/q1_ctbc 下原有运行目录，CHECKPOINT 为实际模型编号。
python wheel_legged_gym/scripts/train.py --task=q1_ctbc --ctbc_stage=stairs --resume --load_run=RUN_NAME --checkpoint=CHECKPOINT --headless
python wheel_legged_gym/scripts/train.py --task=q1_ctbc --ctbc_stage=robust --resume --load_run=RUN_NAME --checkpoint=CHECKPOINT --headless
```

`stairs` 为三阶楼梯，单级高度 0.04–0.12 m，踏面随课程从 0.5 m 减至 0.35 m。`robust` 保留这一几何范围，启用配置中较保守的摩擦、质量、重心、驱动、延迟、扰动及观测噪声随机化；惯量随机化保持关闭。`single`、`stairs` 关闭这些随机化，先解决抬腿及承重转换。阶段参数在 `q1_ctbc_config.py` 的 `configure_ctbc_stage` 中。

续训来源应为 `q1_ctbc` 本任务检查点。虽然本体观测维数沿用 Q1，腿动作尺度、训练目标和动作历史语义已改变，不把 `q1` 模型直接作为完整训练状态恢复。旧平地策略若需迁移，应另做有记录的模型权重初始化实验。

续训会在创建环境前读取该检查点旁的 `resolved_config.json`，恢复保存的环境、网络、算法和退火配置，并固定解析出的运行目录和 `model_N.pt`，避免后续再次选择“最新”模型。不传 `--ctbc_stage` 时完整沿用保存配置；显式指定阶段时，无论是否与保存阶段相同，都会重新应用当前 `configure_ctbc_stage()` 中的课程、指令和随机化设置。因此修改楼梯高度后，同阶段续训也要显式传入 `--ctbc_stage=stairs`。阶段函数未设置的动作尺度、奖励、网络和退火参数仍沿用检查点配置，并非重新加载全部 Python 默认值。`--num_envs`、`--max_iterations`、`--seed`、`--logger` 等命令行覆盖仍优先于保存值。新运行默认名称为最终阶段，可用 `--run_name` 指定。

快照必须包含 CTBC 配置及完整接口：Actor 31 维、critic 特权观测 151 维、5 帧历史、6 维动作、155 维历史编码输入和四个有限正数的腿动作尺度。缺少快照、误选 `q1` 快照或维度不匹配时明确报错。自定义腿动作尺度和退火时间从原快照保留，不退回代码默认值。

`max_iterations` 是本次追加的更新轮数。前馈退火按检查点保存和恢复的**累计迭代数**计算，切换 `single → stairs → robust` 不会自动重启指导。默认前 3000 轮保留初始前馈，随后 12000 轮线性减至零；不要把指导开启的回报当作纯策略成功率。阶段切换时若还需要指导，应在退火结束前切换，或明确修改退火课程并记录配置。

## 训练机制与动作语义

- 对左右轮的水平接触力使用连续三帧阈值确认；单侧受阻优先该腿，双侧受阻优先稳定接触/较大力的一侧。竖直承重接触用于落地和成功判断，与水平碰撞触发区分。
- 单腿摆动采用 0.6 s 余弦脉冲，髋与膝方向依 Q1 关节轴适配，另一腿保留支撑。按摆动相位启用目标姿态、轮子间隙和接触匹配奖励。
- 左右腿对称约束 `leg_symmetry` 默认权重为 -0.10，平地和台阶均启用。复用 `ctbc.contact_threshold`（默认 12 N）：任一轮的水平接触力模长超过阈值时，立即将该环境的对称惩罚置零；两轮都不超过阈值时立即恢复。竖直承重力不参与此开关，也不等待连续三帧确认或摆动结束；关闭指导后仍按受力控制奖励。续训旧快照时，需将 `env.rewards.rough_leg_symmetry_weight` 从 0.0 改为 1.0，并确认 `env.rewards.scales.leg_symmetry` 为 -0.10，否则保存配置会覆盖新默认值。
- PPO 存储和计算概率比的 action 是网络原始输出。环境另存**实际执行的动作**，用于控制、观测中的上一动作、动作变化代价和诊断；前馈不能写回修改 PPO 的原始动作张量。
- Actor 不读取仿真接触力、地形高度、成功标记或触发状态；这些信息用于训练指导、奖励、critic 与评估。纯策略验收调用 `env.set_guidance_mode("off")`，前馈权重为零，并停止指导 FSM 对动作的作用。`"on"` 使用初始前馈系数，仅用于诊断；它仍尊重配置中的 `feedforward_enabled=False` 消融开关。`"annealed"` 按累计训练轮数计算权重。

前馈在这里是训练辅助。关掉指导后仍能过台阶，才说明网络学会了接触后的动作。论文对实机接触估计器与 FSM 的具体接口披露不足，本任务以无需仿真特权输入的纯策略作为正式验收条件。

`feedforward_enabled=False` 可用于保留触发/奖励、撤去显式前馈的诊断。`trigger_enabled=False` 会停止本实现的事件触发指导；这里没有实现周期抬腿前馈，因此该开关**不等同于论文中保留周期前馈的“去接触触发”消融**。若做论文式四组对照，还需单独定义和实现周期前馈基线，不能仅用此开关给结果贴上同名标签。

## 固定几何批量评估

```bash
# --heights 的单位是厘米；--tread 的单位是米。
python wheel_legged_gym/scripts/evaluate_q1_ctbc.py --task=q1_ctbc --load_run=RUN_NAME --checkpoint=CHECKPOINT --headless --num_envs=64 --heights=2.5,5,8,10,12 --tread=0.5 --steps=1 --episodes=100 --speed=0.2 --guidance=both --seed=1

# 三阶连续楼梯正式纯策略评估。
python wheel_legged_gym/scripts/evaluate_q1_ctbc.py --task=q1_ctbc --load_run=RUN_NAME --checkpoint=CHECKPOINT --headless --num_envs=64 --heights=4,6,8,10,12 --tread=0.35 --steps=3 --episodes=100 --speed=0.2 --guidance=off --seed=1
```

每个高度/指导模式单独建立固定地形，关闭地形课程，确定性执行策略均值，以固定正向速度、零转向接近台阶。默认 `guidance=off`；`both` 分别报告纯策略与指导开启的结果。物理和观测随机化沿用检查点保存的配置，相同 seed 用于配对比较；JSON 中记录具体随机化设置。

评估从检查点目录的 `resolved_config.json` 验证 CTBC 接口后恢复网络和环境配置，加载模型时不恢复优化器，也不启动训练日志或云端上传。正式结果读取 `step` 返回的 `infos["ctbc_episode"]` 中**重置前**的成功/失败快照；不读取已被重置清零的 `env.ctbc_success`。

成功要求连续 0.15 s 同时满足：**双轮中心的 X 坐标均到达目标线，双轮几何底部距最终平台高度均不超过 ±0.025 m，双轮各自竖直接触力均大于 5 N**。默认目标线在最后一级立面前方 0.3 m。实际倒地或越出规定地块均否决成功，不以机身越过边缘或暂时腾空代替落台。

每个条件精确统计 `episodes` 个已完成回合，最后一批按环境编号截断。JSON 保存成功数、物理失败数、超时数、其他终止数、成功率、checkpoint、seed、指导模式、网格量化后的实际高度/踏面及步数。默认保存在检查点旁的 `evaluations/ctbc_时间戳.json`；可传 `--output=路径.json`。`--eval_max_steps` 设置每个条件的策略步数上限，默认按回合长度推导；不足指定回合数时输出 `complete=false`，独立评估入口以非零状态结束，不能把部分样本当成完整验收。

普通播放入口也支持本任务：

```bash
python wheel_legged_gym/scripts/play.py --task=q1_ctbc --load_run=RUN_NAME --checkpoint=CHECKPOINT --num_envs=1
```

该入口自动测试 5 cm 单阶、0.5 m 踏面、0.2 m/s、纯策略模式，默认累计 100 个回合；有 viewer 时可观察，不提供 Q1 键盘遥控。需要指定楼梯参数、样本数或比较指导时使用独立评估脚本；不带 `--headless` 即可显示 viewer。

## 验证范围

CPU 回归覆盖接触/相位、前馈退火与恢复、动作历史、方向楼梯与成功课程，以及 reset 后的评估统计。它们检查实现逻辑，不模拟真实轮子接触，也不证明已具备上楼梯能力。当前尚未在 Isaac Gym 中完成实际训练或成功率评估；训练曲线、单阶落台、连续三阶和关闭指导后的表现仍需在支持 Isaac Gym 的 GPU 主机上运行上述命令确认。
