# Q1 接触触发盲上楼梯任务

`q1_ctbc` 是独立任务，沿用 Q1 六关节机器人与本体观测，在训练中增加接触触发、单腿摆动指导、前馈退火和方向明确的上台阶课程。`q1` 现基于本控制器采用四轮足 T0 混合地形与距离课程，见 [q1 说明](../q1/README.md)；`q1_flat` 保留平地基线。本实现是依据 [CTBC 论文 v3](https://arxiv.org/html/2509.02986v3) 为 Q1 编写的适配，作者官网目前没有公开训练代码和权重，不能视为作者实现的逐参数复制。

论文的八自由度 Tron1 与 Q1 的六自由度硬件不同。保留 2.5–8 cm 单阶和 4–12 cm 连续三阶基准；`stairs20` 现提供 12 级、4–22 cm 连续三阶课程，旧阶段名称保留用于命令兼容。用户已反馈旧版 8 cm 引导能够通过 10 cm 实体楼梯；新课程尚未验证实际 22 cm 成功率。接触阈值、动作幅值、奖励权重和退火时间仍需由接触仿真和实机结果调整。

## 启动与课程

需要已安装 Isaac Gym Preview 4、匹配的 CUDA/PyTorch 与项目依赖。从项目根目录运行：

```bash
pip install -e .
python wheel_legged_gym/scripts/train.py --task=q1_ctbc --headless
```

默认对齐桌面 `log/resolved_config.json` 保存的训练：`stairs` 阶段、1024 个环境、30000 轮、种子 1，从头训练。20% 平地、80% 连续三阶，单级高度 0.04–0.12 m，踏面随课程从 0.5 m 减至 0.35 m，正向速度 0.15–0.3 m/s，不采样转向/后退，关闭域随机化与观测噪声。出生在第一个台阶前的低处，课程升降依据实际到达目标平台的结果。完整比较见 [训练对齐记录](../../../docs/q1_ctbc_training_alignment.md)。

如需单阶课程，显式传入 `--ctbc_stage=single`：20% 平地、80% 单阶，单级高度 0.025–0.08 m、踏面 0.5 m。`q1_ctbc_highcom` 保留原有的 `single` 默认阶段和 4096 环境。

普通阶段每个楼梯环境累计 5 个有效回合后判定课程；新的 `stairs20` 改为 10 个。成功率 ≥80% 升一级，<30% 降一级，其余保持；判定后清零该组计数。平地回合不用于楼梯升级。

选择单阶课程时，可先用固定 5 cm 单阶评估指导开启与纯策略表现，再续训连续楼梯。默认三阶运行也可从检查点继续训练或切换到随机化阶段：

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

## 4–22 cm 配套课程

`stairs20` 的新预设以 `ctbc.guidance_profile=dense22_v1` 标识，将原六级课程加密为十二级，单级高度范围设为 4–22 cm。普通 `stairs` 仍是 4–12 cm 基线。`q1` 的 T0 配置和 `q1_ctbc_highcom` 的原高阶段参数保持独立，后者仍为 10–20 cm、8–16 cm 引导。

新预设保持 Actor/critic/历史维度和 `[0.5, 0.9, 0.5, 0.9]` 腿动作归一化尺度，从旧 CTBC 检查点恢复网络与优化器；原 `[-1,1]` 范围内动作对应的角度不变。原来被 ±1 裁掉的腿动作尾部会在新阶段开放，所以仍需观察旧策略进入高阶段时的饱和与姿态变化，前馈也采用渐入适应。

| 参数 | `stairs20` 设置 |
|---|---|
| 连续台阶 | 每块地形连续三阶；12 行经 5 mm 网格量化后的单级高度为 4/5.5/7/8.5/10.5/12/13.5/15/17/18.5/20/22 cm |
| 踏面与速度 | 踏面固定 0.50 m，前进速度 0.15–0.20 m/s，初期关闭域随机化；保留 20% 平地与 80% 楼梯 |
| 升降级 | 从第 0 级开始；每 10 个有效楼梯回合判定，成功至少 8 次升一级，少于 3 次降一级，其他保持；等级限制为 0–11 |
| 抬轮参考 | `clamp(0.8 × 实际单级高度, 8 cm, 17.6 cm)`，IK 保持名义轮轴前后位置；Actor 不接收高度或相位 |
| 腿动作范围 | 髋目标偏移 ±0.70 rad、膝目标偏移 ±1.35 rad；归一化腿通道限幅 ±1.4/±1.5，轮通道仍为 ±1 |
| 时间 | 0.85 s 余弦脉冲，至少 0.4 s 且观察到卸载后重新承重才结束；落地还需较触发时上升至少 2 cm |
| 前馈渐入 | 默认用 1000 轮把权重从 0 增至 1，给已有抬腿策略适应叠加引导的时间；渐入包含在 3000 轮保持期内 |
| 指导奖励 | 关节轨迹跟踪权重 0.3 并随前馈退火；轮底比参考高 0–6 cm 不扣净空匹配奖励，超过这一带宽仍惩罚 |

8–17.6 cm 的参考由 `ctbc.guide_lift_height_range` 和 `ctbc.guide_lift_ratio=0.8` 控制，高阶段使用 IK 计算每个环境的幅值；原 `hip_amplitude/knee_amplitude` 仍用于旧阶段。对应十二级的参考为 8/8/8/8/8.4/9.6/10.8/12/13.6/14.8/16/17.6 cm。最大参考的髋/膝增量约为 `+0.57522/-1.15045 rad`，目标角约为 `1.17522/-2.35045 rad`，在软件关节限位内。0.85 s 脉冲的膝峰速度约 4.25 rad/s、峰加速度约 31.43 rad/s²，与旧 8 cm/0.6 s 参考的 32.38 rad/s² 接近。腿力矩仍受原 14 N·m 限制，轮力矩保持 0.8 N·m；速度与几何计算不能证明动态跟踪成功。

17.6 cm 参考给膝目标下限 -2.511 rad 留出约 0.161 rad。固定机身、轮轴前后不变的完整 22 cm 缩腿目标会超出现有软件目标范围，因此需要策略学习支撑腿伸长和机身上升的配合。8/10 的比例依据已有试验反馈，17.6/22 仍是待验证的训练假设。增大参考时，超过动作范围或软件关节裕量会报错，避免静默裁剪成较低轨迹。

```bash
# 从头训练新的十二级课程。
python wheel_legged_gym/scripts/train.py --task=q1_ctbc --ctbc_stage=stairs20 --num_envs=1024 --run_name=stairs22_dense --headless

# RUN_NAME 使用已经能通过低台阶的 q1_ctbc 运行；CHECKPOINT 为验证过的模型编号。
python wheel_legged_gym/scripts/train.py --task=q1_ctbc --ctbc_stage=stairs20 --resume --load_run=RUN_NAME --checkpoint=CHECKPOINT --run_name=stairs22_dense --max_iterations=30000 --headless

# 比较指导开启与纯策略；高度单位为厘米。
python wheel_legged_gym/scripts/evaluate_q1_ctbc.py --task=q1_ctbc --load_run=HIGH_RUN_NAME --checkpoint=-1 --headless --num_envs=64 --heights=4,5.5,7,8.5,10.5,12,13.5,15,17,18.5,20,22 --tread=0.5 --steps=3 --episodes=100 --speed=0.2 --guidance=both --seed=1
```

首次从普通阶段进入 `stairs20`，或显式将旧版 `stairs20` 迁移到 `dense22_v1` 时，会读取检查点真实 `iter`，保存为 `ctbc.feedforward_start_iteration`，从该点重新计时。默认前 1000 轮渐入、到第 3000 轮维持全强度，再用 12000 轮退火。例如从累计 30000 轮进入，权重在 30000–31000 轮从 0 增至 1，33000 轮开始下降、45000 轮归零；后续同一 `dense22_v1` 配置中断再续训保留起点，不重复重启。`feedforward_ramp_iterations` 控制渐入时间（不超过保持期），诊断 `--guidance=on` 始终使用全强度，不经过渐入。不传 `--ctbc_stage` 时保留快照的原地形、引导和起点，包括旧版六级 10–20 cm 配置；要启用本次修改必须显式指定 `--ctbc_stage=stairs20`。显式退回普通阶段则恢复原累计轮数退火语义。

部署高阶段纯策略时必须同步腿通道的扩展限幅。若下位机仍把所有动作裁剪为 ±1，就会再次丢失高抬腿能力；保持原归一化尺度、轮通道 ±1、软件关节裕量、速度与力矩限制。训练日志的 `ctbc_guide_lift_m` 记录参考高度，`ctbc_feedforward_weight` 记录当前指导权重。

新课程默认开启诊断，除原指标外还记录：

- `Episode/ctbc_stair_success_h040mm` 等按实际单级高度分组的成功率，以及 `Episode/ctbc_stair_level`。这些统计采用重置前的数据；第 0 级对应 4 cm，不是平地。
- `Episode/ctbc_stall_rate` 和各高度的 `ctbc_stall_hXXXmm`：一次楼梯回合中是否曾在接触障碍后，连续 3 s 未向前累计推进 3 cm、也未增加 2 cm 的有效支撑高度。悬空抬轮、相同高度的重复落轮不算高度进展；平地和静止命令不参与检测。当前卡阶状态可在恢复进展后清除，回合记录仍保留曾经卡阶的标记。
- 六个关节各自的 `Episode/ctbc_policy_clip_<关节名>` 与 `Episode/ctbc_composite_clip_<关节名>`：原始策略输出超过动作限幅的时间占比，以及裁剪原始输出后叠加引导时再次超限的时间占比。每个回合先计算比例，再对完成的回合取均值。
- `Policy/noise_std_<关节名>`：逐关节策略探索标准差，便于区分腿与轮的探索变化。

按高度的成功率和卡阶率按完成的回合数加权，没有该高度完成回合时不记为零。卡阶检测本次**只记录**，不终止回合、不增加惩罚、不进入 Actor 观测；需先在真实接触仿真中验证误判率。新增诊断字段随 `resolved_config.json` 保存。

## 训练机制与动作语义

- 对左右轮的水平接触力使用连续三帧阈值确认；单侧受阻优先该腿，双侧受阻优先稳定接触/较大力的一侧。竖直承重接触用于落地和成功判断，与水平碰撞触发区分。
- 普通阶段单腿摆动采用 0.6 s 余弦脉冲，新 `stairs20` 为 0.85 s；髋与膝方向依 Q1 关节轴适配，另一腿保留支撑。按摆动相位启用目标姿态、轮子间隙和接触匹配奖励。
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

播放时默认记录第 0 台机器人的六个关节力矩。点击 viewer 的关闭按钮、按 Esc / Ctrl+C 或正常结束后，会自动生成 `evaluations/ctbc_时间戳_h5cm_policy_torques.png`，并在终端打印 `Q1 torque PNG:` 及完整路径；对应 JSON 的 `torque_plots` 字段也会保存图片路径。图中左右两列分别是左右腿，三行分别是大腿、小腿和轮子，横轴为累计仿真时间（秒），纵轴为施加到仿真器的电机力矩（N·m），红色虚线为正负力矩限值。每个物理步记录一次，并在回合重置清零前保存该步数据。图片直接写盘，不弹出绘图窗口。尚未完成任何物理步就关闭，或使用 `--export_only` 时，不生成无数据的力矩图。独立批量评估入口默认不记录力矩。

`play.py --task=q1_ctbc` 在加载检查点后、评估循环前自动导出 ONNX，并打印 `Exported Q1 CTBC policy as ONNX:` 和文件绝对路径。文件位于该运行目录的 `exported/model_模型编号/policy.onnx`，评估中按 Ctrl+C 不影响已经完成的导出。独立评估脚本默认只评估，可加 `--export_onnx` 同时导出。

只需导出时，可以跳过评估回合并在导出后退出（仍需初始化 Isaac Gym 环境）：

```bash
python wheel_legged_gym/scripts/play.py --task=q1_ctbc --load_run=RUN_NAME --checkpoint=-1 --num_envs=1 --headless --export_only
```

导出包含历史编码器和 actor，输入为 `obs: [batch, 31]` 与 `obs_history: [batch, 155]`，输出 `actions: [batch, 6]`，支持动态 batch。历史是按时间从旧到新排列的 5 帧已归一化观测，最新一帧为当前观测；重置时以首帧重复填充。动作顺序为左大腿、左小腿、左轮、右大腿、右小腿、右轮，输出为策略原始动作。观测缩放、动作裁剪、腿/轮动作尺度、前馈和 PD 控制仍需按该检查点的 `resolved_config.json` 与环境代码实现；ONNX 不包含 CTBC 指导或控制器。导出依赖 `onnx`（项目安装依赖已列出），缺失时运行 `python -m pip install onnx`。`--export_only` 的 JSON 标记 `export_only=true`，其中 `complete` 表示导出完成，`results` 为空，不表示完成了爬阶评估。

## 验证范围

CPU 回归覆盖接触/相位、前馈退火与配置迁移、动作历史、方向楼梯与成功课程，以及 reset 后的评估统计；新课程还检查十二级实际几何、17.6 cm 参考未被裁剪、旧动作映射保持一致、十回合升降级、卡阶检测和按高度统计。固定机身的 22 cm 目标超限也有回归保护。这些测试不模拟真实轮子接触，不证明新课程的动力学或实机能力。本次修改尚未在 Isaac Gym 中完成实际训练或 22 cm 成功率评估；请分别验收指导开启和关闭后的表现。
