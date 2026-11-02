# Q1 CTBC 高站姿对照任务

`q1_ctbc_highcom` 继承 `q1_ctbc` 的定向爬阶、接触指导、奖励、PPO 和四个训练阶段，修改名义站姿及其运动学依赖。日志独立存入 `logs/q1_ctbc_highcom/`。本任务保留 `single` 阶段、4096 环境的默认设置；`q1_ctbc` 已对齐保存的基线运行，默认 `stairs`、1024 环境。

本任务的 `stairs20` 仍保留六级 10–20 cm、8–16 cm 引导和 0.8 s 摆动；`q1_ctbc` 新增的十二级 4–22 cm 配置不应用到本对照任务。

## 选择的姿态

采用用户参考图的中等弯曲姿态，左右相同：

- 大腿 `+0.3746 rad`，小腿 `−0.7854 rad`，轮角 `0`。
- 水平地面接触时，`trunk` 原点高度 **0.43601504 m**；它不是整机质心高度。
- 全 URDF 质量加权质心高度约 **0.32467588 m**，前后位置与轮轴相差约 **0.0115 mm**，假设机身 pitch 为零。
- 重置根坐标 Z 为 **0.44135504 m**，保留原任务约 **5.34 mm** 轮底离地量。
- 高度指令与高度奖励目标均同步到 **0.43601504 m**。

实际 URDF 前向运动学比较如下，均假设 pitch=0、轮子接触水平地面：

| 姿态 | 大腿 / 小腿 rad | 机身原点高度 m | 整机质心高度 m | 距腿完全竖直伸展的高度余量 |
|---|---|---|---|---|
| 原任务 | 0.6000 / −1.2000 | 0.39666 | 0.29774 | 6.99 cm |
| 较直 | 0.2314 / −0.5236 | 0.45272 | 0.33616 | 1.38 cm |
| **选用中等弯曲** | **0.3746 / −0.7854** | **0.43602** | **0.32468** | **3.05 cm** |
| 较深 | 0.5182 / −1.0472 | 0.41293 | 0.30888 | 5.36 cm |

中等姿态比原任务高约 3.94 cm，同时避免较直姿态只剩约 1.4 cm 的伸腿余量。这是几何折中，不代表已验证其动态性能最优。略偏离“大腿等于负半膝角”的取值用于按整机质量分布对齐质心与轮轴。

## 配套控制适配

所有阶段都保留原来的物理抬轮参考：旧阶段约 8 cm，`stairs20` 为每级台阶高度的 0.8 倍并限制在 8–16 cm。新站姿重新通过 IK 求角度幅值，不能直接使用旧 `+0.2952/−0.5905 rad` 脉冲，否则物理抬轮高度会改变。

在 `stairs20` 中，固定机身、保持名义轮轴 X，20 cm 抬轮的目标约为 `[1.09343, −2.26575] rad`，相对新默认角度的增量约 `[+0.71883, −1.48035] rad`。原高阶段的 `±0.70/±1.35 rad` 偏移限额会截断它，因此新任务的高阶段使用 `±0.80/±1.55 rad`。仍遵守 URDF 的 0.97 软限位与 0.05 rad 目标裕量。

腿动作归一化尺度继续为 `[0.5, 0.9, 0.5, 0.9]`，`stairs20` 的归一化限幅随之变为大腿 ±1.6、小腿 ±1.72222；轮通道仍为 ±1。单阶/普通连续阶阶段保留原 ±1 动作范围。PD 增益、100 Hz 策略频率、腿 **14 N·m**、轮 **0.8 N·m** 上限保持原设置，没有增大电机力矩。

## 训练、恢复与评估

在已安装 Isaac Gym 的 Linux 训练环境、项目根目录执行：

```bash
# 从头训练较高站姿，默认 2.5–8 cm 单阶。
python wheel_legged_gym/scripts/train.py --task=q1_ctbc_highcom --headless

# 使用本任务的单阶策略进入 10–20 cm 连续三级楼梯。
# HIGHCOM_RUN 替换为 logs/q1_ctbc_highcom 中的实际运行目录。
python wheel_legged_gym/scripts/train.py --task=q1_ctbc_highcom \
  --ctbc_stage=stairs20 --resume --load_run=HIGHCOM_RUN --checkpoint=-1 --headless

# 同任务续训：完整恢复保存的姿态、控制、地形、网络和指导时钟。
python wheel_legged_gym/scripts/train.py --task=q1_ctbc_highcom \
  --resume --load_run=HIGHCOM_RUN --checkpoint=-1 --headless

# 高台阶评估；高度单位是厘米，分别比较纯策略和指导开启。
python wheel_legged_gym/scripts/evaluate_q1_ctbc.py --task=q1_ctbc_highcom \
  --load_run=HIGHCOM_RUN --checkpoint=-1 --headless --num_envs=64 \
  --heights=10,12,14,16,18,20 --tread=0.4 --steps=3 --guidance=both

# 与原 CTBC 一样，播放默认评估 5 cm 单阶并导出 ONNX、记录六个电机力矩。
python wheel_legged_gym/scripts/play.py --task=q1_ctbc_highcom \
  --load_run=HIGHCOM_RUN --checkpoint=-1 --num_envs=1
```

`--ctbc_stage=single/stairs/robust/stairs20` 均可使用。显式指定阶段时重新应用新任务对应的姿态和阶段预设；省略时保留快照中的调参。新任务与原 `q1_ctbc` 改变了动作零点和观测位置误差基准，**拒绝跨任务 `--resume`**，即使网络维度相同也不直接恢复。进入高阶段的指导渐入/退火仍按检查点真实累计迭代重新计时，同阶段续训不重启。

ONNX 的单帧/历史输入仍为 31/155 维，输出六维；部署必须读取本任务的保存配置，复现新默认角、动作限幅、高度和执行动作历史。

## 验证范围

静态检查报告见 [高站姿模型报告](../../../docs/q1_ctbc_highcom_posture_check.json)。复查命令：

```bash
python tools/check_q1_model.py --thigh-angle=0.3746 --calf-angle=-0.7854 \
  --initial-height=0.4413550396778036 --output=docs/q1_ctbc_highcom_posture_check.json
python -m unittest discover -s tests -p 'test_q1_ctbc*.py' -v
```

全 URDF 检查覆盖关节树、正轴、轮限位、质量与惯量有效性/数量级筛查、STL 资源和新出生姿态下所有简单碰撞体的地面间隙。模型有 9 个 link、6 个活动关节、2 个固定关节、13 个碰撞体，总质量约 6.87 kg；没有 mesh collision 待替换项。URDF 动力学和几何均未修改。惯量精度受缺少 CAD 质量分布的限制，完整自碰撞分离未认证。

CPU 回归覆盖各阶段的真实运动学、指导净空、软限位、动作裁剪、原始 PPO 动作保护、力矩上限、任务来源验证、训练快照与恢复接口。张量 Gym 桩不模拟接触；本机无 Isaac Gym，尚未验证接触加载、viewer、训练收敛或高楼梯成功率。
