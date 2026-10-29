# Q1：CTBC 控制与四轮足 T0 地形课程

当前 `--task=q1` 注册为 `Q1T0CTBC(Q1CTBC)`：沿用双轮足的接触抬腿指导、六维动作、执行动作历史、31 维 Actor、151 维 critic、五帧历史和 PPO，地形生成与距离课程采用四轮足普通 `--task=t0` 的设置。速度采样也采用 T0，仅去掉 Y 向平移。独立 `q1_ctbc` 保持原来的定向上楼课程；`q1_flat` 保持平地基线。

旧 `q1.py` / `q1_config.py` 仍是 CTBC 的内部运动基类，避免循环继承；当前训练配置在 **`q1_t0_config.py`**，环境在 **`q1_t0.py`**。旧实验记录移至 [legacy_locomotion.md](legacy_locomotion.md)。

## 训练

在项目根目录、已安装 Isaac Gym 的 Linux 环境运行：

```bash
python wheel_legged_gym/scripts/train.py --task=q1 --headless
```

默认 4096 环境、追加 30000 次 PPO 更新，日志保存在 `logs/q1/<运行目录>`，SwanLab 项目沿用 `Q1`。不再使用旧的 `--q1_stage` 或专项 `--ctbc_stage`；这两个参数对新 `q1` 明确报错。周期视频沿用 CTBC 的默认关闭状态。

## 与 T0 对齐的设置

| 项目 | 当前 q1 |
|---|---|
| 地图 | 20×20 块，每块 8×8 m，25 m 外边界，水平网格 0.1 m、垂直网格 0.005 m |
| 类型 | 平地 10%、上下坡各 10%、粗糙坡 15%、上下阶各 20%、离散障碍 15% |
| 坡度 | 普通坡 0.1–0.7，粗糙坡 0.1–0.5，粗糙高度 ±0.05 m |
| 台阶 | 名义单级高 0.05–0.25 m，名义踏面 0.25 m，中心平台 3 m |
| 障碍 | 高度 0.05–0.30 m，尺寸 1–2 m，每块 20 个 |
| 难度行 | 从 0 到 0.95；最高行名义台阶高 0.24 m，网格取整同 T0 |
| 出生 | 地块中心平台，以中心局部最高地面确定 Z；初始随机第 0–5 行 |
| 指令 | vx 均匀采样 ±1.8 m/s，每 10 s 重采；abs(vx)≤0.2 清零，无额外停车概率 |
| 转向 | 采样目标航向 ±3.14 rad，每步 yaw=clip(0.5×wrap(目标航向−当前航向), −1, 1) rad/s |
| 回合 | 20 s；不再套用旧 Q1 指令加速度渐变 |

T0 的空心楼梯和 highplatform 当前比例为零，不属于本次有效地形。两边使用相同 Gym 金字塔 primitives，名义 25 cm 踏面在 10 cm 网格上实际通常是 20 cm。`stairs_up` 中心高，从中心向外是下楼；`stairs_down` 中心低，向外是上楼。地图启动时生成，重置时切换地块。

域随机化、Q1 机器人模型和 PPO 网络沿用 CTBC，未切换为 T0 的四轮控制、随机化或碰撞估计网络。出生地块及中心地面原点采用 T0，机器人自身重置仍用 Q1 的小速度扰动，没有加入 T0 的 ±0.3 m 出生位置或六维 ±0.3 速度扰动。T0 的 ±1.8 m/s 指令范围超过 Q1 原轮速上限对应的约 1.35 m/s 纯滚动速度；控制器继续保留原轮速和力矩限制，可能发生饱和，不能据采样范围认为已具备该速度能力。

## 地形升降级

每次结束回合，以机器人末态相对地块原点的 XY 净位移判定，完全采用 T0 距离规则：

- 净位移 **>4 m**：升一级。
- 未升级且净位移 **<abs(vx)×20 s×0.5**：降一级。Q1 的命令是 `[vx, yaw_rate, height]`，不能把 yaw 当成 T0 的 vy 计算平移距离。
- 最低夹住第 0 行，超出最高行时随机回某一行；平地也使用同一规则。

没有五回合成功率窗口，也不要求速度追踪误差或完整爬完一座楼梯。保留 T0 每轮更新前随机初始 episode 时钟的训练行为。重置前保存末态指令，避免基类清零后使降级失效。

新 q1 取消 CTBC 固定终点、登顶即重置、走廊侧边界和世界 X 前进奖励，保留 Q1 身体失败与超时结束。日志 `q1_t0_episode` / `q1_t0_distance_m` 表示距离课程结果，不是爬楼成功率。

## CTBC 控制与续训

保留高阶段腿动作尺度 `[0.5,0.9,0.5,0.9]`，腿偏移限额 ±0.70/±1.35 rad，轮动作仍 ±1。接触指导使用 0.8 s 脉冲，楼梯参考按实际单级高度的 0.8 倍限制在 8–16 cm；真正台阶可能高于参考，仍需策略学习机身和支撑腿配合。正向、倒退和转向运动均允许接触触发，Actor 不直接输入接触力、地形高度或指导相位。

新建训练的指导时钟从零开始。迁移现有 CTBC 策略时，用源运行的**完整绝对路径**，恢复网络、优化器与累计迭代，并从该迭代重新开始指导的 1000 轮渐入、总计 3000 轮保持及随后 12000 轮退火：

```bash
python wheel_legged_gym/scripts/train.py --task=q1 --resume \
  --load_run=/home/bit/tf/q1/logs/q1_ctbc/CTBC_RUN \
  --checkpoint=-1 --run_name=t0 --headless
```

新 q1 的同任务续训恢复检查点旁 `resolved_config.json`，保留已保存的地形、指令、控制配置和指导时钟：

```bash
python wheel_legged_gym/scripts/train.py --task=q1 --resume \
  --load_run=Q1_T0_RUN --checkpoint=-1 --headless
```

不要用 `--experiment_name=q1_ctbc` 指定迁移来源；新日志保持在 q1 实验下。旧版非 CTBC 的 q1、q1_flat 检查点会明确拒绝，避免形状相同但动作语义不同的错误恢复。改 Python 默认值不会覆盖同任务续训保存的配置。

## 播放

```bash
python wheel_legged_gym/scripts/play.py --task=q1 --load_run=Q1_T0_RUN --num_envs=1
python wheel_legged_gym/scripts/play.py --task=q1 --load_run=Q1_T0_RUN --play_mode=auto --headless
python wheel_legged_gym/scripts/keyboard_play.py --task=q1 --load_run=Q1_T0_RUN \
  --terrain=stairs_down --difficulty=0.75 --num_envs=1
```

播放先恢复模型配套配置，再应用环境数或固定地形覆盖，默认关闭指导评估纯网络。键盘 W/S 前后、A/D 转向、X 停车、空格暂停、F 相机；自动播放保留 T0 随机指令。专项 `evaluate_q1_ctbc.py` 继续只接受 `q1_ctbc`，其定向登顶成功率不能用于新金字塔任务。

CPU 测试验证几何、课程、命令、控制和恢复接口；本机没有 Isaac Gym，尚未验证新任务的真实仿真收敛。
