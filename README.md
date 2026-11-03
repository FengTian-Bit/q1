# Q1 双轮腿机器人训练

基于 Wheel-Legged-Gym 与 Isaac Gym Preview 4，包含 Q1 平地、定向上楼、高站姿对照、T0 混合地形和原项目训练任务。

## 当前状态（2026-11-03）

**目前使用 `q1_ctbc` 的 `stairs` 阶段训练。22 cm 楼梯课程训练时会出现地形难度等级降为 0 的情况，仍需要调参。** 这是当前训练反馈，22 cm 课程还不能视为已稳定收敛。

正确任务名是 `q1_ctbc`，不是 `q1_cbtc`。22 cm 课程沿用参数名 `stairs20`，没有 `stairs22cm` 或 `starirs22cm` 参数。新版课程第 0 级对应 4 cm 楼梯，等级降为 0 不表示楼梯高度为 0 cm。

当前使用的命令：

```bash
python wheel_legged_gym/scripts/train.py --task=q1_ctbc --ctbc_stage=stairs --num_envs=1024 --max_iterations=30000 --run_name=stairs --headless
```

## 安装

训练需要支持 Isaac Gym Preview 4 的 Linux、NVIDIA GPU 和驱动环境。以下命令在 Linux 终端执行。沿用原项目的 Python 3.8 环境；本仓库依赖旧版 Isaac Gym 和 NumPy，不要直接套用最新 Python/PyTorch 环境。

### 1. 创建环境、获取仓库

```bash
conda create -n q1 python=3.8 -y
conda activate q1
git clone --branch tf/q1-ctbc https://github.com/FengTian-Bit/q1.git
cd q1
```

### 2. 安装 Isaac Gym Preview 4

从 [NVIDIA 下载页](https://developer.nvidia.com/isaac-gym) 获取 Preview 4 并解压。根据下载包的 `docs/index.html` 配置匹配的驱动、CUDA 和 PyTorch。本仓库不包含 Isaac Gym 安装包。

```bash
# 替换为实际解压路径，保持 q1 环境已激活。
python -m pip install -e /path/to/isaacgym/python
python -c "import isaacgym; import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

GPU 训练前应确认 CUDA 可用，并按安装包说明运行自带示例验证仿真环境。

### 3. 安装仓库依赖、检查启动

在仓库根目录执行：

```bash
python -m pip install -e .
python -m pip check
# 小规模启动检查，不代表完成训练。
python wheel_legged_gym/scripts/train.py --task=q1_ctbc --ctbc_stage=stairs --num_envs=32 --max_iterations=20 --logger=tensorboard --headless
```

依赖以 [setup.py](setup.py) 为准，包含 NumPy `<1.20.0`、TensorBoard、SwanLab、ONNX 等。使用默认 SwanLab 在线记录前运行 `swanlab login`；只用本地日志时，在训练命令末尾加 `--logger=tensorboard`，或使用 `--swanlab_mode=offline`。

## 全部任务与训练命令

所有命令在仓库根目录运行。`--headless` 关闭显示窗口；显存不足时降低 `--num_envs`。以下覆盖当前注册的全部七个任务。

| 任务 | 用途 | 阶段参数 |
|---|---|---|
| `q1_ctbc` | 当前使用的定向上楼 | `--ctbc_stage` |
| `q1_ctbc_highcom` | 高站姿 CTBC 对照 | `--ctbc_stage` |
| `q1` | T0 混合地形、距离课程 | 不使用阶段参数 |
| `q1_flat` | 平衡、低速前后运动 | `--q1_stage` |
| `wheel_legged` | 原项目开链轮腿地形训练 | 默认配置 |
| `wheel_legged_vmc` | 原项目 VMC 地形训练 | 默认配置 |
| `wheel_legged_vmc_flat` | 原项目 VMC 平地训练 | 默认配置 |

### 1. q1_ctbc：当前上楼任务

默认 1024 环境、30000 次更新、`stairs` 阶段，20% 平地与 80% 楼梯。

| 阶段 | 地形与用途 |
|---|---|
| `single` | 2.5–8 cm 单阶，基础抬腿 |
| `stairs` | 4–12 cm 连续三阶，当前使用 |
| `robust` | 4–12 cm 连续三阶，加入域随机化与观测噪声 |
| `stairs20` | 12 级、4–22 cm 连续三阶，目前会降到等级 0，待调参 |

各阶段从头训练：

```bash
python wheel_legged_gym/scripts/train.py --task=q1_ctbc --ctbc_stage=single --headless
python wheel_legged_gym/scripts/train.py --task=q1_ctbc --ctbc_stage=stairs --num_envs=1024 --run_name=stairs --headless
python wheel_legged_gym/scripts/train.py --task=q1_ctbc --ctbc_stage=robust --headless
# 22 cm 实验课程：目前存在难度降为 0 的问题，仍需调参。
python wheel_legged_gym/scripts/train.py --task=q1_ctbc --ctbc_stage=stairs20 --num_envs=1024 --run_name=stairs22_dense --headless
```

22 cm 课程配套设置：踏面固定 0.5 m、速度 0.15–0.20 m/s、抬轮参考 8–17.6 cm、摆动 0.85 s；髋/膝动作偏移范围为 ±0.70/±1.35 rad。每 10 个有效楼梯回合判定课程，成功至少 8 次升级，少于 3 次降级。卡阶检测目前只记录，不直接终止回合。

续训或切换课程：

```bash
# RUN_NAME 替换为 logs/q1_ctbc 下的运行目录名。
# CHECKPOINT 替换为模型编号，例如 model_3000.pt 对应 3000。
python wheel_legged_gym/scripts/train.py --task=q1_ctbc --ctbc_stage=stairs --resume --load_run=RUN_NAME --checkpoint=CHECKPOINT --headless
python wheel_legged_gym/scripts/train.py --task=q1_ctbc --ctbc_stage=robust --resume --load_run=RUN_NAME --checkpoint=CHECKPOINT --headless
python wheel_legged_gym/scripts/train.py --task=q1_ctbc --ctbc_stage=stairs20 --resume --load_run=RUN_NAME --checkpoint=CHECKPOINT --run_name=stairs22_dense --headless
```

续训需要模型旁的 `resolved_config.json`。省略 `--ctbc_stage` 时保留保存配置；显式指定时应用当前阶段预设，因此启用新的 4–22 cm 配置必须指定 `--ctbc_stage=stairs20`。首次迁入新高阶段重新计时引导渐入和退火，同配置续训保留起点。详见 [CTBC 说明](wheel_legged_gym/envs/q1_ctbc/README.md)。

### 2. q1_ctbc_highcom：高站姿对照

默认 `single`、4096 环境，修改名义站姿及配套引导。这里的 `stairs20` 仍是六级 10–20 cm、8–16 cm 引导、0.8 s 摆动，与普通 `q1_ctbc` 新版课程不同。

```bash
python wheel_legged_gym/scripts/train.py --task=q1_ctbc_highcom --ctbc_stage=single --headless
python wheel_legged_gym/scripts/train.py --task=q1_ctbc_highcom --ctbc_stage=stairs --headless
python wheel_legged_gym/scripts/train.py --task=q1_ctbc_highcom --ctbc_stage=robust --headless
python wheel_legged_gym/scripts/train.py --task=q1_ctbc_highcom --ctbc_stage=stairs20 --headless
# 从本任务已训练的运行进入高阶段。
python wheel_legged_gym/scripts/train.py --task=q1_ctbc_highcom --ctbc_stage=stairs20 --resume --load_run=HIGHCOM_RUN --checkpoint=CHECKPOINT --headless
```

不能把普通 `q1_ctbc` 模型直接作为该任务的完整续训状态。详见 [高站姿说明](wheel_legged_gym/envs/q1_ctbc_highcom/README.md)。

### 3. q1：T0 混合地形

默认 4096 环境、30000 次更新，使用 T0 混合地形、距离升降级及速度指令。不接受 `--q1_stage` 或 `--ctbc_stage`。

```bash
python wheel_legged_gym/scripts/train.py --task=q1 --headless
python wheel_legged_gym/scripts/train.py --task=q1 --resume --load_run=Q1_RUN --checkpoint=CHECKPOINT --headless
# 从 CTBC 运行迁移到 T0，替换为实际运行目录。
python wheel_legged_gym/scripts/train.py --task=q1 --resume --load_run=/path/to/q1/logs/q1_ctbc/CTBC_RUN --checkpoint=CHECKPOINT --headless
```

详见 [T0 说明](wheel_legged_gym/envs/q1/README.md)。

### 4. q1_flat：平地基线

先训练平衡，再从通过评估的模型续训低速前后运动。当前入口只支持 `balance` 和 `low_speed`，会拒绝 `turning`。

```bash
python wheel_legged_gym/scripts/train.py --task=q1_flat --q1_stage=balance --num_envs=1024 --max_iterations=2000 --headless
python wheel_legged_gym/scripts/train.py --task=q1_flat --q1_stage=low_speed --resume --load_run=BALANCE_RUN --checkpoint=CHECKPOINT --num_envs=1024 --max_iterations=2000 --headless
# 也可从头启动低速训练。
python wheel_legged_gym/scripts/train.py --task=q1_flat --q1_stage=low_speed --num_envs=1024 --headless
```

详见 [平地说明](wheel_legged_gym/envs/q1_flat/README.md)。

### 5. 原项目三个任务

这些任务使用原项目机器人和配置，不是 Q1 上楼任务。

```bash
python wheel_legged_gym/scripts/train.py --task=wheel_legged --headless
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc --headless
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_flat --headless
```

## 日志、续训与评估

模型通常保存在 `logs/<experiment_name>/<运行目录>/model_<编号>.pt`，请保留同目录配置快照。`--checkpoint=-1` 选择指定运行的最新模型；排查训练退化时应指定已验证的模型编号，避免直接使用退化后的最新模型。当前 Q1 训练流程的 `--max_iterations` 表示本次追加的更新次数。

```bash
tensorboard --logdir=logs --port=8080
# 当前 stairs：固定三阶，比较纯策略和指导开启表现。
python wheel_legged_gym/scripts/evaluate_q1_ctbc.py --task=q1_ctbc --load_run=RUN_NAME --checkpoint=CHECKPOINT --headless --num_envs=64 --heights=4,6,8,10,12 --tread=0.35 --steps=3 --episodes=100 --speed=0.2 --guidance=both
# 22 cm 课程评估：高度单位 cm，踏面单位 m。
python wheel_legged_gym/scripts/evaluate_q1_ctbc.py --task=q1_ctbc --load_run=HIGH_RUN_NAME --checkpoint=CHECKPOINT --headless --num_envs=64 --heights=4,5.5,7,8.5,10.5,12,13.5,15,17,18.5,20,22 --tread=0.5 --steps=3 --episodes=100 --speed=0.2 --guidance=both
# CTBC 普通播放默认是 5 cm 单阶，不是整个训练课程。
python wheel_legged_gym/scripts/play.py --task=q1_ctbc --load_run=RUN_NAME --checkpoint=CHECKPOINT --num_envs=1
python wheel_legged_gym/scripts/play.py --task=q1 --load_run=Q1_RUN --checkpoint=CHECKPOINT --num_envs=1
```

以 `--guidance=off` 的纯策略成功率判断是否学会爬阶。22 cm 课程应结合分高度成功率、卡阶率、动作裁剪率和探索标准差调参，不能仅看总回报。CPU 回归测试通过不等于物理仿真收敛或实机通过。

## 项目文件与致谢

- 任务注册：`wheel_legged_gym/envs/__init__.py`。
- 训练入口：`wheel_legged_gym/scripts/train.py`。
- 上楼配置：`wheel_legged_gym/envs/q1_ctbc/q1_ctbc_config.py`。
- 历史基线：[训练对齐记录](docs/q1_ctbc_training_alignment.md)。
- 日志、检查点、导出模型、视频和本地环境不提交 Git；机器人资源、源码、配置与测试纳入版本管理。

本项目基于 Wheel-Legged-Gym，使用 [legged_gym](https://github.com/leggedrobotics/legged_gym) 和 [rsl_rl](https://github.com/leggedrobotics/rsl_rl) 的工作。原项目和依赖的版权声明保留在源码和 [LICENSE](LICENSE) 中。
