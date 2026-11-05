# q1_tcn_v7_joint_v1：停车与运动从零共同训练

固定任务名为 `q1_tcn`，使用与 `q1` 相同的公共入口。从随机权重同时训练停车 TCN/Actor 和运动 TCN/Actor，共用特权 Critic 和 PPO；不读取任何旧模型，不依赖 `logs/`、pilot JSON、Loco 用户目录或旧停车模型。

## 训练、续训与播放

在已配置 Isaac Gym Preview 4、匹配的 Python/PyTorch/CUDA 的 Linux NVIDIA GPU 环境中，从项目根目录运行：

```bash
pip install -e .
# 激活训练环境；Isaac Gym 必须先于 torch 导入。
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# 新训练：默认 4096 环境，30000 次 PPO 更新。
python wheel_legged_gym/scripts/train.py --task=q1_tcn --headless

# 先检查少量环境的完整训练流程。
python wheel_legged_gym/scripts/train.py --task=q1_tcn --headless \
  --num_envs=64 --max_iterations=2 --logger=tensorboard --run_name=smoke

# RUN_NAME 替换为 logs/q1_tcn 中本任务实际运行目录。
python wheel_legged_gym/scripts/train.py --task=q1_tcn --headless \
  --resume --load_run=RUN_NAME --checkpoint=-1

python wheel_legged_gym/scripts/play.py --task=q1_tcn \
  --load_run=RUN_NAME --checkpoint=-1 --num_envs=1

# 自动正反向与转向播放。
python wheel_legged_gym/scripts/play.py --task=q1_tcn \
  --load_run=RUN_NAME --checkpoint=-1 --num_envs=1 --play_mode=auto --play_duration=60
```

默认记录 SwanLab 项目 `Q1` 和本地 TensorBoard；只需本地日志可传 `--logger=tensorboard`。CSV、配置、模型和源码保存在 `logs/q1_tcn/<时间>_<名称>/`。`--max_iterations` 表示本次追加更新数。续训恢复保存的配置、网络、Adam、计数和随机数状态；仿真环境重新开始回合。可覆盖环境数、种子、日志参数和更新预算，不能静默更换奖励或网络。旧实验的冻结停车 v7 检查点不属于此共同训练版本，不能通过本入口续训。

键盘播放 W/S 调整线速度、A/D 调整偏航，空格停车。完整的两个策略分支都保存在一个 `model_N.pt` 内，播放不需要另一份停车模型。

## 固定训练条件

| 项目 | 设置 |
|---|---|
| 任务版本 | `q1_tcn_v7_joint_v1` |
| 地形 | 平地，摩擦 0.8，保留观测噪声；关闭原版本已关闭的摩擦/质量/COM 随机化和推扰 |
| 速度范围 | vx ±1.2 m/s、yaw ±1 rad/s，从第一轮覆盖完整范围 |
| 指令加速度 | 线速度每秒变化不超过 2 m/s |
| 环境分配 | 20% 专门停车；其余环境中 75% 使用直行/原地转向/组合运动与启停序列，25% 使用随机指令 |
| 观测 | 30 维本体/里程计观测、16 帧历史共 480 维，Critic 146 维 |
| 线速度估计 | A0：预测头和监督损失关闭，Actor 不接收仿真真实线速度 |
| 主奖励 | 线速度 4、机身角速度 2、实测轮速差引导 0.5 |
| 跟踪形式 | 当前 v7 的相对误差指数核 + Huber；零速与停车参数沿用当前版本 |
| 停车奖励 | 漂移 -1、速度 -0.2，5 mm 死区、3 cm 尺度、0.5 s 制动和 0.5 s 渐入 |
| PPO | 学习率 3e-5、KL 限制 0.02、3 epochs、4 minibatches、gamma 0.998 |
| 对称损失 | 0.1 |
| 网络/控制 | 保留 v7 轮速前馈、差速前馈、轮残差及腿/轮控制限幅 |

区别于原先加载基础策略后冻结停车分支的续训，这一版停车与运动的 TCN 和 Actor 均可训练。停车分支的独立旧 Critic 和禁用的辅助头不参与优化，共享 Critic 为两个动作分支提供优势估计。原有实验构造器仍保持冻结停车的行为，只有本任务使用共同训练子类。

这固定的是代码、结构和训练条件；此前 model1750 的成绩来自多阶段续训，不能作为从零共同训练已经收敛的证明。当前本地仅验证 CPU 张量接口、两个分支的梯度/PPO 更新、配置恢复与入口；还需在 Isaac Gym 上检查 4096 环境运行及独立种子性能。

## 文件位置与版本快照

- `wheel_legged_gym/envs/q1_tcn/task_config.py`：固定环境、奖励、网络及 PPO 参数。
- `joint_task.py`：仅作用于本任务的环境分配与可训练停车网络子类。
- `joint_runner.py`：共同 PPO、保存/恢复、CSV 和训练日志。
- `task.py`：公共入口适配、配置恢复、源码及运行环境记录。
- `play.py`：复用原有播放循环，增加可传入已解析配置的接口。
- `wheel_legged_gym/envs/__init__.py`：只更新 `q1_tcn` 的注册。
- `wheel_legged_gym/scripts/train.py`、`play.py`：只增加 `q1_tcn` 分派。
- `tests/test_q1_tcn_joint.py`：共同训练测试；其他任务仍执行其原分支。
- `tools/package_q1_tcn.py`：生成无模型依赖的源码包及逐文件 SHA256 清单。

```bash
python tools/package_q1_tcn.py
```

源码包默认位于 `releases/q1_tcn_v7_joint_v1.tar.gz`；已存在时拒绝覆盖。迁移后解压到新的项目目录，安装依赖，执行上面的公共训练命令。源码包不包含 Isaac Gym、CUDA、Conda 环境或旧检查点；训练时另存 Python/PyTorch/CUDA 版本和所用源码散列。
