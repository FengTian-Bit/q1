# Q1 平地训练

`q1_flat` 基于 `wheel_legged` 的开链关节控制，使用 `resources/robots/q1/urdf/q1.urdf`。
保持原来的三个任务。Q1 使用独立环境覆盖原机器人的腿角计算、关节映射、重置、噪声索引和终止条件。

## 接口与初始状态

- 策略动作顺序：`L_thigh_joint, L_calf_joint, L_foot_joint, R_thigh_joint, R_calf_joint, R_foot_joint`。
- 六个关节轴均为 `+Y`，左右对应关节的动作同号。实际导入顺序通过名字映射，不能把旧版右腿负号用于实机接口。
- 左右默认角度均为 `[0.6, -1.2, 0.0] rad`；根坐标初始高度 `0.402 m`，目标高度 `0.397 m`。
- 初始轮底离地约 `0.00534 m`；不随机改变初始腿角。初始水平速度扰动 ±0.02 m/s，roll/pitch 角速度扰动 ±0.03 rad/s。
- 腿动作乘 `0.35 rad` 后叠加默认角度，经 `Kp=40, Kd=1` 得到力矩；轮动作乘 `10 rad/s`，经 `Kd=0.10` 得到力矩。
- 动作裁剪到 ±1，所有电机力矩遵循 URDF：腿 14 N·m、轮 0.8 N·m。增益是初始仿真参数，需要用真实执行器特性验证。
- 策略周期 10 ms（100 Hz），物理周期 5 ms（200 Hz）。
- Actor 27 维，历史 5 帧共 135 维，特权观测 143 维，速度估计器输出 3 维；critic 最终输入 146 维。
- Actor 顺序：角速度(3)、投影重力(3)、指令 `[vx, yaw_rate, height]`(3)、关节位置误差(6)、关节速度(6)、上一动作(6)。
- 两个轮位置误差通道恒为零，避免累计轮角泄露到观测；轮速保留。不使用 wheel-still 奖励，以允许原地平衡修正。
- 平面摩擦 0.8、恢复系数 0；第一版关闭域随机化、推扰和观测噪声。
- 机身接触、倾斜超过约 43° 或根坐标高度低于 0.20 m 时立即终止；20 s 正常结束单独标记 timeout。

## 训练

从项目根目录，在已安装 Isaac Gym Preview 4、匹配的 CUDA PyTorch 的 Linux Python 环境执行。
本项目原安装约束见 `setup.py`；不要把系统 Python 或当前四轮腿项目环境直接当作已验证的 Q1 环境。

```bash
pip install -e .

# 先观察模型和短流程。训练初期倒地属于正常现象。
python wheel_legged_gym/scripts/train.py --task=q1_flat --q1_stage=balance --num_envs=32 --max_iterations=20

# 第一阶段：前进与偏航指令都为零，学习动态平衡。
python wheel_legged_gym/scripts/train.py --task=q1_flat --q1_stage=balance --num_envs=1024 --max_iterations=2000 --headless

tensorboard --logdir=logs/q1_flat
```

第一阶段验收：多个随机种子下，多数回合能完成 20 s，机身保持直立，高度接近目标，
水平速度小，腿/轮力矩没有持续饱和。回报上涨和单次演示不足以证明通过。

通过后，从明确的平衡 checkpoint 继续训练。以下 `BALANCE_RUN` 替换成
`logs/q1_flat/` 内实际的运行目录名，`2000` 替换成实际 checkpoint 编号。

```bash
# 第二阶段：每 4 s 重新采样；50% 零指令，其他指令均匀采样自 [-0.3, 0.3] m/s。
# 偏航始终为零，高度保持 0.397 m。
python wheel_legged_gym/scripts/train.py --task=q1_flat --q1_stage=low_speed --resume --load_run=BALANCE_RUN --checkpoint=2000 --num_envs=1024 --max_iterations=2000 --headless
```

不带 `--resume` 的 low_speed 阶段会从头训练。默认阶段为 low_speed，默认环境数 1024；
根据显存和吞吐再提高环境数。阶段切换保留网络接口。

日志位置 `logs/q1_flat/<时间>_<阶段>/`，保存模型、Q1 源码、实际生效的 `resolved_config.json` 和本次 URDF。
本仓库旧 runner 的续训文件编号使用本次 `max_iterations`，不能仅凭编号推断累计训练量；用实际输出目录和时间区分运行。

## 播放与导出

```bash
# 用实际运行目录和 checkpoint 替换占位值。
python wheel_legged_gym/scripts/play.py --task=q1_flat --q1_stage=balance --load_run=BALANCE_RUN --checkpoint=2000 --num_envs=1
python wheel_legged_gym/scripts/play.py --task=q1_flat --q1_stage=low_speed --load_run=LOW_SPEED_RUN --checkpoint=2000 --num_envs=1
```

Q1 专用播放共运行 60 s；balance 全程零速度；low_speed 按 `0 → +0.2 → 0 → -0.2 → 0 m/s` 循环，每段 4 s。
它使用 Q1 高度，无额外速度积分补偿，支持单个环境。每秒输出环境 0 的速度、高度及当前步重置数量。
JIT 导出到 `logs/q1_flat/exported/policies/`；该模型输入为单帧观测和历史观测，部署必须复现相同的关节顺序、缩放和 100 Hz 时序。

## 可复现检查及限制

```bash
python -m unittest discover -s tests -p test_q1_flat.py -v
python tools/check_q1_model.py --output docs/q1_model_check.json
python resources/robots/q1/tools/check_collision_clearance.py --urdf resources/robots/q1/urdf/q1.urdf --output docs/q1_collision_clearance.json
```

CPU 测试覆盖真实环境方法、动作映射、观测、重置和 PPO 两次更新，但 Gym 接口使用张量桩，**不模拟接触或平衡动力学**。
静态模型检查见 `docs/q1_model_check.json`。所有惯量正定、满足主惯量三角不等式，
回转半径与可见 mesh 尺寸的筛查没有发现明显异常；缺 CAD 质量分布，不能证明每个惯量精确。
模型有 9 个 link、8 个 joint（6 活动、2 fixed）、13 个简单几何碰撞体、9 个可解析 STL。
轮关节保持 revolute，限位由 ±999999 修正为用户规范要求的 ±99999，其他动力学和碰撞几何未改。

现有全范围分离证明脚本检查了 78 对碰撞体，其中 6 对未能完成证明：
机身与左右膝电机、机身与左右轮边电机、左右小腿上段与对应轮子。
部分原因是多关节证明不受支持或轮角范围导致细分量过大；这不等同于确认发生碰撞。
初始姿态下各碰撞体均未穿地。初始自碰撞、实际 PhysX 接触、viewer 和策略收敛仍需在 Isaac Gym 环境验证。

## Git 管理

本项目 remote 为 `git@github.com:FengTian-Bit/q1.git`，保留原项目 BSD 许可和版权。
源码、URDF、STL、测试和静态报告提交；日志、检查点、导出模型、视频、环境文件被 `.gitignore` 排除。

```bash
git status
git add wheel_legged_gym/envs/q1_flat
git commit -m "Tune Q1 flat balance parameters"
git push
```
