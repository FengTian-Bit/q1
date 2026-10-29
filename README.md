# Q1 双轮腿训练

正式任务 **q1** 现在基于 `q1_ctbc`：保留六维动作与接触抬腿指导，
完整采用四轮足普通 T0 的有效混合地形、中心平台出生、难度行和距离升降级机制。
速度采样也采用 T0，只去掉 Y 向平移：vx ±1.8 m/s，目标航向 ±3.14 rad，
航向反馈 yaw 限制 ±1 rad/s，10 s 重采、20 s 回合。默认 4096 环境、30000 轮 PPO。
`q1_ctbc` 保留定向上楼专项，`q1_flat` 保留平地基线。

```bash
# 已安装 Isaac Gym Preview 4 的训练环境，项目根目录执行
pip install -e .
swanlab login
python wheel_legged_gym/scripts/train.py --task=q1 --headless
python wheel_legged_gym/scripts/play.py --task=q1 --load_run=Q1_T0_RUN --num_envs=1
python wheel_legged_gym/scripts/keyboard_play.py --task=q1 --load_run=Q1_T0_RUN --terrain=stairs_down
```

当前配置在 `wheel_legged_gym/envs/q1/q1_t0_config.py`。不再使用旧的 `--q1_stage`；
新 q1 同任务恢复需配套的 `resolved_config.json`，旧非 CTBC 的 q1 模型会明确拒绝。
已有 CTBC 模型可用 `--resume --load_run=/绝对路径/logs/q1_ctbc/CTBC_RUN` 迁移，
恢复网络和优化器后启动 T0 地形课程及新的指导渐入/退火，输出仍进入 `logs/q1`。

播放默认关闭指导评估纯策略，键盘布局沿用 W/S、A/D、X、空格、F；自动播放保留 T0 随机指令。
T0 最高难度行约 24 cm 单阶，踏面网格取整通常为 20 cm；`stairs_down` 从中心向外是上楼。
指导仍采用 8–16 cm 高抬腿参考和 Q1 原关节/轮速/力矩约束，不保证能跟踪 T0 的全部速度或地形。
周期视频默认关闭，日志仍使用 SwanLab 项目 `Q1` 并保留本地 TensorBoard。

具体几何、升降级、续训和播放见 [q1 任务说明](wheel_legged_gym/envs/q1/README.md)。
本地 CPU 接口测试不代表已验证真实仿真收敛。

## q1_ctbc 盲上楼梯专项

新增独立任务 `q1_ctbc`：以定向单台阶训练接触后交替抬腿，提供训练前馈、累计轮数退火、
真实落台成功判据及专项课程。默认 4096 环境、30000 轮，先学习 2.5–8 cm 单阶，
再通过 `--ctbc_stage=stairs` 训练 4–12 cm 连续三阶；`q1` 使用上述 T0 混合地形课程。

```bash
python wheel_legged_gym/scripts/train.py --task=q1_ctbc --ctbc_stage=single --headless
# RUN_NAME 和 CHECKPOINT 替换为本任务的实际运行目录和模型编号。
python wheel_legged_gym/scripts/evaluate_q1_ctbc.py --load_run=RUN_NAME --checkpoint=CHECKPOINT --headless --heights=2.5,5,8 --guidance=both
```

正式验收使用 `--guidance=off` 的纯网络成功率；CPU 测试尚不能证明物理上楼能力。
阶段续训、参数、评估单位与验证范围见 [CTBC 任务说明](wheel_legged_gym/envs/q1_ctbc/README.md)。

已有低台阶策略可用 `--ctbc_stage=stairs20 --resume --load_run=RUN_NAME --checkpoint=-1`
继续训练 10–20 cm 连续楼梯。该阶段按课程把引导从 8 cm 增至 16 cm，扩大腿目标范围，
保留原动作归一化尺度，并在跨入阶段时重新计时指导渐入与退火。新阶段的 20 cm 实际能力仍需评估。

---

## q1_flat 基线与原项目说明

基于 Wheel-Legged-Gym 的个人 Q1 训练仓库。
远程仓库：`git@github.com:FengTian-Bit/q1.git`。
原项目与依赖的 BSD 许可、版权声明保留在 LICENSE 和源码中。

新增任务 **q1_flat**：开链关节位置 PD + 轮速度反馈，平地固定高度、零转向，
支持纯平衡与低速前后行驶。参见 [Q1 训练说明](wheel_legged_gym/envs/q1_flat/README.md)。

```bash
# 项目根目录，已安装 Isaac Gym Preview 4 的 Linux Python 环境
pip install -e .
python wheel_legged_gym/scripts/train.py --task=q1_flat --q1_stage=balance --num_envs=32 --max_iterations=20
python wheel_legged_gym/scripts/train.py --task=q1_flat --q1_stage=balance --num_envs=1024 --max_iterations=2000 --headless
# 平衡通过后，按训练说明从指定 checkpoint 继续低速阶段。
```

训练日志、检查点、导出模型、视频和本地环境不提交 Git。URDF、STL、配置、源码和测试纳入版本管理。
当前配置经过静态模型检查和 CPU 控制/训练接口测试；未在 Isaac Gym 中验证收敛。

---

以下保留原项目说明。

# Wheel Legged Gym Environments #

### Acknowledgment

The implementation of Wheel-Legged-Gym relies on resources from [legged_gym](https://github.com/leggedrobotics/legged_gym) and [rsl_rl](https://github.com/leggedrobotics/rsl_rl) projects, created by the [Robotic Systems Lab](https://rsl.ethz.ch/).

Related Links:

- [Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://arxiv.org/abs/2109.11978)
- [Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion](https://arxiv.org/abs/2202.05481)

### Installation ###
1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
2. Install pytorch with cuda from https://pytorch.org/get-started/
3. Install Isaac Gym
   - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`)
4. Install wheel_legged_gym
    - Clone this repository
   - `cd Wheel-Legged-Gym && pip install -e .`

### Code Structure ###
1. Each environment is defined by an env file (`legged_robot.py`) and a config file (`legged_robot_config.py`). The config file contains two classes: one containing  all the environment parameters (`LeggedRobotCfg`) and one for the training parameters (`LeggedRobotCfgPPo`).  
2. Both env and config classes use inheritance.  
3. Each non-zero reward scale specified in `cfg` will add a function with a corresponding name to the list of elements which will be summed to get the total reward.  
4. Tasks must be registered using `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`. This is done in `envs/__init__.py`, but can also be done from outside of this repository.  

### Usage ###
1. Train:  
    ```python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_flat```
    -  To run on CPU add following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
    -  To run headless (no rendering) add `--headless`.
    - **Important**: To improve performance, once the training starts press `v` to stop the rendering. You can then enable it later to check the progress.
    - The trained policy is saved in `logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
    - Use TensorBoard to monitor training process `tensorboard --logdir=./ --port=8080` 
    -  The following command line arguments override the values set in the config files:
     - --task TASK: Task name.
     - --resume:   Resume training from a checkpoint
     - --experiment_name EXPERIMENT_NAME: Name of the experiment to run or load.
     - --run_name RUN_NAME:  Name of the run.
     - --load_run LOAD_RUN:   Name of the run to load when resume=True. If -1: will load the last run.
     - --checkpoint CHECKPOINT:  Saved model checkpoint number. If -1: will load the last checkpoint.
     - --num_envs NUM_ENVS:  Number of environments to create.
     - --seed SEED:  Random seed.
     - --max_iterations MAX_ITERATIONS:  Maximum number of training iterations.
     - --exptid EXPTID:  Experiment ID.
2. Play a trained policy:
   ```python wheel_legged_gym/scripts/play.py --task=wheel_legged_vmc_flat```
    - By default, the loaded policy is the last model of the last run of the experiment folder.
    - Other runs/model iteration can be selected by setting `load_run` and `checkpoint` in the train config.
3. Existing tasks:
   - wheel_legged: End-to-End training for open-chain robot in various terrains.
   - wheel_legged_vmc: Using VMC to unify motion control of open-chain and closed-chain mechanisms facilitates deploying policy onto closed-chain robots.
   - wheel_legged_vmc_flat: Train robot in flat terrain (low VRAM requirements).

### Adding a new environment ###
The base environment `legged_robot` implements a rough terrain locomotion task. The corresponding cfg does not specify a robot asset (URDF/ MJCF) and has no reward scales. 

1. Add a new folder to `envs/` with `'<your_env>_config.py`, which inherit from an existing environment cfgs  
2. If adding a new robot:
    - Add the corresponding assets to `resources/`.
    - In `cfg` set the asset path, define body names, default_joint_positions and PD gains. Specify the desired `train_cfg` and the name of the environment (python class).
    - In `train_cfg` set `experiment_name` and `run_name`
3. (If needed) implement your environment in <your_env>.py, inherit from an existing environment, overwrite the desired functions and/or add your reward functions.
4. Register your env in `isaacgym_anymal/envs/__init__.py`.
5. Modify/Tune other parameters in your `cfg`, `cfg_train` as needed. To remove a reward set its scale to zero. Do not modify parameters of other envs!


### Troubleshooting ###
1. If you get the following error: `ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory`, do: `sudo apt install libpython3.8`. It is also possible that you need to do `export LD_LIBRARY_PATH=/path/to/libpython/directory` / `export LD_LIBRARY_PATH=/path/to/conda/envs/your_env/lib`(for conda user. Replace /path/to/ to the corresponding path.).

### Known Issues ###
1. The contact forces reported by `net_contact_force_tensor` are unreliable when simulating on GPU with a triangle mesh terrain. A workaround is to use force sensors, but the force are propagated through the sensors of consecutive bodies resulting in an undesirable behaviour. However, for a legged robot it is possible to add sensors to the feet/end effector only and get the expected results. When using the force sensors make sure to exclude gravity from the reported forces with `sensor_options.enable_forward_dynamics_forces`. Example:
```
    sensor_pose = gymapi.Transform()
    for name in feet_names:
        sensor_options = gymapi.ForceSensorProperties()
        sensor_options.enable_forward_dynamics_forces = False # for example gravity
        sensor_options.enable_constraint_solver_forces = True # for example contacts
        sensor_options.use_world_frame = True # report forces in world frame (easier to get vertical components)
        index = self.gym.find_asset_rigid_body_index(robot_asset, name)
        self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)
    (...)

    sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
    self.gym.refresh_force_sensor_tensor(self.sim)
    force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
    self.sensor_forces = force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]
    (...)

    self.gym.refresh_force_sensor_tensor(self.sim)
    contact = self.sensor_forces[:, :, 2] > 1.
```



