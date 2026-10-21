# Q1 双轮腿训练

正式任务 **q1**：4096 个并行环境、30000 轮 PPO，平地固定高度、零转向，
混合停车与 ±0.15–0.3 m/s 前后行驶，包含实际腿姿态、双轮支撑、接触滑移、停车位置与直线路径约束。
保留 `q1_flat` 和原有三个任务作为对照。

```bash
# 已安装 Isaac Gym Preview 4 的训练环境，项目根目录执行
python wheel_legged_gym/scripts/train.py --task=q1 --headless
python wheel_legged_gym/scripts/play.py --task=q1 --num_envs=1
tensorboard --logdir=logs/q1
```

[q1 任务与训练说明](wheel_legged_gym/envs/q1/README.md)。q1 的观测接口已扩展，需要从头训练，不能直接续用 q1_flat 检查点。
本地 CPU 接口与 PPO 测试不代表已验证真实仿真性能；实际收敛需训练评估。

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



