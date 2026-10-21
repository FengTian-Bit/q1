"""Checkpoint recording adapted from Dreamwaq's isolated training-video process."""
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch


def ffmpeg_executable():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
    except ModuleNotFoundError:
        raise RuntimeError("Video recording requires ffmpeg or imageio-ffmpeg; run pip install -e .") from None
    return imageio_ffmpeg.get_ffmpeg_exe()


def video_settings(cfg):
    result = dict(cfg)
    for key, default in (("interval", 1500), ("fps", 25), ("width", 1280), ("height", 720)):
        value = cfg.get(key, default)
        if not math.isfinite(float(value)) or int(value) != float(value) or int(value) <= 0:
            raise ValueError("periodic_video.{} must be a positive integer".format(key))
        result[key] = int(value)
    if result["width"] % 2 or result["height"] % 2:
        raise ValueError("periodic_video width and height must be even")
    for key, default in (("duration_s", 12.0), ("timeout_s", 180.0)):
        result[key] = float(cfg.get(key, default))
        if not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError("periodic_video.{} must be positive and finite".format(key))
    for key, default in (("camera_offset", [1.5, 1.5, 0.8]), ("camera_target_offset", [0, 0, 0.1])):
        result[key] = [float(value) for value in cfg.get(key, default)]
        if len(result[key]) != 3 or not all(math.isfinite(v) for v in result[key]):
            raise ValueError("{} requires three finite values".format(key))
    schedule = cfg.get("command_schedule", []) or [(result["duration_s"], 0, 0)]
    result["command_schedule"] = []
    for entry in schedule:
        if len(entry) != 3:
            raise ValueError("Q1 video schedule entries are (duration_s, vx, yaw_rate)")
        duration, vx, yaw = map(float, entry)
        if not all(math.isfinite(v) for v in (duration, vx, yaw)) or duration <= 0:
            raise ValueError("Q1 video schedule values must be finite with positive durations")
        if abs(vx) > 0.3+1e-6 or abs(yaw) > 0.6+1e-6:
            raise ValueError("Q1 video targets must stay within +/-0.3 m/s and +/-0.6 rad/s")
        result["command_schedule"].append((duration, vx, yaw))
    result["duration_s"] = sum(entry[0] for entry in result["command_schedule"])
    return result


class CheckpointVideoRecorder:
    def __init__(self, sim_device, rl_device, log_dir, cfg):
        self.cfg = video_settings(cfg)
        self.sim_device, self.rl_device = str(sim_device), str(rl_device)
        self.log_dir = Path(log_dir).resolve()
        self.attempted_iterations = set()

    def should_record(self, iteration):
        return iteration > 0 and iteration % self.cfg["interval"] == 0

    def announce(self, iteration):
        upcoming = (iteration//self.cfg["interval"]+1)*self.cfg["interval"]
        print("[training video] enabled; next checkpoint {}; output: {}".format(
            upcoming, self.log_dir/"videos"/"training"), flush=True)

    def on_checkpoint(self, checkpoint, iteration):
        if not self.should_record(iteration) or iteration in self.attempted_iterations:
            return
        self.attempted_iterations.add(iteration)
        output = self.log_dir/"videos"/"training"/"iteration_{:06d}.mp4".format(iteration)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.is_file() and output.stat().st_size:
            return
        render_log = output.with_suffix(".render.log")
        command = [sys.executable, "-m", "wheel_legged_gym.scripts.record_checkpoint",
                   "--task", "q1", "--headless", "--sim_device", self.sim_device,
                   "--rl_device", self.rl_device, "--num_envs", "1",
                   "--checkpoint_path", str(Path(checkpoint).resolve()), "--video_output", str(output)]
        print("[training video] rendering checkpoint {}; log: {}".format(iteration, render_log), flush=True)
        try:
            with render_log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=self.cfg["timeout_s"], check=False)
            if result.returncode or not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError("child exit {}; no completed nonempty MP4".format(result.returncode))
        except Exception as exc:
            print("[training video] recording failed: {}; training continues; inspect {}".format(exc, render_log), flush=True)
            return
        print("[training video] saved {}".format(output), flush=True)


def capture_video(env, policy, output, cfg, stage):
    """Stream camera RGB frames to MP4; advance only the isolated evaluation env."""
    from isaacgym import gymapi

    cfg = video_settings(cfg)
    if cfg["fps"] > 1.0/env.dt+1e-9:
        raise ValueError("Video fps cannot exceed the control frequency")
    output = Path(output)
    if output.exists():
        raise FileExistsError(str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.mp4")
    props = gymapi.CameraProperties()
    props.width, props.height, props.enable_tensors = cfg["width"], cfg["height"], False
    handle = env.gym.create_camera_sensor(env.envs[0], props)
    if handle < 0:
        raise RuntimeError("Failed to create recording camera; check Vulkan/graphics support")
    process = None
    frames = 0
    try:
        process = subprocess.Popen([
            ffmpeg_executable(), "-y", "-nostdin", "-loglevel", "error", "-f", "rawvideo",
            "-pixel_format", "rgb24", "-video_size", "{}x{}".format(cfg["width"], cfg["height"]),
            "-framerate", str(cfg["fps"]), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-crf", "28", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
        ], stdin=subprocess.PIPE)
        with torch.inference_mode():
            env.reset()
            next_frame = 1.0/cfg["fps"]
            steps = int(math.ceil(cfg["duration_s"]/env.dt-1e-9))
            for index in range(steps):
                elapsed = index*env.dt
                end = 0.0
                for duration, vx, yaw in cfg["command_schedule"]:
                    end += duration
                    if elapsed < end:
                        break
                if stage == "balance":
                    vx, yaw = 0.0, 0.0
                elif stage == "low_speed":
                    yaw = 0.0
                env.set_command_target(vx, yaw)
                actions, _ = policy(env.obs_buf, env.obs_history)
                env.step(actions)
                if (index+1)*env.dt+1e-9 < next_frame:
                    continue
                position = env.root_states[0, :3].detach().cpu().tolist()
                camera = gymapi.Vec3(*[position[i]+cfg["camera_offset"][i] for i in range(3)])
                target = gymapi.Vec3(*[position[i]+cfg["camera_target_offset"][i] for i in range(3)])
                env.gym.set_camera_location(handle, env.envs[0], camera, target)
                env.gym.fetch_results(env.sim, True)
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)
                rgba = env.gym.get_camera_image(env.sim, env.envs[0], handle, gymapi.IMAGE_COLOR)
                rgb = np.asarray(rgba, dtype=np.uint8).reshape(cfg["height"], cfg["width"], 4)[:, :, :3]
                process.stdin.write(np.ascontiguousarray(rgb).tobytes())
                frames += 1
                next_frame += 1.0/cfg["fps"]
        process.stdin.close()
        if process.wait(timeout=60) != 0 or not frames or not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("ffmpeg did not produce a completed video")
        os.replace(str(temporary), str(output))
        return frames
    finally:
        if process is not None:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                process.kill()
                process.wait()
        env.gym.destroy_camera_sensor(env.sim, env.envs[0], handle)
