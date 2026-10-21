"""Checkpoint trigger, headless camera, real FFmpeg encoding and mocked cloud media."""
from collections import deque
import contextlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

import test_q1_swanlab as logging_tests
from test_q1 import improved
from test_q1_flat import NS, load_code


VIDEO = dict(NS, os=os, Path=Path, shutil=shutil, subprocess=subprocess, sys=sys)
load_code("wheel_legged_gym/rsl_rl/utils/video.py", VIDEO)
WRITER = dict(logging_tests.WRITER_NS, time=logging_tests.time, subprocess=subprocess,
              ffmpeg_executable=VIDEO["ffmpeg_executable"])
load_code("wheel_legged_gym/rsl_rl/utils/swanlab_utils.py", WRITER)


@contextlib.contextmanager
def fake_gym(api):
    previous = sys.modules.get("isaacgym")
    sys.modules["isaacgym"] = SimpleNamespace(gymapi=api)
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop("isaacgym", None)
        else:
            sys.modules["isaacgym"] = previous


class Q1VideoTests(unittest.TestCase):
    def setUp(self):
        logging_tests.Q1SwanLabTests.setUp(self)
        self.sdk.Video = Mock(side_effect=lambda path, **kw: {"path": path, **kw})

    def test_video_defaults_and_recording_checkpoint_independent_of_save_interval(self):
        defaults = NS["class_to_dict"](NS["Q1CfgPPO"]())["runner"]["periodic_video"]
        self.assertEqual((defaults["enabled"], defaults["interval"], defaults["duration_s"],
                          defaults["fps"], defaults["width"], defaults["height"]),
                         (True, 1500, 12, 25, 1280, 720))
        with tempfile.TemporaryDirectory(dir=os.environ.get("Q1_TEST_TMP")) as folder:
            runner = logging_tests.Q1SwanLabTests.make_runner(self, folder)
            runner._learn.__globals__["CheckpointVideoRecorder"] = VIDEO["CheckpointVideoRecorder"]
            runner._learn.__globals__["SwanLabSummaryWriter"] = WRITER["SwanLabSummaryWriter"]
            runner.cfg.update(save_interval=3, periodic_video=dict(defaults, interval=2))
            runner.current_learning_iteration = 1  # resumed past a previous save
            preview = Path(folder)/"preview.gif"
            preview.write_bytes(b"GIF89a")

            def child(command, **kwargs):
                output = Path(command[command.index("--video_output")+1])
                output.write_bytes(b"MP4 original")
                self.assertEqual(command[command.index("--num_envs")+1], "1")
                self.assertTrue((Path(folder)/"model_2.pt").is_file())
                return subprocess.CompletedProcess(command, 0)

            with patch.object(VIDEO["subprocess"], "run", side_effect=child) as render, \
                    patch.object(WRITER["SwanLabSummaryWriter"], "_video_preview", return_value=preview), \
                    contextlib.redirect_stdout(io.StringIO()):
                runner.learn(2)
            self.assertEqual(render.call_count, 1)
            self.assertTrue((Path(folder)/"model_3.pt").is_file())
            self.assertEqual(self.sdk.Video.call_count, 1)
            media = [call for call in self.runs[0].log.call_args_list if "video" in call.args[0]]
            self.assertEqual(len(media), 1)
            self.assertEqual(media[0].kwargs["step"], 1)  # matches scalar iteration of model_2
            self.assertEqual(runner.env.num_envs, 8)
            self.assertFalse(runner.env.cfg.env.enable_camera_sensors)

    def test_child_failures_and_timeouts_do_not_stop_training_or_retry_final_save(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("Q1_TEST_TMP")) as folder:
            recorder = VIDEO["CheckpointVideoRecorder"]("cpu", "cpu", folder, {"interval": 2})
            with patch.object(VIDEO["subprocess"], "run", side_effect=subprocess.TimeoutExpired("render", 1)) as child, \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                recorder.on_checkpoint(str(Path(folder)/"model_2.pt"), 2)
                recorder.on_checkpoint(str(Path(folder)/"model_2.pt"), 2)
            self.assertEqual(child.call_count, 1)
            self.assertIn("training continues", output.getvalue())
            self.assertFalse(recorder.should_record(0))
            self.assertFalse(recorder.should_record(3))

    def test_upload_retry_and_dedup_preserve_local_media(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("Q1_TEST_TMP")) as folder:
            writer = WRITER["SwanLabSummaryWriter"](folder, 10, {"swanlab_project": "Q1"})
            self.addCleanup(writer.close)
            video = Path(folder)/"clip.mp4"
            video.write_bytes(b"original")
            preview = video.with_suffix(".gif")
            preview.write_bytes(b"GIF89a")
            self.runs[0].log.side_effect = [RuntimeError("network unavailable"), None]
            with patch.object(writer, "_video_preview", return_value=preview), \
                    patch.object(WRITER["time"], "monotonic", return_value=100), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(writer.save_video(video, 10))
                self.assertFalse(writer.save_video(video, 11))
            self.assertEqual(self.runs[0].log.call_count, 1)
            with patch.object(writer, "_video_preview", return_value=preview), \
                    patch.object(WRITER["time"], "monotonic", return_value=161):
                self.assertTrue(writer.save_video(video, 12))
                self.assertFalse(writer.save_video(video, 13))
            self.assertEqual(self.runs[0].log.call_count, 2)
            self.assertEqual(video.read_bytes(), b"original")
            self.assertTrue(preview.is_file())

    def test_headless_camera_is_enabled_only_for_recording_child(self):
        gym = SimpleNamespace(prepare_sim=lambda sim: None)
        api = SimpleNamespace(acquire_gym=lambda: gym)
        namespace = dict(NS, gymapi=api, gymutil=SimpleNamespace(parse_device_str=lambda device: ("cuda", 2)))
        load_code("wheel_legged_gym/envs/base/base_task.py", namespace)
        BaseTask = namespace["BaseTask"]

        class Task(BaseTask):
            def create_sim(self):
                self.sim = object()

        cfg = NS["Q1Cfg"]()
        cfg.env.num_envs = 1
        task = Task(cfg, SimpleNamespace(use_gpu_pipeline=False), None, "cuda:2", True)
        self.assertEqual(task.graphics_device_id, -1)
        cfg.env.enable_camera_sensors = True
        task = Task(cfg, SimpleNamespace(use_gpu_pipeline=False), None, "cuda:2", True)
        self.assertEqual(task.graphics_device_id, 2)

    def test_checkpoint_entry_uses_snapshot_and_one_camera_env_without_cloud(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("Q1_TEST_TMP")) as folder:
            checkpoint = Path(folder)/"model_1500.pt"
            snapshot = {"stage": "balance", "env": NS["class_to_dict"](NS["Q1Cfg"]()),
                        "train": NS["class_to_dict"](NS["Q1CfgPPO"]())}
            (Path(folder)/"resolved_config.json").write_text(json.dumps(snapshot))
            cfg, train = NS["Q1Cfg"](), NS["Q1CfgPPO"]()
            args = SimpleNamespace(task="q1", checkpoint_path=str(checkpoint), video_output=str(Path(folder)/"video.mp4"))

            class CameraReady(Exception):
                pass

            def make_env(task, **kwargs):
                self.assertEqual(kwargs["env_cfg"].env.num_envs, 1)
                self.assertTrue(kwargs["env_cfg"].env.enable_camera_sensors)
                self.assertEqual(kwargs["env_cfg"].terrain.mesh_type, "plane")
                self.assertFalse(kwargs["env_cfg"].terrain.curriculum)
                self.assertTrue(args.headless)
                raise CameraReady()

            scope = dict(NS, json=json, Path=Path, video_settings=VIDEO["video_settings"],
                         get_args=lambda **kw: args, task_registry=SimpleNamespace(
                             get_cfgs=lambda task: (cfg, train), make_env=make_env))
            load_code("wheel_legged_gym/scripts/record_checkpoint.py", scope)
            with self.assertRaises(CameraReady):
                scope["main"]()
            self.sdk.init.assert_not_called()

    def test_real_camera_stream_to_mp4_and_gif_with_ffmpeg(self):
        try:
            from PIL import Image
            VIDEO["ffmpeg_executable"]()
        except (ImportError, RuntimeError):
            self.skipTest("Pillow and ffmpeg/imageio-ffmpeg are required")
        with tempfile.TemporaryDirectory(dir=os.environ.get("Q1_TEST_TMP")) as folder:
            env = improved(1)
            env.reset = lambda: None
            frames, commands = [], []
            env.gym.create_camera_sensor = lambda *a: 1
            env.gym.destroy_camera_sensor = Mock()
            env.gym.set_camera_location = Mock()
            env.gym.step_graphics = lambda *a: None
            env.gym.render_all_camera_sensors = lambda *a: None

            def image(*args):
                frames.append(1)
                rgba = np.zeros((720, 1280, 4), dtype=np.uint8)
                rgba[:, :, 0] = len(frames)*20
                rgba[:, :, 1] = (np.arange(1280)%256).astype(np.uint8)
                rgba[:, :, 3] = 255
                return rgba.reshape(720, -1)

            env.gym.get_camera_image = image
            original = env.set_command_target
            env.set_command_target = lambda vx, yaw: (commands.append((vx, yaw)), original(vx, yaw))
            api = SimpleNamespace(CameraProperties=SimpleNamespace, Vec3=lambda *xyz: xyz, IMAGE_COLOR=0)
            video = Path(folder)/"stream.mp4"
            cfg = {"width": 1280, "height": 720, "fps": 25,
                   "command_schedule": [(0.2, 0.2, 0.4), (0.2, -0.2, -0.4)]}
            with fake_gym(api):
                count = VIDEO["capture_video"](env, lambda obs, history: (torch.zeros(1, 6), None), video, cfg, "balance")
            self.assertEqual(count, 10)
            self.assertEqual(set(commands), {(0.0, 0.0)})
            env.gym.destroy_camera_sensor.assert_called_once()
            original_bytes = video.read_bytes()
            writer = WRITER["SwanLabSummaryWriter"](folder, 10, {"swanlab_project": "Q1"})
            self.addCleanup(writer.close)
            self.assertTrue(writer.save_video(video, 1499))
            preview = video.with_suffix(".swanlab.gif")
            with Image.open(preview) as gif:
                self.assertEqual((gif.format, gif.size, gif.n_frames), ("GIF", (640, 360), 4))
                self.assertEqual(gif.info["duration"], 100)
            self.assertEqual(video.read_bytes(), original_bytes)
            self.assertFalse(preview.with_suffix(".tmp.gif").exists())
            self.sdk.Video.assert_called_once_with(str(preview), caption="stream.mp4")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
