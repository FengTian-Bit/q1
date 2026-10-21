# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# SPDX-License-Identifier: BSD-3-Clause
"""SwanLab scalar/config logging adapted from Dreamwaq-highplatform's writer."""
import os
import subprocess
import time
from pathlib import Path
from dataclasses import asdict, is_dataclass

from torch.utils.tensorboard import SummaryWriter
from .video import ffmpeg_executable


class SwanLabSummaryWriter(SummaryWriter):
    """SwanLab training curves with a local TensorBoard backup."""

    def __init__(self, log_dir, flush_secs, cfg):
        try:
            import swanlab
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "SwanLab logging requires swanlab. Install project dependencies with "
                "pip install -e . or use --logger=tensorboard."
            ) from None
        self._sdk = swanlab
        self._stopped = False
        self.logged_videos = set()
        self._video_retry_after = {}
        self.video_fps = int(cfg.get("swanlab_video_fps", 10))
        self.video_width = int(cfg.get("swanlab_video_width", 640))
        if self.video_fps <= 0 or self.video_width <= 0:
            raise ValueError("SwanLab video fps and width must be positive")
        super().__init__(log_dir=log_dir, flush_secs=flush_secs)
        try:
            self.run = swanlab.init(
                project=cfg.get("swanlab_project", cfg.get("experiment_name", "wheel_legged_gym")),
                workspace=cfg.get("swanlab_workspace") or os.getenv("SWANLAB_WORKSPACE"),
                name=os.path.basename(os.path.normpath(log_dir)),
                group=cfg.get("swanlab_group") or cfg.get("run_name"),
                tags=cfg.get("swanlab_tags", []),
                mode=os.getenv("SWANLAB_MODE", cfg.get("swanlab_mode", "online")),
                logdir=os.path.join(log_dir, "swanlog"),
                config={"log_dir": log_dir},
            )
            if self.run is None:
                raise RuntimeError("swanlab.init returned no experiment")
        except BaseException:
            super().close()
            raise

    @property
    def run_id(self):
        return getattr(self.run, "id", None) or getattr(getattr(self.run, "public", None), "run_id", None)

    def store_config(self, env_cfg, train_cfg, resume_context=None):
        self.run.config.update({
            "env_cfg": self._to_config(env_cfg),
            "train_cfg": self._to_config(train_cfg),
            "resume": self._to_config(resume_context or {}),
        })

    def _to_config(self, obj):
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, Path):
            return str(obj)
        if is_dataclass(obj) and not isinstance(obj, type):
            return self._to_config(asdict(obj))
        if isinstance(obj, dict):
            return {str(key): self._to_config(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._to_config(value) for value in obj]
        if hasattr(obj, "__dict__"):
            return {key: self._to_config(getattr(obj, key)) for key in dir(obj)
                    if not key.startswith("_") and not callable(getattr(obj, key))}
        return str(obj)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None,
                   new_style=False, double_precision=False):
        super().add_scalar(tag, scalar_value, global_step=global_step, walltime=walltime,
                           new_style=new_style, double_precision=double_precision)
        value = scalar_value.item() if hasattr(scalar_value, "item") else scalar_value
        self.run.log({tag: float(value)}, step=None if global_step is None else int(global_step))

    def upload_videos(self, log_dir, iteration):
        if log_dir is None:
            return
        for video in sorted((Path(log_dir)/"videos"/"training").glob("*.mp4")):
            self.save_video(video, iteration)

    def save_video(self, video, iteration):
        """Keep full MP4 locally and publish a GIF through SwanLab's media API."""
        video = Path(video)
        key = str(video.resolve())
        if key in self.logged_videos or time.monotonic() < self._video_retry_after.get(key, 0.0):
            return False
        try:
            preview = video if video.suffix.lower() == ".gif" else self._video_preview(video)
            self.run.log({"video": self._sdk.Video(str(preview), caption=video.name)}, step=int(iteration))
        except Exception as exc:
            self._video_retry_after[key] = time.monotonic()+60.0
            detail = (getattr(exc, "stderr", None) or str(exc)).strip()[-1000:]
            print("[SwanLab video] {} failed: {}; retry after 60 s; local video retained".format(video.name, detail), flush=True)
            return False
        self.logged_videos.add(key)
        self._video_retry_after.pop(key, None)
        print("[SwanLab video] uploaded {} at iteration {}".format(preview.name, iteration), flush=True)
        return True

    def _video_preview(self, video):
        preview = video.with_suffix(".swanlab.gif")
        if preview.is_file() and preview.stat().st_size and preview.stat().st_mtime_ns >= video.stat().st_mtime_ns:
            return preview
        temporary = preview.with_suffix(".tmp.gif")
        filters = ("fps={},scale='min({},iw)':-1:flags=lanczos,split[a][b];"
                   "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a").format(self.video_fps, self.video_width)
        try:
            subprocess.run([ffmpeg_executable(), "-y", "-nostdin", "-loglevel", "error", "-i", str(video),
                            "-filter_complex_threads", "1", "-filter_complex", filters,
                            "-an", "-loop", "0", str(temporary)], check=True, capture_output=True,
                           text=True, errors="replace", timeout=120)
            if temporary.stat().st_size == 0:
                raise RuntimeError("ffmpeg produced an empty GIF")
            temporary.replace(preview)
        finally:
            if temporary.exists():
                temporary.unlink()
        return preview

    def close(self):
        if self._stopped:
            return
        self._stopped = True
        try:
            self._sdk.finish()
        finally:
            super().close()
