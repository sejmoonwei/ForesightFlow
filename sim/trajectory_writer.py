

import io
import json
import logging
import pathlib
from datetime import datetime
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


_HF_FEATURES_METADATA = {
    "info": {
        "features": {
            "image": {"_type": "Image"},
            "wrist_image": {"_type": "Image"},
            "state": {
                "feature": {"dtype": "float32", "_type": "Value"},
                "length": 8,
                "_type": "Sequence",
            },
            "actions": {
                "feature": {"dtype": "float32", "_type": "Value"},
                "length": 7,
                "_type": "Sequence",
            },
            "timestamp": {"dtype": "float32", "_type": "Value"},
            "frame_index": {"dtype": "int64", "_type": "Value"},
            "episode_index": {"dtype": "int64", "_type": "Value"},
            "index": {"dtype": "int64", "_type": "Value"},
            "task_index": {"dtype": "int64", "_type": "Value"},
        }
    }
}


def _png_encode(arr: np.ndarray) -> bytes:
    """Encode an (H,W,3) uint8 RGB array to PNG bytes."""
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _build_arrow_schema() -> pa.Schema:
    image_struct = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    schema = pa.schema(
        [
            ("image", image_struct),
            ("wrist_image", image_struct),
            ("state", pa.list_(pa.float32(), 8)),
            ("actions", pa.list_(pa.float32(), 7)),
            ("timestamp", pa.float32()),
            ("frame_index", pa.int64()),
            ("episode_index", pa.int64()),
            ("index", pa.int64()),
            ("task_index", pa.int64()),
        ],
        metadata={"huggingface": json.dumps(_HF_FEATURES_METADATA)},
    )
    return schema


class LeRobotTrajectoryWriter:
    """Collects per-step rollout data and flushes one parquet per episode."""

    CHUNK_SIZE = 1000
    IMAGE_HW = (256, 256)
    STATE_DIM = 8
    ACTION_DIM = 7
    ROBOT_TYPE = "panda"
    FPS = 10

    def __init__(self, out_dir: str, run_id: str = "", fps: int = FPS):
        if not run_id:
            run_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_id = run_id
        self.root = pathlib.Path(out_dir) / run_id
        self.fps = fps

        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        (self.root / "data").mkdir(parents=True, exist_ok=True)
        (self.root / "videos").mkdir(parents=True, exist_ok=True)

        self._schema = _build_arrow_schema()
        self._task_to_idx: dict[str, int] = {}
        self._episodes: list[dict] = []
        self._success_eps: list[int] = []
        self._failed_eps: list[int] = []
        self._per_task_stats: dict[str, dict] = {}
        self._global_frame_idx = 0
        self._current_ep_idx = 0
        self._current_task: Optional[str] = None
        self._current_buf: dict[str, list] = {}
        self._finalized = False

        logging.info(
            "LeRobotTrajectoryWriter initialized; output root = %s", self.root
        )


    def video_out_dir(self) -> pathlib.Path:
        """Where main.py should write per-episode .mp4 files."""
        return self.root / "videos"

    def start_episode(self, task_description: str) -> int:
        if self._current_task is not None:
            raise RuntimeError(
                "Previous episode not finalized; call end_episode() first"
            )
        if task_description not in self._task_to_idx:
            self._task_to_idx[task_description] = len(self._task_to_idx)
        self._current_task = task_description
        self._current_buf = {
            "image": [],
            "wrist_image": [],
            "state": [],
            "actions": [],
        }
        return self._current_ep_idx

    def add_step(
        self,
        image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
    ) -> None:
        if self._current_task is None:
            raise RuntimeError("Must call start_episode() before add_step()")

        img = np.ascontiguousarray(image, dtype=np.uint8)
        wrist = np.ascontiguousarray(wrist_image, dtype=np.uint8)
        if img.shape[:2] != self.IMAGE_HW or wrist.shape[:2] != self.IMAGE_HW:
            raise ValueError(
                f"Image shape mismatch: got {img.shape}/{wrist.shape}, "
                f"expected (*, {self.IMAGE_HW}, 3)"
            )
        st = np.asarray(state, dtype=np.float32).reshape(-1)
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        if st.shape != (self.STATE_DIM,):
            raise ValueError(f"state shape {st.shape} != ({self.STATE_DIM},)")
        if act.shape != (self.ACTION_DIM,):
            raise ValueError(f"action shape {act.shape} != ({self.ACTION_DIM},)")

        self._current_buf["image"].append(img)
        self._current_buf["wrist_image"].append(wrist)
        self._current_buf["state"].append(st)
        self._current_buf["actions"].append(act)

    def end_episode(self, success: bool) -> None:
        if self._current_task is None:
            raise RuntimeError("end_episode() called without start_episode()")

        length = len(self._current_buf["image"])
        if length == 0:
            logging.warning(
                "Empty episode %d skipped (no steps recorded)", self._current_ep_idx
            )
            self._current_task = None
            self._current_buf = {}
            return

        ep_idx = self._current_ep_idx
        task = self._current_task
        task_idx = self._task_to_idx[task]

        images = [
            {"bytes": _png_encode(im), "path": f"frame_{i:06d}.png"}
            for i, im in enumerate(self._current_buf["image"])
        ]
        wrists = [
            {"bytes": _png_encode(im), "path": f"frame_{i:06d}.png"}
            for i, im in enumerate(self._current_buf["wrist_image"])
        ]

        cols = {
            "image": images,
            "wrist_image": wrists,
            "state": [s.tolist() for s in self._current_buf["state"]],
            "actions": [a.tolist() for a in self._current_buf["actions"]],
            "timestamp": [float(i / self.fps) for i in range(length)],
            "frame_index": list(range(length)),
            "episode_index": [ep_idx] * length,
            "index": [self._global_frame_idx + i for i in range(length)],
            "task_index": [task_idx] * length,
        }

        table = pa.Table.from_pydict(cols, schema=self._schema)

        chunk_idx = ep_idx // self.CHUNK_SIZE
        chunk_dir = self.root / "data" / f"chunk-{chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        out_path = chunk_dir / f"episode_{ep_idx:06d}.parquet"
        pq.write_table(table, out_path)

        self._episodes.append(
            {
                "episode_index": ep_idx,
                "tasks": [task],
                "length": length,
                "success": bool(success),
            }
        )
        (self._success_eps if success else self._failed_eps).append(ep_idx)
        stat = self._per_task_stats.setdefault(task, {"total": 0, "success": 0})
        stat["total"] += 1
        stat["success"] += int(bool(success))

        self._global_frame_idx += length
        self._current_ep_idx += 1
        self._current_task = None
        self._current_buf = {}

    def finalize(self, extra_summary: Optional[dict] = None) -> None:
        """Write meta/info.json, episodes.jsonl, tasks.jsonl, success indexes, summary."""
        if self._finalized:
            return
        if self._current_task is not None:
            logging.warning(
                "finalize() called with unfinished episode %d; dropping it",
                self._current_ep_idx,
            )
            self._current_task = None
            self._current_buf = {}

        meta_dir = self.root / "meta"
        total_episodes = len(self._episodes)
        total_frames = self._global_frame_idx
        total_tasks = len(self._task_to_idx)
        total_chunks = (total_episodes + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE if total_episodes else 0

        info = {
            "codebase_version": "v2.0",
            "robot_type": self.ROBOT_TYPE,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": total_tasks,
            "total_videos": 0,
            "total_chunks": total_chunks,
            "chunks_size": self.CHUNK_SIZE,
            "fps": self.fps,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "image": {
                    "dtype": "image",
                    "shape": [self.IMAGE_HW[0], self.IMAGE_HW[1], 3],
                    "names": ["height", "width", "channel"],
                },
                "wrist_image": {
                    "dtype": "image",
                    "shape": [self.IMAGE_HW[0], self.IMAGE_HW[1], 3],
                    "names": ["height", "width", "channel"],
                },
                "state": {
                    "dtype": "float32",
                    "shape": [self.STATE_DIM],
                    "names": ["state"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": [self.ACTION_DIM],
                    "names": ["actions"],
                },
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            },
        }
        (meta_dir / "info.json").write_text(json.dumps(info, indent=4))

        with (meta_dir / "episodes.jsonl").open("w") as f:
            for ep in self._episodes:
                f.write(json.dumps(ep) + "\n")

        with (meta_dir / "tasks.jsonl").open("w") as f:
            for task, idx in sorted(self._task_to_idx.items(), key=lambda kv: kv[1]):
                f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

        with (meta_dir / "success_episodes.jsonl").open("w") as f:
            for ep_idx in self._success_eps:
                f.write(json.dumps({"episode_index": ep_idx}) + "\n")
        with (meta_dir / "failed_episodes.jsonl").open("w") as f:
            for ep_idx in self._failed_eps:
                f.write(json.dumps({"episode_index": ep_idx}) + "\n")

        per_task = {
            task: {
                "total": v["total"],
                "success": v["success"],
                "success_rate": (v["success"] / v["total"]) if v["total"] else 0.0,
            }
            for task, v in self._per_task_stats.items()
        }
        overall_total = sum(v["total"] for v in self._per_task_stats.values())
        overall_success = sum(v["success"] for v in self._per_task_stats.values())
        summary = {
            "run_id": self.run_id,
            "total_episodes": total_episodes,
            "total_success": overall_success,
            "total_failed": overall_total - overall_success,
            "overall_success_rate": (
                (overall_success / overall_total) if overall_total else 0.0
            ),
            "per_task": per_task,
        }
        if extra_summary:
            summary.update(extra_summary)
        (self.root / "summary.json").write_text(json.dumps(summary, indent=2))

        self._finalized = True
        logging.info(
            "TrajectoryWriter finalized: %d episodes (%d success / %d failed) in %s",
            total_episodes,
            overall_success,
            overall_total - overall_success,
            self.root,
        )
