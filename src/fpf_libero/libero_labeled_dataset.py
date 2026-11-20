"""Load LIBERO episodes and aligned stage-milestone labels.

For stage_chunk labels, the target indicates whether the stage attempted at
the chunk anchor is completed, and is broadcast across the action horizon."""

import glob
import io
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch.utils.data

logger = logging.getLogger("fpf_libero.dataset")

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


class LayoutWeightedSampler(torch.utils.data.Sampler):
    """Sample in the full index space, then map to the distributed trajectories."""

    def __init__(self, weights, positions, size, generator):
        positions = torch.as_tensor(positions, dtype=torch.int64)
        weights = torch.as_tensor(weights, dtype=torch.float64)
        if len(positions) != len(weights) or len(positions) == 0:
            raise ValueError("Sampling positions must match nonempty sample weights")
        if positions[0] < 0 or positions[-1] >= size or torch.any(positions[1:] <= positions[:-1]):
            raise ValueError("Sampling positions must be strictly increasing and within the layout")
        expanded = torch.zeros(size, dtype=torch.float64)
        expanded[positions] = weights
        self.inverse = torch.full((size,), -1, dtype=torch.int64)
        self.inverse[positions] = torch.arange(len(positions))
        self.sampler = torch.utils.data.WeightedRandomSampler(
            expanded, num_samples=size, replacement=True, generator=generator,
        )

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        for index in self.sampler:
            mapped = int(self.inverse[index])
            if mapped < 0:
                raise RuntimeError("Sampler selected a zero-weight undistributed trajectory")
            yield mapped


def compute_stage_completed_for_frame(
    frame_idx: int,
    milestone_indices: list[int],
    milestone_stages: list[int],
    total_stages: int,
    is_success: bool,
) -> bool:
    """与 B1K labeled_data_loader.compute_stage_completed_for_frame_format_b 逐行一致。"""
    if len(milestone_indices) == 0:
        return False

    attempting_stage = 1
    for m_idx, m_stage in zip(milestone_indices, milestone_stages):
        if frame_idx < m_idx:
            attempting_stage = int(m_stage)
            break
        attempting_stage = int(m_stage) + 1

    max_milestone_stage = max(milestone_stages) if milestone_stages else 0
    if attempting_stage <= max_milestone_stage:
        return True
    if attempting_stage == total_stages and is_success:
        return True
    return False


class LiberoLabeledDataset(torch.utils.data.Dataset):
    """帧级样本;__getitem__ 输出经过 openpi transform 管线后的模型输入 dict。"""

    def __init__(
        self,
        data_root: str,
        suites: list[str],
        action_horizon: int,
        transform_fns: list,
        label_mode: str = "stage",
        tasks: list[str] | None = None,
        use_success_only: bool = False,
        expert_root: str | None = None,
        rollout_manifest: str | None = None,
        drop_post_success_anchors: bool = False,
    ):
        self.action_horizon = action_horizon
        self.transform_fns = transform_fns
        if label_mode not in {"stage", "stage_chunk", "episode"}:
            raise ValueError(
                "label_mode must be 'stage', 'stage_chunk', or 'episode', "
                f"got {label_mode!r}"
            )
        self.label_mode = label_mode
        self.drop_post_success_anchors = drop_post_success_anchors

        selected_rollouts: dict[str, set[int]] | None = None
        if rollout_manifest:
            manifest_path = Path(rollout_manifest)
            manifest = json.loads(manifest_path.read_text())
            manifest_tasks = manifest.get("tasks")
            if not isinstance(manifest_tasks, dict) or not manifest_tasks:
                raise ValueError(
                    f"rollout manifest must contain a non-empty 'tasks' mapping: {manifest_path}"
                )
            selected_rollouts = {}
            for task_name, selection in manifest_tasks.items():
                if not isinstance(selection, dict):
                    raise ValueError(
                        f"rollout manifest task entry must be an object: {task_name}"
                    )
                success = selection.get("success", [])
                failure = selection.get("failure", [])
                episode_indices = [*success, *failure]
                if not episode_indices or not all(isinstance(x, int) for x in episode_indices):
                    raise ValueError(
                        f"rollout manifest task must contain integer success/failure ids: {task_name}"
                    )
                if len(set(episode_indices)) != len(episode_indices):
                    raise ValueError(f"duplicate rollout episode id in manifest task {task_name}")
                selected_rollouts[task_name] = set(episode_indices)
            logger.info(
                "Using rollout manifest %s: %d tasks / %d episodes",
                manifest_path,
                len(selected_rollouts),
                sum(len(x) for x in selected_rollouts.values()),
            )


        self.episodes: list[dict] = []

        self.samples: list[tuple[int, int]] = []
        self.sampling_positions: list[int] = []
        self.sampling_size = 0
        self.has_sampling_layout = False


        roots = [(data_root, False)]
        if expert_root:
            roots.append((expert_root, True))

        n_success = n_fail = n_expert = 0
        n_dropped_post_success_anchors = 0
        seen_rollouts: dict[str, set[int]] = {}
        for root, root_is_expert in roots:
            for suite in suites:
                task_dirs = sorted(glob.glob(os.path.join(root, suite, f"{suite}_t*")))
                if not task_dirs:
                    raise FileNotFoundError(f"No task dirs found under {root}/{suite}")
                for tdir in task_dirs:
                    tname = os.path.basename(tdir)
                    if tasks is not None and tname not in tasks:
                        continue
                    layout_path = Path(tdir) / "meta/sampling_layout.json"
                    layout = json.loads(layout_path.read_text()) if layout_path.is_file() else None
                    if layout is not None:
                        if (layout["action_horizon"] != action_horizon
                                or layout["drop_post_success_anchors"] != drop_post_success_anchors
                                or selected_rollouts is not None):
                            raise ValueError("Sampling layout requires its recorded horizon/filter and no rollout manifest")
                        self.has_sampling_layout = True
                    task_sample_start = len(self.samples)
                    task_sampling_base = self.sampling_size
                    selected_ids = (
                        None
                        if root_is_expert or selected_rollouts is None
                        else selected_rollouts.get(tname)
                    )
                    if selected_rollouts is not None and not root_is_expert and selected_ids is None:
                        raise ValueError(
                            f"rollout manifest has no entry for requested task {tname}"
                        )
                    eps_meta = {}
                    with open(os.path.join(tdir, "meta", "episodes.jsonl")) as f:
                        for line in f:
                            d = json.loads(line)
                            eps_meta[d["episode_index"]] = d


                    ep_records = []
                    stages_path = Path(tdir) / "meta/stages.json"
                    recorded_stages = (
                        int(json.loads(stages_path.read_text())["num_stages"])
                        if stages_path.is_file() else None
                    )
                    if recorded_stages is not None and recorded_stages <= 0:
                        raise ValueError(f"{tname}: invalid stage count")
                    total_stages = recorded_stages or 0
                    for pq_path in sorted(glob.glob(os.path.join(tdir, "data", "chunk-000", "*.parquet"))):
                        ep_idx = int(os.path.basename(pq_path).split("_")[1].split(".")[0])
                        if selected_ids is not None and ep_idx not in selected_ids:
                            continue
                        meta = eps_meta[ep_idx]
                        if selected_ids is not None:
                            seen_rollouts.setdefault(tname, set()).add(ep_idx)
                        is_success = bool(meta.get("success"))
                        need_milestones = (
                            self.label_mode in {"stage", "stage_chunk"}
                            or self.drop_post_success_anchors
                        )
                        if need_milestones:
                            rw_path = os.path.join(
                                tdir, "rewards", "chunk-000", f"episode_{ep_idx:06d}.npy"
                            )
                            rewards = np.load(rw_path)
                            milestone_indices = np.where(rewards >= 0.99)[0].tolist()
                            if (
                                is_success
                                and self.label_mode in {"stage", "stage_chunk"}
                            ):
                                total_stages = max(total_stages, len(milestone_indices))
                        else:
                            milestone_indices = []
                        ep_records.append((pq_path, meta, is_success, milestone_indices))
                    if selected_ids is not None:
                        missing = selected_ids - seen_rollouts.get(tname, set())
                        if missing:
                            raise ValueError(
                                f"rollout manifest references missing episodes for {tname}: "
                                f"{sorted(missing)}"
                            )
                    if recorded_stages is not None and total_stages != recorded_stages:
                        raise ValueError(f"{tname}: stage count contradicts labels")
                    if self.label_mode in {"stage", "stage_chunk"} and total_stages == 0:
                        raise ValueError(f"{tname}: no success episode found, cannot infer num_stages")
                    if self.label_mode == "episode":
                        total_stages = 1

                    for pq_path, meta, is_success, milestone_indices in ep_records:
                        if use_success_only and not is_success:
                            continue
                        if is_success:
                            n_success += 1
                        else:
                            n_fail += 1
                        if root_is_expert:
                            n_expert += 1
                        ep_id = len(self.episodes)
                        self.episodes.append(
                            dict(
                                parquet=pq_path,
                                prompt=meta["tasks"][0],
                                is_success=is_success,
                                length=int(meta["length"]),
                                milestone_indices=milestone_indices,
                                milestone_stages=list(range(1, len(milestone_indices) + 1)),
                                total_stages=total_stages,
                                task_name=tname,
                                is_expert=root_is_expert,
                                rollout_failure_source=(
                                    None
                                    if root_is_expert or is_success
                                    else meta.get("rollout_failure_source")
                                ),
                            )
                        )
                        valid_frames = max(
                            int(meta["length"]) - action_horizon,
                            0,
                        )
                        if self.drop_post_success_anchors and is_success:
                            if not milestone_indices:
                                raise ValueError(
                                    f"{tname} episode {meta['episode_index']}: "
                                    "successful episode has no milestone, cannot "
                                    "drop post-success anchors"
                                )
                            filtered_frames = min(
                                valid_frames,
                                max(int(milestone_indices[-1]), 0),
                            )
                            n_dropped_post_success_anchors += (
                                valid_frames - filtered_frames
                            )
                            valid_frames = filtered_frames
                        if layout is not None:
                            entry = layout["episodes"][str(meta["episode_index"])]
                            if entry["count"] != valid_frames:
                                raise ValueError(f"{tname}: sampling layout anchor count mismatch")
                            sample_offset = entry["offset"]
                        else:
                            sample_offset = len(self.samples) - task_sample_start
                        self.sampling_positions.extend(range(
                            task_sampling_base + sample_offset,
                            task_sampling_base + sample_offset + valid_frames,
                        ))
                        for frame_idx in range(valid_frames):
                            self.samples.append((ep_id, frame_idx))
                    self.sampling_size += (
                        layout["num_anchors"] if layout is not None
                        else len(self.samples) - task_sample_start
                    )

        self._cache_ep_id = None
        self._cache_table = None
        logger.info(
            f"LiberoLabeledDataset: {len(self.episodes)} episodes "
            f"({n_success} success / {n_fail} fail), {len(self.samples)} frame samples, "
            f"label_mode={self.label_mode}"
        )
        if expert_root:
            logger.info(f"  其中专家 episodes {n_expert} (expert_root={expert_root})")
        if self.drop_post_success_anchors:
            logger.info(
                "  Dropped %d anchors at/after the final success milestone",
                n_dropped_post_success_anchors,
            )

    def __len__(self) -> int:
        return len(self.samples)


    def make_weighted_sampler(self, weights, generator):
        if self.has_sampling_layout:
            return LayoutWeightedSampler(weights, self.sampling_positions, self.sampling_size, generator)
        return torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(self), replacement=True, generator=generator,
        )


    def _frame_s_target(self, ep: dict, frame_idx: int) -> bool:
        if self.label_mode == "episode":
            return bool(ep["is_success"])
        return compute_stage_completed_for_frame(
            frame_idx, ep["milestone_indices"], ep["milestone_stages"], ep["total_stages"], ep["is_success"]
        )

    def progress_targets_for_chunk(self, ep: dict, frame_idx: int) -> np.ndarray:
        """Return binary targets [H,1] under the configured label granularity."""
        if self.label_mode == "episode":
            return np.full(
                (self.action_horizon, 1),
                1.0 if ep["is_success"] else 0.0,
                dtype=np.float32,
            )
        if self.label_mode == "stage_chunk":
            return np.full(
                (self.action_horizon, 1),
                1.0 if self._frame_s_target(ep, frame_idx) else 0.0,
                dtype=np.float32,
            )


        targets = []
        for i in range(self.action_horizon):
            future_frame_idx = min(frame_idx + i, ep["length"] - 1)
            targets.append(1.0 if self._frame_s_target(ep, future_frame_idx) else 0.0)
        return np.array(targets, dtype=np.float32)[:, None]

    def get_sample_weights_by_outcome(self, target_success_ratio: float) -> torch.Tensor:
        """按轨迹成败平衡帧采样:成功轨迹的帧在每个 batch 中期望占 target_success_ratio。"""
        is_succ = np.array([self.episodes[ep_id]["is_success"] for ep_id, _ in self.samples])
        n_pos = int(is_succ.sum())
        n_neg = len(is_succ) - n_pos
        if n_pos == 0 or n_neg == 0:
            logger.warning("Outcome sampling: only one class present, using uniform weights")
            return torch.ones(len(self.samples), dtype=torch.double)
        w_pos = target_success_ratio / n_pos
        w_neg = (1.0 - target_success_ratio) / n_neg
        logger.info(
            f"[Outcome Sampling] 成功帧 {n_pos} ({n_pos/len(is_succ):.1%}) -> 采样占比 {target_success_ratio:.0%}, "
            f"失败帧 {n_neg} ({n_neg/len(is_succ):.1%}) -> 采样占比 {1-target_success_ratio:.0%}"
        )
        weights = np.where(is_succ, w_pos, w_neg)
        return torch.from_numpy(weights.astype(np.float64))

    def get_sample_weights_by_source(self, expert_ratio: float, rollout_success_ratio: float) -> torch.Tensor:

        if not (0.0 <= expert_ratio <= 1.0 and 0.0 <= rollout_success_ratio <= 1.0):
            raise ValueError(
                f"ratios must be in [0,1]: expert_ratio={expert_ratio}, "
                f"rollout_success_ratio={rollout_success_ratio}"
            )
        is_exp = np.array([bool(self.episodes[ep_id].get("is_expert")) for ep_id, _ in self.samples])
        is_succ = np.array([self.episodes[ep_id]["is_success"] for ep_id, _ in self.samples])
        n_expert = int(is_exp.sum())
        n_roll_succ = int((~is_exp & is_succ).sum())
        n_roll_fail = int((~is_exp & ~is_succ).sum())
        if n_expert == 0 or (n_roll_succ + n_roll_fail) == 0:
            logger.warning("Source sampling: expert 或 rollout 桶为空, 退回均匀采样")
            return torch.ones(len(self.samples), dtype=torch.double)
        mass_succ = (1.0 - expert_ratio) * rollout_success_ratio
        mass_fail = (1.0 - expert_ratio) * (1.0 - rollout_success_ratio)
        if n_roll_succ == 0:
            logger.warning("Source sampling: rollout 无成功帧, 其质量全部给失败帧")
            mass_fail += mass_succ
            mass_succ = 0.0
        elif n_roll_fail == 0:
            logger.warning("Source sampling: rollout 无失败帧, 其质量全部给成功帧")
            mass_succ += mass_fail
            mass_fail = 0.0
        w_exp = expert_ratio / n_expert
        w_succ = mass_succ / n_roll_succ if n_roll_succ else 0.0
        w_fail = mass_fail / n_roll_fail if n_roll_fail else 0.0
        logger.info(
            f"[Source Sampling] 专家帧 {n_expert} -> 采样占比 {expert_ratio:.0%}, "
            f"rollout 成功帧 {n_roll_succ} -> {mass_succ:.0%}, "
            f"rollout 失败帧 {n_roll_fail} -> {mass_fail:.0%}"
        )
        weights = np.where(is_exp, w_exp, np.where(is_succ, w_succ, w_fail))
        return torch.from_numpy(weights.astype(np.float64))

    def get_sample_weights_by_task_source_outcome_episode(
        self,
        expert_ratio: float,
        rollout_success_ratio: float,
        rollout_new_failure_ratio: float | None = None,
    ) -> torch.Tensor:

        if not 0.0 <= expert_ratio <= 1.0:
            raise ValueError(
                f"expert_ratio must be in [0,1], got {expert_ratio}"
            )
        if not 0.0 <= rollout_success_ratio <= 1.0:
            raise ValueError(
                "rollout_success_ratio must be in [0,1], got "
                f"{rollout_success_ratio}"
            )
        if (
            rollout_new_failure_ratio is not None
            and not 0.0 <= rollout_new_failure_ratio <= 1.0
        ):
            raise ValueError(
                "rollout_new_failure_ratio must be in [0,1] or None, got "
                f"{rollout_new_failure_ratio}"
            )

        episode_sample_indices: dict[int, list[int]] = {}
        for sample_index, (ep_id, _) in enumerate(self.samples):
            episode_sample_indices.setdefault(ep_id, []).append(sample_index)
        if not episode_sample_indices:
            raise ValueError("cannot build hierarchical sampler for empty dataset")

        common_bucket_mass = {
            "expert": expert_ratio,
            "rollout_success": (1.0 - expert_ratio) * rollout_success_ratio,
        }
        rollout_failure_mass = (
            (1.0 - expert_ratio) * (1.0 - rollout_success_ratio)
        )
        use_failure_source = rollout_new_failure_ratio is not None
        bucket_names = ["expert", "rollout_success"]
        if use_failure_source:
            bucket_names.extend(
                ["rollout_failure_original", "rollout_failure_new"]
            )
        else:
            bucket_names.append("rollout_failure")

        task_buckets: dict[str, dict[str, list[int]]] = {}
        for ep_id in sorted(episode_sample_indices):
            ep = self.episodes[ep_id]
            task_name = ep["task_name"]
            if ep.get("is_expert"):
                bucket = "expert"
            elif ep["is_success"]:
                bucket = "rollout_success"
            elif use_failure_source:
                failure_source = ep.get("rollout_failure_source")
                if failure_source not in {"original", "new"}:
                    raise ValueError(
                        f"{task_name}: failed rollout is missing valid "
                        "rollout_failure_source metadata; expected original/new, "
                        f"got {failure_source!r}"
                    )
                bucket = f"rollout_failure_{failure_source}"
            else:
                bucket = "rollout_failure"
            task_buckets.setdefault(
                task_name,
                {name: [] for name in bucket_names},
            )[bucket].append(ep_id)

        task_bucket_mass: dict[str, dict[str, float]] = {}
        for task_name, buckets in task_buckets.items():
            masses = dict(common_bucket_mass)
            if use_failure_source:
                has_original = bool(buckets["rollout_failure_original"])
                has_new = bool(buckets["rollout_failure_new"])
                if has_original and has_new:
                    assert rollout_new_failure_ratio is not None
                    masses["rollout_failure_new"] = (
                        rollout_failure_mass * rollout_new_failure_ratio
                    )
                    masses["rollout_failure_original"] = (
                        rollout_failure_mass
                        * (1.0 - rollout_new_failure_ratio)
                    )
                elif has_new:
                    masses["rollout_failure_new"] = rollout_failure_mass
                    masses["rollout_failure_original"] = 0.0
                elif has_original:
                    masses["rollout_failure_new"] = 0.0
                    masses["rollout_failure_original"] = rollout_failure_mass
                elif rollout_failure_mass > 0.0:
                    raise ValueError(
                        f"{task_name}: requested {rollout_failure_mass:.1%} "
                        "failed-rollout mass but no failures are available"
                    )
            else:
                masses["rollout_failure"] = rollout_failure_mass
            task_bucket_mass[task_name] = masses
            for bucket, requested_mass in masses.items():
                if requested_mass > 0.0 and not buckets[bucket]:
                    raise ValueError(
                        f"{task_name}: requested {requested_mass:.1%} mass for "
                        f"empty {bucket} bucket"
                    )

        weights = np.zeros(len(self.samples), dtype=np.float64)
        task_mass = 1.0 / len(task_buckets)
        for task_name, buckets in task_buckets.items():
            for bucket, ep_ids in buckets.items():
                requested_mass = task_bucket_mass[task_name].get(bucket, 0.0)
                if requested_mass == 0.0:
                    continue
                per_episode_mass = task_mass * requested_mass / len(ep_ids)
                for ep_id in ep_ids:
                    sample_indices = episode_sample_indices[ep_id]
                    per_anchor_mass = per_episode_mass / len(sample_indices)
                    weights[sample_indices] = per_anchor_mass

        total_mass = float(weights.sum())
        if not np.isclose(total_mass, 1.0, rtol=1e-9, atol=1e-12):
            raise RuntimeError(
                f"hierarchical mixed-source sampler mass must sum to 1, got {total_mass}"
            )

        actual_mass = {name: 0.0 for name in bucket_names}
        positive_mass = 0.0
        for weight, (ep_id, frame_idx) in zip(weights, self.samples):
            ep = self.episodes[ep_id]
            if ep.get("is_expert"):
                bucket = "expert"
            elif ep["is_success"]:
                bucket = "rollout_success"
            elif use_failure_source:
                failure_source = ep.get("rollout_failure_source")
                bucket = f"rollout_failure_{failure_source}"
            else:
                bucket = "rollout_failure"
            actual_mass[bucket] += float(weight)
            if self._frame_s_target(ep, frame_idx):
                positive_mass += float(weight)

        if use_failure_source:
            logger.info(
                "[Task/Source/Outcome/Episode Sampling] tasks=%d, "
                "expert/rollout-success/original-failure/new-failure="
                "%.1f%%/%.1f%%/%.1f%%/%.1f%%, "
                "anchor label positive/negative=%.2f%%/%.2f%%",
                len(task_buckets),
                100.0 * actual_mass["expert"],
                100.0 * actual_mass["rollout_success"],
                100.0 * actual_mass["rollout_failure_original"],
                100.0 * actual_mass["rollout_failure_new"],
                100.0 * positive_mass,
                100.0 * (1.0 - positive_mass),
            )
        else:
            logger.info(
                "[Task/Source/Outcome/Episode Sampling] tasks=%d, "
                "expert/rollout-success/rollout-failure=%.1f%%/%.1f%%/%.1f%%, "
                "anchor label positive/negative=%.2f%%/%.2f%%",
                len(task_buckets),
                100.0 * actual_mass["expert"],
                100.0 * actual_mass["rollout_success"],
                100.0 * actual_mass["rollout_failure"],
                100.0 * positive_mass,
                100.0 * (1.0 - positive_mass),
            )
        return torch.from_numpy(weights)

    def get_sample_weights_by_task_outcome_episode(
        self,
        target_success_ratio: float,
    ) -> torch.Tensor:

        if not 0.0 < target_success_ratio < 1.0:
            raise ValueError(
                "target_success_ratio must be strictly between 0 and 1, "
                f"got {target_success_ratio}"
            )

        frame_counts: dict[int, int] = {}
        for ep_id, _ in self.samples:
            frame_counts[ep_id] = frame_counts.get(ep_id, 0) + 1
        active_ep_ids = sorted(frame_counts)
        if not active_ep_ids:
            raise ValueError("cannot build hierarchical sampler for empty dataset")
        if any(self.episodes[ep_id].get("is_expert") for ep_id in active_ep_ids):
            raise ValueError(
                "task/outcome/episode sampler is rollout-only, but expert "
                "episodes are present in the critic dataset"
            )

        task_buckets: dict[str, dict[bool, list[int]]] = {}
        for ep_id in active_ep_ids:
            ep = self.episodes[ep_id]
            task_buckets.setdefault(
                ep["task_name"],
                {True: [], False: []},
            )[bool(ep["is_success"])].append(ep_id)

        for task_name, buckets in task_buckets.items():
            if not buckets[True] or not buckets[False]:
                raise ValueError(
                    f"{task_name}: hierarchical critic sampling requires both "
                    "success and failure episodes"
                )

        task_mass = 1.0 / len(task_buckets)
        outcome_mass = {
            True: target_success_ratio,
            False: 1.0 - target_success_ratio,
        }
        episode_weights: dict[int, float] = {}
        for buckets in task_buckets.values():
            for successful, ep_ids in buckets.items():
                per_episode_mass = (
                    task_mass * outcome_mass[successful] / len(ep_ids)
                )
                for ep_id in ep_ids:
                    episode_weights[ep_id] = (
                        per_episode_mass / frame_counts[ep_id]
                    )

        weights = np.asarray(
            [episode_weights[ep_id] for ep_id, _ in self.samples],
            dtype=np.float64,
        )
        total_mass = float(weights.sum())
        if not np.isclose(total_mass, 1.0, rtol=1e-9, atol=1e-12):
            raise RuntimeError(
                f"hierarchical sampler mass must sum to 1, got {total_mass}"
            )

        success_mass = float(
            sum(
                weight
                for weight, (ep_id, _) in zip(weights, self.samples)
                if self.episodes[ep_id]["is_success"]
            )
        )
        positive_mass = float(
            sum(
                weight
                for weight, (ep_id, frame_idx) in zip(weights, self.samples)
                if self._frame_s_target(self.episodes[ep_id], frame_idx)
            )
        )
        logger.info(
            "[Task/Outcome/Episode Sampling] tasks=%d, task mass=%.3f%%, "
            "success/failure=%.1f%%/%.1f%%, anchor label positive/negative="
            "%.2f%%/%.2f%%",
            len(task_buckets),
            100.0 * task_mass,
            100.0 * success_mass,
            100.0 * (1.0 - success_mass),
            100.0 * positive_mass,
            100.0 * (1.0 - positive_mass),
        )
        return torch.from_numpy(weights)

    def get_sample_weights(self, target_ratio: float) -> torch.Tensor:
        """帧级平衡采样权重(正样本 = 当前帧 s_target=1),与 B1K get_sample_weights 一致。"""
        positive = np.zeros(len(self.samples), dtype=bool)
        for i, (ep_id, frame_idx) in enumerate(self.samples):
            positive[i] = self._frame_s_target(self.episodes[ep_id], frame_idx)
        n_pos = int(positive.sum())
        n_neg = len(positive) - n_pos
        if n_pos == 0 or n_neg == 0:
            logger.warning("Balanced sampling: only one class present, using uniform weights")
            return torch.ones(len(self.samples), dtype=torch.double)
        w_pos = target_ratio / n_pos
        w_neg = (1.0 - target_ratio) / n_neg
        logger.info(
            f"[Balanced Sampling] positive={n_pos} ({n_pos/len(positive):.1%}) w={w_pos:.2e}, "
            f"negative={n_neg} ({n_neg/len(positive):.1%}) w={w_neg:.2e}"
        )
        weights = np.where(positive, w_pos, w_neg)
        return torch.from_numpy(weights.astype(np.float64))


    def _load_table(self, ep_id: int):

        if self._cache_ep_id != ep_id:
            import pyarrow.parquet as pq

            self._cache_table = pq.read_table(self.episodes[ep_id]["parquet"])
            self._cache_ep_id = ep_id
        return self._cache_table

    @staticmethod
    def _decode_image(cell: dict) -> np.ndarray:
        from PIL import Image

        img = Image.open(io.BytesIO(cell["bytes"]))
        return np.asarray(img.convert("RGB"), dtype=np.uint8)

    def __getitem__(self, idx: int) -> dict:
        ep_id, frame_idx = self.samples[idx]
        ep = self.episodes[ep_id]
        table = self._load_table(ep_id)

        row_image = table.column("image")[frame_idx].as_py()
        row_wrist = table.column("wrist_image")[frame_idx].as_py()
        state = np.asarray(table.column("state")[frame_idx].as_py(), dtype=np.float32)

        actions = np.stack(
            [
                np.asarray(table.column("actions")[frame_idx + i].as_py(), dtype=np.float32)
                for i in range(self.action_horizon)
            ]
        )

        data = {
            "observation/image": self._decode_image(row_image),
            "observation/wrist_image": self._decode_image(row_wrist),
            "observation/state": state,
            "actions": actions,
            "prompt": ep["prompt"],
        }
        for fn in self.transform_fns:
            data = fn(data)

        data["progress_target"] = self.progress_targets_for_chunk(ep, frame_idx)

        data["is_expert"] = np.float32(1.0 if ep.get("is_expert") else 0.0)
        return data


def collate(batch: list[dict]) -> dict:
    """嵌套 dict 逐叶 stack 成 numpy batch。"""
    out = {}
    first = batch[0]
    for key, val in first.items():
        if isinstance(val, dict):
            out[key] = {k: np.stack([b[key][k] for b in batch]) for k in val}
        else:
            out[key] = np.stack([np.asarray(b[key]) for b in batch])
    return out
