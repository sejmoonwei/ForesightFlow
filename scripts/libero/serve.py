"""Serve LIBERO action candidates with per-decision random keys.

Keys are derived from policy seed, evaluation seed, suite, task, episode and
replan index. Candidate selection supports progress ranking and random choice."""

import argparse
import copy
import hashlib
import dataclasses
import json
import logging
import os
import re
import socket

import jax
import openpi.policies.policy_config as _policy_config
import openpi.training.config as _config
from openpi.serving import websocket_policy_server

from fpf_libero.model import Pi0FPFV31Config


def detect_checkpoint_architecture(
    checkpoint_dir: str,
) -> tuple[bool, int, int, int, int, int, int]:
    """Infer optional FPF leaves from committed checkpoint artifacts."""
    meta_path = os.path.join(checkpoint_dir, "params", "_METADATA")
    metadata_text = ""
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8", errors="ignore") as f:
            metadata_text = f.read()
    separate_progress = "progress_in_proj" in metadata_text


    adapter_hidden_dim = 0
    critic_adapter_depth = 1
    critic_ensemble_size = 1
    critic_feature_layer = 0
    v31_hidden_dim = 256
    v31_depth = 2
    config_path = os.path.join(
        os.path.dirname(checkpoint_dir.rstrip(os.sep)),
        "training_config.json",
    )
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            training_config = json.load(f)
        adapter_hidden_dim = int(
            training_config.get("critic_adapter_hidden_dim", 0)
        )
        critic_adapter_depth = int(
            training_config.get("critic_adapter_depth", 1)
        )
        critic_ensemble_size = int(
            training_config.get("critic_ensemble_size", 1)
        )
        critic_feature_layer = int(
            training_config.get("critic_feature_layer", 0)
        )
        v31_hidden_dim = int(
            training_config.get("v31_flow_critic_hidden_dim", 256)
        )
        v31_depth = int(
            training_config.get("v31_flow_critic_depth", 2)
        )
    elif "critic_adapter_down" in metadata_text:
        metadata = json.loads(metadata_text)
        for key, value in metadata.get("tree_metadata", {}).items():
            if "critic_adapter_down" not in key or "'kernel'" not in key:
                continue
            write_shape = value.get("value_metadata", {}).get("write_shape")
            if write_shape and len(write_shape) == 2:
                adapter_hidden_dim = int(write_shape[-1])
                break
        block_indices = {
            int(index)
            for index in re.findall(
                r"critic_adapter_down_block(\d+)", metadata_text
            )
        }
        if block_indices:
            critic_adapter_depth = max(block_indices) + 1
        head_indices = {
            0 if suffix == "" else int(suffix)
            for suffix in re.findall(
                r"critic_adapter_down(?:_(\d+))?",
                metadata_text,
            )
        }
        if head_indices:
            critic_ensemble_size = max(head_indices) + 1

    return (
        separate_progress,
        adapter_hidden_dim,
        critic_adapter_depth,
        critic_ensemble_size,
        critic_feature_layer,
        v31_hidden_dim,
        v31_depth,
    )


class DecisionSeededPolicy:
    """Reset the underlying policy RNG from request identity on every replan."""

    REQUIRED = (
        "eval_seed",
        "suite",
        "task_id",
        "episode_idx",
        "replan_idx",
    )

    def __init__(self, policy, policy_seed: int):
        self._policy = policy
        self._policy_seed = int(policy_seed)

    @property
    def metadata(self):
        return {
            **self._policy.metadata,
            "v31_decision_seeded": True,
            "v31_policy_seed": self._policy_seed,
        }

    def infer(self, obs):
        request = copy.copy(obs)
        identity = request.pop("__v31_seed_identity", None)
        if not isinstance(identity, dict):
            raise ValueError("V31 request missing __v31_seed_identity")
        missing = [key for key in self.REQUIRED if key not in identity]
        if missing:
            raise ValueError(f"V31 seed identity missing keys: {missing}")
        canonical = "|".join(
            (
                str(self._policy_seed),
                str(int(identity["eval_seed"])),
                str(identity["suite"]),
                str(int(identity["task_id"])),
                str(int(identity["episode_idx"])),
                str(int(identity["replan_idx"])),
            )
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).digest()
        derived_seed = int.from_bytes(digest[:4], "little", signed=False)
        if not hasattr(self._policy, "_rng"):
            raise RuntimeError("V31 deterministic evaluation requires JAX policy")
        self._policy._rng = jax.random.key(derived_seed)
        result = self._policy.infer(request)
        result["v31_seed"] = {
            "derived_seed": derived_seed,
            "identity": identity,
        }
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", type=str, required=True,
                   help="FPF checkpoint step dir (contains params/ and assets/)")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--policy_seed", type=int, required=True)
    p.add_argument("--action_horizon", type=int, default=10)
    p.add_argument("--num_steps", type=int, default=10, help="Flow 积分步数")
    p.add_argument("--num_candidates", type=int, default=1,
                   help=">1 时启用 MC ranking(按 progress 势能选最优候选)")
    p.add_argument(
        "--selection_mode",
        choices=["argmax", "argmin", "random"],
        default="argmax",
        help=(
            "K>1 的候选择方式：argmax 使用 score_mode 的 scorer；"
            "random 均匀随机选择且要求 score_mode=raw"
        ),
    )
    p.add_argument(
        "--score_mode",
        choices=["raw", "residual", "shared_raw", "critic"],
        default="raw",
        help=(
            'Configure score mode.'
        ),
    )
    p.add_argument(
        "--actor_only",
        action="store_true",
        help="加载为原生 Pi0，严格丢弃全部 progress 参数和联合 progress flow",
    )
    p.add_argument(
        "--critic_uncertainty_beta",
        type=float,
        default=0.0,
        help=(
            "ensemble critic LCB 系数：score=mean(logit)-beta*std(logit)；"
            "S4 主评测建议 0.5"
        ),
    )
    p.add_argument(
        "--critic_score_horizon",
        type=int,
        default=0,
        help=(
            'Configure critic score horizon.'
        ),
    )
    p.add_argument(
        "--log_mc_diagnostics",
        action="store_true",
        help=(
            "每次 MC 决策在 server log 记录候选 score、"
            "selected_idx、score_std 和 top_margin"
        ),
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    if args.actor_only and args.num_candidates != 1:
        p.error("--actor_only 只支持 --num_candidates 1（普通 Pi0 没有 critic ranking）")
    if args.actor_only and args.score_mode != "raw":
        p.error("--actor_only 不使用 progress scorer，--score_mode 必须为 raw")
    if args.selection_mode == "random" and args.num_candidates <= 1:
        p.error("--selection_mode random requires --num_candidates > 1")
    if args.selection_mode == "random" and args.score_mode not in {"raw", "shared_raw"}:
        p.error("--selection_mode random requires raw or shared_raw")
    if args.selection_mode == "argmin" and args.num_candidates <= 1:
        p.error("--selection_mode argmin requires --num_candidates > 1")
    if args.selection_mode == "argmin" and args.score_mode != "critic":
        p.error("--selection_mode argmin requires --score_mode critic")
    if args.score_mode in {"shared_raw", "critic"} and args.num_candidates <= 1:
        p.error(
            f"--score_mode {args.score_mode} requires --num_candidates > 1"
        )
    if args.critic_uncertainty_beta < 0:
        p.error("--critic_uncertainty_beta must be non-negative")
    if not 0 <= args.critic_score_horizon <= args.action_horizon:
        p.error(
            "--critic_score_horizon must be in "
            f"[0, {args.action_horizon}]"
        )
    if args.critic_score_horizon > 0 and args.score_mode != "critic":
        p.error("--critic_score_horizon > 0 requires --score_mode critic")
    if args.log_mc_diagnostics and (
        args.score_mode != "critic" or args.num_candidates <= 1
    ):
        p.error(
            "--log_mc_diagnostics requires direct critic scoring with K > 1"
        )
    if (
        args.critic_uncertainty_beta > 0
        and args.score_mode != "critic"
    ):
        p.error(
            "--critic_uncertainty_beta > 0 requires --score_mode critic"
        )

    base_config = _config.get_config("pi05_libero")
    if args.actor_only:


        train_config = dataclasses.replace(
            base_config,
            model=dataclasses.replace(base_config.model, action_horizon=args.action_horizon),
            assets_base_dir=os.environ.get("FPF_ASSETS_DIR", "assets"),
        )
        logging.info("Inference mode: actor_only (vanilla Pi0; all progress parameters excluded)")
    else:

        (
            separate,
            critic_hidden_dim,
            critic_adapter_depth,
            critic_ensemble_size,
            critic_feature_layer,
            v31_hidden_dim,
            v31_depth,
        ) = detect_checkpoint_architecture(args.checkpoint_dir)
        if args.score_mode == "critic":
            p.error("V31 uses joint progress flow; direct BCE critic scoring is disabled")
        if (
            args.critic_uncertainty_beta > 0
            and critic_ensemble_size <= 1
        ):
            p.error(
                "--critic_uncertainty_beta > 0 requires an ensemble "
                "checkpoint with critic_ensemble_size > 1"
            )
        logging.info(
            "Checkpoint architecture: separate_progress_in_proj=%s, "
            "critic_adapter_hidden_dim=%d, critic_adapter_depth=%d, "
            "critic_ensemble_size=%d, critic_feature_layer=%d",
            separate,
            critic_hidden_dim,
            critic_adapter_depth,
            critic_ensemble_size,
            critic_feature_layer,
        )

        train_config = dataclasses.replace(
            base_config,


            model=Pi0FPFV31Config(
                action_horizon=args.action_horizon,
                separate_progress_in_proj=True,
                stop_progress_gradient=True,
                mc_num_candidates=args.num_candidates,
                mc_selection_mode=args.selection_mode,
                mc_remove_initial_progress_noise=args.score_mode == "residual",
                mc_share_progress_noise=args.score_mode == "shared_raw",
                mc_use_critic_score=False,
                critic_adapter_hidden_dim=0,
                critic_adapter_depth=1,
                critic_feature_layer=0,
                critic_ensemble_size=1,
                v31_flow_critic_hidden_dim=v31_hidden_dim,
                v31_flow_critic_depth=v31_depth,
            ),
            assets_base_dir=os.environ.get("FPF_ASSETS_DIR", "assets"),
        )

    sample_kwargs = {"num_steps": args.num_steps}

    policy = _policy_config.create_trained_policy(
        train_config, args.checkpoint_dir, sample_kwargs=sample_kwargs
    )
    policy = DecisionSeededPolicy(policy, args.policy_seed)
    model_name = "vanilla Pi0 actor-only" if args.actor_only else "Pi0FPFV31"
    logging.info(
        f"{model_name} policy loaded from {args.checkpoint_dir} "
        f"(num_steps={args.num_steps}, num_candidates={args.num_candidates}, "
        f"score_mode={args.score_mode}, selection_mode={args.selection_mode}, "
        f"critic_uncertainty_beta={args.critic_uncertainty_beta}, "
        f"critic_score_horizon={args.critic_score_horizon or args.action_horizon}, "
        f"log_mc_diagnostics={args.log_mc_diagnostics})"
    )

    hostname = socket.gethostname()
    logging.info(f"Creating server on {hostname}:{args.port}")
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
