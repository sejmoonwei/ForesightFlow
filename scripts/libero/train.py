"""Train the LIBERO actor and progress-flow critic with decoupled AWR."""

import argparse
import dataclasses
import functools
import json
import logging
import os
import platform

import etils.epath as epath
import flax.nnx as nnx
from flax import struct
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import torch
import tqdm

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms
import openpi.models.tokenizer as _tokenizer
from openpi.policies import libero_policy

from fpf_libero.model import Pi0FPFV31, Pi0FPFV31Config
from fpf_libero.libero_labeled_dataset import LiberoLabeledDataset, collate, SUITES


def parse_args():
    p = argparse.ArgumentParser(description='LIBERO policy training and evaluation.')
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--data_root", type=str,
                   default=os.environ.get("FPF_ROLLOUT_DATA", "data/rollouts_stage"))
    p.add_argument("--suites", type=str, nargs="+", default=list(SUITES))
    p.add_argument("--init_checkpoint", type=str,
                   default=os.environ.get("FPF_INIT_CHECKPOINT", "checkpoints/actor_init/params"),
                   help="openpi params dir to initialize from (incremental training)")
    p.add_argument("--norm_stats_dir", type=str,
                   default=os.environ.get("FPF_NORM_STATS_DIR", "assets/pi05_libero"),
                   help="Assets dir containing <asset_id>/norm_stats.json")
    p.add_argument("--asset_id", type=str, default="physical-intelligence/libero")
    p.add_argument("--checkpoint_base_dir", type=str,
                   default=os.environ.get("FPF_CHECKPOINT_BASE_DIR", "outputs/checkpoints"))


    p.add_argument("--batch_size", type=int, default=128, help="全局 batch size")
    p.add_argument("--num_train_steps", type=int, default=8000)
    p.add_argument(
        "--lr_schedule_steps",
        type=int,
        default=None,
        help=(
            'Configure lr schedule steps.'
        ),
    )
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fsdp_devices", type=int, default=8)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=2000)
    p.add_argument(
        "--save_steps",
        type=str,
        default="",
        help=(
            "额外保存的显式 step，使用逗号分隔（例如 3000,5000）；"
            "不改变 --save_interval 和最终 step 的保存行为"
        ),
    )
    p.add_argument(
        "--skip_checkpoint_save",
        action="store_true",
        help="仅用于冒烟/显存验证；不创建或保存 checkpoint",
    )
    p.add_argument("--overwrite", action="store_true")


    p.add_argument("--action_horizon", type=int, default=10, help="必须与 init checkpoint 一致")
    p.add_argument("--num_flow_samples", type=int, default=15,
                   help="每步 flow matching 采样次数(B1K FPF 用 15)")
    p.add_argument(
        "--critic_num_flow_samples",
        type=int,
        default=4,
        help='Configure critic num flow samples.',
    )
    p.add_argument("--ema_decay", type=str, default="0.999",
                   help="EMA decay, 'none' 关闭(推理 params 存 EMA)")

    p.add_argument("--separate_progress_in_proj", action=argparse.BooleanOptionalAction, default=True)


    p.add_argument("--stop_prefix_gradient", action=argparse.BooleanOptionalAction, default=True)


    p.add_argument("--awr_tau", type=float, default=0.5)
    p.add_argument("--awr_clip_max", type=float, default=10.0)
    p.add_argument(
        "--awr_zero_below",
        type=float,
        default=0.0,
        help=(
            "AWR 原始权重小于该阈值时置 0；0 表示关闭。阈值仅在 "
            "awr_warmup_steps 结束后生效"
        ),
    )
    p.add_argument("--awr_warmup_steps", type=int, default=1000,
                   help="纯 BC 预热步数(w=1),之后 AWR 权重生效")


    p.add_argument(
        "--awr_baseline",
        type=str,
        choices=["constant", "critic", "direct_critic", "binary", "expert"],
        default="constant",
        help=(
            'Configure awr baseline.'
        ),
    )


    p.add_argument("--s_hat_k_samples", type=int, default=1)


    p.add_argument("--progress_loss_weight", type=float, default=1.0)


    p.add_argument("--stop_progress_gradient", action="store_true")
    p.add_argument("--v31_flow_critic_hidden_dim", type=int, default=256)
    p.add_argument("--v31_flow_critic_depth", type=int, default=2)
    p.add_argument(
        "--reset_v31_flow_critic",
        action="store_true",
        help='Configure reset flow critic.',
    )
    p.add_argument(
        "--critic_balance_progress_loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="将纯 progress-flow MSE 按 positive/negative stage chunk 各占 0.5",
    )


    p.add_argument(
        "--critic_adapter_hidden_dim",
        type=int,
        default=0,
        help='Configure critic adapter hidden dim.',
    )
    p.add_argument(
        "--critic_adapter_depth",
        type=int,
        default=1,
        help='Configure critic adapter depth.',
    )
    p.add_argument(
        "--critic_update_backbone",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Configure critic update backbone.'
        ),
    )
    p.add_argument(
        "--critic_feature_layer",
        type=int,
        default=0,
        help=(
            'Configure critic feature layer.'
        ),
    )
    p.add_argument(
        "--critic_ensemble_size",
        type=int,
        default=1,
        help=(
            'Configure critic ensemble size.'
        ),
    )
    p.add_argument(
        "--critic_bootstrap_prob",
        type=float,
        default=1.0,
        help=(
            "每个 ensemble head 独立采样每条 chunk 的 Bernoulli 保留概率；"
            "1.0 保持单头/非 bootstrap 行为"
        ),
    )
    p.add_argument(
        "--reset_direct_critic",
        action="store_true",
        help=(
            "加载 actor/progress，但丢弃 checkpoint 中所有 direct-critic "
            "adapter/score leaves，改用当前模型的 fresh initialization"
        ),
    )
    p.add_argument(
        "--critic_clean_loss_weight",
        type=float,
        default=0.0,
        help='Configure critic clean loss weight.',
    )
    p.add_argument(
        "--critic_balance_clean_loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Configure critic balance clean loss.'
        ),
    )


    p.add_argument("--use_differential_lr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--high_lr", type=float, default=1e-5, help="HIGH 组(progress head)LR")
    p.add_argument("--low_lr", type=float, default=1e-7, help="LOW 组(backbone)LR")
    p.add_argument(
        "--critic_backbone_lr",
        type=float,
        default=1e-6,
        help="critic BCE 进入共享 backbone 时使用的无动量 SGD 峰值 LR",
    )

    p.add_argument("--high_warmup_steps", type=int, default=500)
    p.add_argument("--backbone_warmup_delay", type=int, default=2000)
    p.add_argument("--backbone_warmup_ramp", type=int, default=1000)
    p.add_argument(
        "--high_lr_patterns",
        type=str,
        default=(
            "progress_out_proj,progress_in_proj,"
            "critic_adapter,critic_score_out_proj"
        ),
                   help="逗号分隔的 HIGH 组参数路径 pattern")
    p.add_argument("--single_lr", type=float, default=2.5e-5,
                   help="use_differential_lr=False 时的统一 LR")


    p.add_argument("--success_frame_ratio", type=float, default=0.7,
                   help='Configure success frame ratio.')

    p.add_argument("--expert_data_root", type=str, default="",
                   help="专家数据根目录(格式与 data_root 相同, 全 success=true, 如 "
                        "datasets/libero_expert)。非空时启用按来源采样 get_sample_weights_by_source")
    p.add_argument(
        "--rollout_manifest",
        type=str,
        default="",
        help="可选 rollout episode 子集 manifest；仅过滤 data_root，不过滤 expert_data_root",
    )
    p.add_argument(
        "--critic_rollout_only",
        action="store_true",
        help=(
            "critic 自定义 loader 只加载 data_root/rollout_manifest，不把 "
            "expert_data_root 合入 critic batch；separate 模式下 actor 仍使用 "
            "official expert BC loader"
        ),
    )
    p.add_argument(
        "--critic_task_uniform_sampling",
        action="store_true",
        help=(
            "rollout-only critic 按 task -> outcome -> episode -> anchor "
            "分层采样；每任务等质量，outcome 比例由 success_frame_ratio 控制"
        ),
    )
    p.add_argument(
        "--critic_mixed_task_source_sampling",
        action="store_true",
        help=(
            "mixed critic 按 task -> expert/rollout-success/rollout-failure "
            "-> episode -> anchor 分层采样；expert_frame_ratio 控制 expert，"
            "success_frame_ratio 控制剩余 rollout 内成功比例"
        ),
    )
    p.add_argument(
        "--critic_rollout_new_failure_ratio",
        type=float,
        default=None,
        help=(
            'Configure critic rollout new failure ratio.'
        ),
    )
    p.add_argument(
        "--drop_post_success_anchors",
        action="store_true",
        help=(
            "成功轨迹丢弃最终 milestone 当帧及其后的 anchors；episode arm "
            "也读取 milestone 位置执行相同过滤，但 target 仍只由 episode "
            "success/failure 决定"
        ),
    )
    p.add_argument("--expert_frame_ratio", type=float, default=0.5,
                   help="按来源采样: 专家帧在每个 batch 中期望占比(其余为 rollout 帧); "
                        "也是 awr_baseline=expert 的权重归一分母 w=is_expert/expert_frame_ratio")
    p.add_argument(
        "--separate_actor_critic_batches",
        action="store_true",
        help=(
            "actor 使用 official LeRobot 低数据 BC batch；critic 默认使用当前 "
            "expert+rollout mixed batch，配合 --critic_rollout_only 时只使用 "
            "rollout；要求 --stop_progress_gradient 与 --no-stop_prefix_gradient"
        ),
    )
    p.add_argument(
        "--separate_actor_awr",
        action="store_true",
        help=(
            "separate 模式下将 actor 的 official BC loader 替换为带 "
            "progress_target 的 LiberoLabeledDataset，并用 AWR 加权 action "
            "flow loss。critic 仍使用完全独立的 batch/optimizer。"
        ),
    )
    p.add_argument(
        "--actor_mixed_task_source_sampling",
        action="store_true",
        help='Configure actor mixed task source sampling.',
    )
    p.add_argument(
        "--actor_expert_frame_ratio",
        type=float,
        default=1.0,
        help=(
            "separate actor AWR batch 中 expert 的期望帧占比；1.0=仅 "
            "expert，小于1时其余为 rollout。"
        ),
    )
    p.add_argument(
        "--actor_rollout_success_ratio",
        type=float,
        default=0.4,
        help="separate actor AWR 的 rollout 子集内 success 帧占比",
    )
    p.add_argument(
        "--actor_only_control",
        action="store_true",
        help=(
            "严格 matched BC control：在 separate 模式下构造普通 Pi0.5，"
            "只执行 official expert actor microsteps，不构造 progress 参数、"
            "不执行 critic update"
        ),
    )
    p.add_argument(
        "--critic_only",
        action="store_true",
        help=(
            'Configure critic only.'
        ),
    )
    p.add_argument(
        "--actor_batch_size",
        type=int,
        default=128,
        help="separate_actor_critic_batches 模式下 official expert BC 有效 batch size",
    )
    p.add_argument(
        "--actor_grad_accum_steps",
        type=int,
        default=1,
        help=(
            "actor 梯度累积次数；official loader 的 microbatch 为 "
            "actor_batch_size / actor_grad_accum_steps"
        ),
    )
    p.add_argument(
        "--critic_updates_per_actor_step",
        type=int,
        default=1,
        help=(
            "separate 模式下每个 actor/global step 执行的独立 critic update 数；"
            "每次 update 使用新的 critic batch 和随机数，global step/EMA 仍只推进一次"
        ),
    )
    p.add_argument("--actor_episodes_per_task", type=int, default=5)
    p.add_argument("--actor_episodes_seed", type=int, default=42)
    p.add_argument(
        "--critic_use_quantile_norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "critic 自定义 loader 是否使用 quantile normalization；"
            "BC-aligned PI0.5 recipe 与 official actor 一样启用"
        ),
    )
    p.add_argument("--target_success_ratio", type=float, default=None,
                   help="按帧级 s_target 标签平衡(B1K 语义), 与 success_frame_ratio 互斥且优先级更低")
    p.add_argument(
        "--critic_target_positive_ratio",
        type=float,
        default=None,
        help=(
            'Configure critic target positive ratio.'
        ),
    )
    p.add_argument("--use_success_only", action="store_true")
    p.add_argument(
        "--label_mode",
        choices=["stage", "stage_chunk", "episode"],
        default="stage",
        help=(
            'Configure label mode.'
        ),
    )
    p.add_argument(
        "--critic_label_mode",
        choices=["stage", "stage_chunk", "episode"],
        default=None,
        help=(
            "仅覆盖 separate critic loader 的标签模式；默认继承 --label_mode。"
            "用于保持 actor stage AWR 不变，同时让 critic 使用 episode 标签。"
        ),
    )


    p.add_argument("--wandb_enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--project_name", type=str, default="fpf_libero")


    p.add_argument("--smoke_test", action="store_true",
                   help="小 batch 跑 3 步验证管线,不保存 wandb")
    args = p.parse_args()
    if args.awr_baseline == "expert" and not args.expert_data_root:
        p.error("--awr_baseline expert 需要 --expert_data_root(按来源二值权重依赖专家数据源)")
    if not 0.0 < args.actor_expert_frame_ratio <= 1.0:
        p.error("--actor_expert_frame_ratio must be in (0, 1]")
    if not 0.0 <= args.actor_rollout_success_ratio <= 1.0:
        p.error("--actor_rollout_success_ratio must be in [0, 1]")
    if args.awr_tau <= 0:
        p.error("--awr_tau must be positive")
    if args.awr_clip_max <= 0:
        p.error("--awr_clip_max must be positive")
    if not 0.0 <= args.awr_zero_below <= args.awr_clip_max:
        p.error("--awr_zero_below must be in [0, awr_clip_max]")
    if args.awr_baseline == "direct_critic" and not args.separate_actor_awr:
        p.error("--awr_baseline direct_critic requires --separate_actor_awr")
    if args.separate_actor_awr:
        if not args.separate_actor_critic_batches:
            p.error("--separate_actor_awr requires --separate_actor_critic_batches")
        if args.critic_only or args.actor_only_control:
            p.error("--separate_actor_awr is incompatible with critic/actor-only modes")
        if not args.expert_data_root:
            p.error("--separate_actor_awr requires --expert_data_root")
        if args.awr_baseline != "critic":
            p.error("V31 --separate_actor_awr requires original flow --awr_baseline critic")
        if args.progress_loss_weight <= 0:
            p.error("V31 --separate_actor_awr requires progress flow training")
    if (
        args.critic_label_mode is not None
        and not args.separate_actor_critic_batches
    ):
        p.error(
            "--critic_label_mode requires --separate_actor_critic_batches"
        )
    if args.lr_schedule_steps is not None:
        if args.lr_schedule_steps <= 0:
            p.error("--lr_schedule_steps must be positive")
        if args.lr_schedule_steps < args.num_train_steps:
            p.error("--lr_schedule_steps must be >= --num_train_steps")
    try:
        explicit_save_steps = {
            int(value.strip())
            for value in args.save_steps.split(",")
            if value.strip()
        }
    except ValueError as exc:
        p.error(f"--save_steps must be comma-separated integers: {exc}")
    if any(step <= 0 or step >= args.num_train_steps for step in explicit_save_steps):
        p.error(
            "every --save_steps value must be positive and smaller than "
            "--num_train_steps"
        )
    if args.critic_adapter_hidden_dim < 0:
        p.error("--critic_adapter_hidden_dim must be non-negative")
    if args.critic_adapter_depth < 1:
        p.error("--critic_adapter_depth must be at least 1")
    if not 0 <= args.critic_feature_layer <= 18:
        p.error("--critic_feature_layer must be in [0, 18]")
    if 0 < args.critic_feature_layer < 18 and args.critic_update_backbone:
        p.error(
            "intermediate --critic_feature_layer currently requires "
            "--no-critic_update_backbone"
        )
    if args.critic_backbone_lr <= 0:
        p.error("--critic_backbone_lr must be positive")
    if args.critic_update_backbone:
        if not args.separate_actor_critic_batches:
            p.error("--critic_update_backbone requires --separate_actor_critic_batches")
        if args.critic_clean_loss_weight <= 0:
            p.error("--critic_update_backbone requires clean critic BCE")
        if args.progress_loss_weight != 0:
            p.error("--critic_update_backbone currently requires --progress_loss_weight 0")
        if not args.use_differential_lr:
            p.error("--critic_update_backbone requires differential LR")
    if args.critic_ensemble_size < 1:
        p.error("--critic_ensemble_size must be at least 1")
    if not 0.0 < args.critic_bootstrap_prob <= 1.0:
        p.error("--critic_bootstrap_prob must be in (0, 1]")
    if args.critic_ensemble_size > 1 and args.critic_adapter_hidden_dim <= 0:
        p.error(
            "--critic_ensemble_size > 1 requires "
            "--critic_adapter_hidden_dim > 0"
        )
    if (
        args.critic_bootstrap_prob < 1.0
        and args.critic_ensemble_size <= 1
    ):
        p.error(
            "--critic_bootstrap_prob < 1 requires "
            "--critic_ensemble_size > 1"
        )
    if args.critic_clean_loss_weight < 0:
        p.error("--critic_clean_loss_weight must be non-negative")
    if args.progress_loss_weight < 0:
        p.error("--progress_loss_weight must be non-negative")
    if args.critic_clean_loss_weight > 0 and args.critic_adapter_hidden_dim <= 0:
        p.error(
            "--critic_clean_loss_weight > 0 requires "
            "--critic_adapter_hidden_dim > 0"
        )
    if args.critic_ensemble_size > 1 and args.critic_clean_loss_weight <= 0:
        p.error(
            "--critic_ensemble_size > 1 requires "
            "--critic_clean_loss_weight > 0"
        )
    if args.reset_direct_critic and args.critic_adapter_hidden_dim <= 0:
        p.error(
            "--reset_direct_critic requires "
            "--critic_adapter_hidden_dim > 0"
        )
    if args.critic_balance_clean_loss and args.critic_clean_loss_weight <= 0:
        p.error(
            "--critic_balance_clean_loss requires "
            "--critic_clean_loss_weight > 0"
        )
    if args.v31_flow_critic_hidden_dim <= 0:
        p.error("--v31_flow_critic_hidden_dim must be positive")
    if args.v31_flow_critic_depth < 1:
        p.error("--v31_flow_critic_depth must be at least 1")
    if args.critic_num_flow_samples < 1:
        p.error("--critic_num_flow_samples must be at least 1")
    if args.critic_clean_loss_weight != 0:
        p.error("V31 is pure flow-matching: --critic_clean_loss_weight must be 0")
    if args.critic_update_backbone:
        p.error("V31 directional flow critic never updates Actor backbone")
    if (
        args.critic_target_positive_ratio is not None
        and not 0.0 < args.critic_target_positive_ratio < 1.0
    ):
        p.error("--critic_target_positive_ratio must be strictly between 0 and 1")
    if args.critic_rollout_only and not args.separate_actor_critic_batches:
        p.error(
            "--critic_rollout_only requires --separate_actor_critic_batches"
        )
    if (
        args.critic_task_uniform_sampling
        and not args.critic_rollout_only
    ):
        p.error(
            "--critic_task_uniform_sampling requires --critic_rollout_only"
        )
    if (
        args.critic_task_uniform_sampling
        and args.critic_target_positive_ratio is not None
    ):
        p.error(
            "--critic_task_uniform_sampling and "
            "--critic_target_positive_ratio are mutually exclusive"
        )
    if (
        args.critic_task_uniform_sampling
        and not 0.0 < args.success_frame_ratio < 1.0
    ):
        p.error(
            "--critic_task_uniform_sampling requires "
            "0 < --success_frame_ratio < 1"
        )
    if args.critic_mixed_task_source_sampling:
        if not args.separate_actor_critic_batches:
            p.error(
                "--critic_mixed_task_source_sampling requires "
                "--separate_actor_critic_batches"
            )
        if args.critic_rollout_only:
            p.error(
                "--critic_mixed_task_source_sampling is incompatible with "
                "--critic_rollout_only"
            )
        if not args.expert_data_root:
            p.error(
                "--critic_mixed_task_source_sampling requires --expert_data_root"
            )
        if args.critic_task_uniform_sampling:
            p.error(
                "--critic_mixed_task_source_sampling and "
                "--critic_task_uniform_sampling are mutually exclusive"
            )
        if args.critic_target_positive_ratio is not None:
            p.error(
                "--critic_mixed_task_source_sampling and "
                "--critic_target_positive_ratio are mutually exclusive"
            )
        if not 0.0 <= args.expert_frame_ratio <= 1.0:
            p.error("--expert_frame_ratio must be in [0, 1]")
        if not 0.0 <= args.success_frame_ratio <= 1.0:
            p.error("--success_frame_ratio must be in [0, 1]")
    if args.critic_rollout_new_failure_ratio is not None:
        if not args.critic_mixed_task_source_sampling:
            p.error(
                "--critic_rollout_new_failure_ratio requires "
                "--critic_mixed_task_source_sampling"
            )
        if not 0.0 <= args.critic_rollout_new_failure_ratio <= 1.0:
            p.error("--critic_rollout_new_failure_ratio must be in [0, 1]")
    return args


@dataclasses.dataclass(frozen=True)
class FPFCheckpointWeightLoader:
    params_path: str
    reset_direct_critic: bool = False
    reset_v31_flow_critic: bool = False

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(self.params_path, restore_type=np.ndarray)

        flat = traverse_util.flatten_dict(loaded, sep="/")
        flat_target = traverse_util.flatten_dict(params, sep="/")
        if self.reset_direct_critic:
            direct_critic_markers = (
                "critic_adapter_down",
                "critic_adapter_up",
                "critic_score_out_proj",
            )
            reset_keys = [
                key
                for key in flat
                if any(marker in key for marker in direct_critic_markers)
            ]
            for key in reset_keys:
                del flat[key]
            logging.info(
                "Reset direct critic: discarded %d checkpoint leaves; "
                "fresh model initialization will be used",
                len(reset_keys),
            )
        if self.reset_v31_flow_critic:
            reset_keys = [key for key in flat if "v31_flow_critic_" in key]
            for key in reset_keys:
                del flat[key]
            logging.info(
                "Reset V31 flow critic: discarded %d checkpoint leaves",
                len(reset_keys),
            )
        for key in list(flat.keys()):
            if "action_in_proj" in key and key.endswith("kernel"):
                target_shape = flat_target[key].shape
                if flat[key].shape[0] != target_shape[0]:
                    pad_rows = target_shape[0] - flat[key].shape[0]
                    logging.info(
                        f"Expanding {key}: {flat[key].shape} -> {target_shape} "
                        f"(zero-pad {pad_rows} progress input row)"
                    )
                    flat[key] = np.pad(flat[key], ((0, pad_rows), (0, 0)), constant_values=0)
        loaded = traverse_util.unflatten_dict(flat, sep="/")


        missing_regex = (
            ".*(progress_(out|in)_proj|critic_adapter_(down|up)|"
            "critic_score_out_proj|v31_flow_critic_).*"
        )
        return _weight_loaders._merge_params(
            loaded,
            params,
            missing_regex=missing_regex,
        )


def build_param_labels(trainable_params, high_lr_patterns: list[str]):
    high_count = low_count = 0

    def make_label(path, leaf):
        nonlocal high_count, low_count
        path_str = jax.tree_util.keystr(path)
        is_high = any(pat in path_str for pat in high_lr_patterns)
        size = leaf.size if hasattr(leaf, "size") else 1
        if is_high:
            high_count += size
        else:
            low_count += size
        return 1 if is_high else 0

    labels = jax.tree_util.tree_map_with_path(make_label, trainable_params)
    logging.info(f"Param labels: {high_count:,} HIGH LR / {low_count:,} LOW LR")
    return labels


def _adamw(schedule):
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=schedule,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=1e-10,
        ),
    )


def _sgd(schedule):
    """Stateless backbone optimizer for the critic branch."""
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.sgd(learning_rate=schedule, momentum=None),
    )


def create_optimizer(args, param_labels):

    schedule_steps = args.lr_schedule_steps or args.num_train_steps
    decay_steps = max(schedule_steps, args.high_warmup_steps + 1)

    if not args.use_differential_lr:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=args.single_lr,
            warmup_steps=args.high_warmup_steps,
            decay_steps=decay_steps, end_value=args.single_lr * 0.1,
        )
        return _adamw(schedule)

    high_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=args.high_lr,
        warmup_steps=args.high_warmup_steps,
        decay_steps=decay_steps, end_value=args.high_lr * 0.01,
    )
    remaining = max(
        schedule_steps - args.backbone_warmup_delay - args.backbone_warmup_ramp,
        1,
    )
    low_schedule = optax.join_schedules(
        schedules=[
            optax.constant_schedule(0.0),
            optax.linear_schedule(0.0, args.low_lr, args.backbone_warmup_ramp),
            optax.cosine_decay_schedule(init_value=args.low_lr, decay_steps=remaining, alpha=0.01),
        ],
        boundaries=[args.backbone_warmup_delay, args.backbone_warmup_delay + args.backbone_warmup_ramp],
    )
    return optax.multi_transform(
        transforms={1: _adamw(high_schedule), 0: _adamw(low_schedule)},
        param_labels=param_labels,
    )


def create_separate_optimizers(args, critic_param_labels=None):
    """Build disjoint actor/critic optimizers for the BC-aligned recipe."""
    schedule_steps = args.lr_schedule_steps or args.num_train_steps
    decay_steps = max(schedule_steps, args.high_warmup_steps + 1)

    def with_actor_accum(tx):
        if args.actor_grad_accum_steps <= 1:
            return tx
        return optax.MultiSteps(
            tx,
            every_k_schedule=args.actor_grad_accum_steps,
            use_grad_mean=True,
        )

    if not args.use_differential_lr:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=args.single_lr,
            warmup_steps=args.high_warmup_steps,
            decay_steps=decay_steps,
            end_value=args.single_lr * 0.1,
        )
        return with_actor_accum(_adamw(schedule)), _adamw(schedule)

    critic_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.high_lr,
        warmup_steps=args.high_warmup_steps,
        decay_steps=decay_steps,
        end_value=args.high_lr * 0.01,
    )
    remaining = max(
        schedule_steps - args.backbone_warmup_delay - args.backbone_warmup_ramp,
        1,
    )
    actor_schedule = optax.join_schedules(
        schedules=[
            optax.constant_schedule(0.0),
            optax.linear_schedule(0.0, args.low_lr, args.backbone_warmup_ramp),
            optax.cosine_decay_schedule(
                init_value=args.low_lr,
                decay_steps=remaining,
                alpha=0.01,
            ),
        ],
        boundaries=[
            args.backbone_warmup_delay,
            args.backbone_warmup_delay + args.backbone_warmup_ramp,
        ],
    )
    critic_tx = _adamw(critic_schedule)
    if critic_param_labels is not None:
        critic_backbone_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=args.critic_backbone_lr,
            warmup_steps=args.high_warmup_steps,
            decay_steps=decay_steps,
            end_value=args.critic_backbone_lr * 0.01,
        )


        critic_tx = optax.multi_transform(
            transforms={
                "head": _adamw(critic_schedule),
                "backbone": _sgd(critic_backbone_schedule),
            },
            param_labels=critic_param_labels,
        )
    return with_actor_accum(_adamw(actor_schedule)), critic_tx


@dataclasses.dataclass(frozen=True)
class AWRConfig:
    tau: float = 0.5
    clip_max: float = 10.0
    zero_below: float = 0.0
    warmup_steps: int = 1000
    num_flow_samples: int = 1
    critic_num_flow_samples: int = 4
    critic_balance_progress_loss: bool = True
    stop_prefix_gradient: bool = True
    baseline: str = "constant"
    s_hat_k_samples: int = 1
    expert_frame_ratio: float = 0.5
    progress_loss_weight: float = 1.0
    critic_clean_loss_weight: float = 0.0
    critic_balance_clean_loss: bool = False
    critic_ensemble_size: int = 1
    critic_bootstrap_prob: float = 1.0
    critic_update_backbone: bool = False


@dataclasses.dataclass(frozen=True)
class PathComponentStartsWith:
    """NNX filter matching a module-name prefix within a graph path."""

    prefix: str

    def __call__(self, path, _value) -> bool:
        return any(
            isinstance(component, str)
            and component.startswith(self.prefix)
            for component in path
        )


V31_FLOW_CRITIC_PATH_FILTER = nnx.All(
    nnx.Param,
    PathComponentStartsWith("v31_flow_critic_"),
)

ACTOR_PARAM_FILTER = nnx.All(
    nnx.Param,
    nnx.Not(
        nnx.Any(
            nnx.PathContains("progress_in_proj"),
            nnx.PathContains("progress_out_proj"),
            PathComponentStartsWith("critic_adapter_down"),
            PathComponentStartsWith("critic_adapter_up"),
            PathComponentStartsWith("critic_score_out_proj"),
            PathComponentStartsWith("v31_flow_critic_"),
        )
    ),
)


def critic_param_filter(
    progress_loss_weight: float,
    clean_loss_weight: float,
    update_backbone: bool = False,
):

    if progress_loss_weight <= 0:
        raise ValueError("V31 critic requires positive progress flow loss")
    if clean_loss_weight != 0:
        raise ValueError("V31 does not train the direct BCE critic")
    if update_backbone:
        raise ValueError("V31 flow critic cannot update Actor backbone")
    return V31_FLOW_CRITIC_PATH_FILTER


@struct.dataclass
class SeparateTrainState:
    """Train state with one optimizer per disjoint parameter set."""

    step: object
    params: nnx.State
    model_def: nnx.GraphDef
    actor_opt_state: object
    critic_opt_state: object
    actor_tx: object = struct.field(pytree_node=False)
    critic_tx: object = struct.field(pytree_node=False)
    ema_decay: float | None = struct.field(pytree_node=False)
    ema_params: nnx.State | None = None


def init_train_state(args, model_config: Pi0FPFV31Config, init_rng, mesh):


    params_shape = jax.eval_shape(lambda rng: nnx.state(model_config.create(rng)), init_rng)
    if args.separate_actor_critic_batches:
        selected_critic_filter = critic_param_filter(
            args.progress_loss_weight,
            args.critic_clean_loss_weight,
            args.critic_update_backbone,
        )
        actor_params_shape = params_shape.filter(ACTOR_PARAM_FILTER)
        critic_params_shape = params_shape.filter(selected_critic_filter)
        actor_count = sum(x.size for x in jax.tree.leaves(actor_params_shape))
        critic_count = sum(x.size for x in jax.tree.leaves(critic_params_shape))
        logging.info(
            "Separate optimizer params: %s actor / %s critic",
            f"{actor_count:,}",
            f"{critic_count:,}",
        )
        critic_param_labels = None
        if args.critic_update_backbone:
            def critic_optimizer_label(path, _leaf):
                path_string = jax.tree_util.keystr(path)
                is_head = any(
                    token in path_string
                    for token in ("critic_adapter_", "critic_score_out_proj")
                )
                return "head" if is_head else "backbone"

            critic_param_labels = jax.tree_util.tree_map_with_path(
                critic_optimizer_label, critic_params_shape
            )
        actor_tx, critic_tx = create_separate_optimizers(
            args, critic_param_labels=critic_param_labels
        )
        if args.critic_only:
            actor_tx = None
            logging.info(
                "Critic-only state: actor optimizer state is not allocated"
            )
    else:
        param_labels = build_param_labels(
            params_shape.filter(nnx.Param),
            [s.strip() for s in args.high_lr_patterns.split(",") if s.strip()],
        )
        tx = create_optimizer(args, param_labels)

    def init(rng, partial_params=None):
        rng, model_rng = jax.random.split(rng)
        model = model_config.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        if args.separate_actor_critic_batches:
            return SeparateTrainState(
                step=0,
                params=params,
                model_def=nnx.graphdef(model),
                actor_opt_state=(
                    None
                    if args.critic_only
                    else actor_tx.init(params.filter(ACTOR_PARAM_FILTER))
                ),
                critic_opt_state=critic_tx.init(
                    params.filter(selected_critic_filter)
                ),
                actor_tx=actor_tx,
                critic_tx=critic_tx,
                ema_decay=None if args.ema_decay.lower() == "none" else float(args.ema_decay),
                ema_params=None if args.ema_decay.lower() == "none" else params,
            )
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(nnx.Param)),
            ema_decay=None if args.ema_decay.lower() == "none" else float(args.ema_decay),
            ema_params=None if args.ema_decay.lower() == "none" else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)

    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    loader = FPFCheckpointWeightLoader(
        args.init_checkpoint,
        reset_direct_critic=args.reset_direct_critic,
        reset_v31_flow_critic=args.reset_v31_flow_critic,
    )
    partial_params = loader.load(train_state_shape.params.to_pure_dict())
    at.check_pytree_equality(
        expected=train_state_shape.params.to_pure_dict(), got=partial_params,
        check_shapes=True, check_dtypes=True,
    )
    partial_params = traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(partial_params).items()
         if not isinstance(v, jax.ShapeDtypeStruct)}
    )

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)
    return train_state, state_sharding


def train_step_awr(awr_config: AWRConfig, rng, state, batch):
    model = nnx.merge(state.model_def, state.params)
    model.train()

    train_rng = jax.random.fold_in(rng, state.step)
    baseline_rng, loss_rng = jax.random.split(train_rng)
    observation, actions, progress_targets, is_expert = batch
    current_step = state.step

    def loss_fn(model: Pi0FPFV31, rng, observation, actions, progress_targets, is_expert):
        preprocess_rng, sample_rng = jax.random.split(rng)


        obs_p = _model.preprocess_observation(preprocess_rng, observation, train=True)
        kv_cache, prefix_mask = model.compute_prefix_forward(obs_p)


        if awr_config.stop_prefix_gradient:
            kv_cache = jax.tree.map(jax.lax.stop_gradient, kv_cache)


        s_hat = jax.lax.stop_gradient(
            model.estimate_s_hat_nfe1(baseline_rng, obs_p, kv_cache, prefix_mask,
                                      k_samples=awr_config.s_hat_k_samples)
        )
        advantage_critic = progress_targets - s_hat


        if awr_config.baseline == "critic":
            advantage = advantage_critic
        else:

            advantage = progress_targets - jnp.mean(progress_targets)
        if awr_config.baseline == "binary":

            w_raw = progress_targets / jnp.maximum(jnp.mean(progress_targets), 1e-6)
        elif awr_config.baseline == "expert":


            w_raw = is_expert[:, None, None] / awr_config.expert_frame_ratio * jnp.ones_like(progress_targets)
        else:
            w_raw = jnp.clip(jnp.exp(advantage / awr_config.tau), 0.0, awr_config.clip_max)
        if awr_config.zero_below > 0:
            w_raw = jnp.where(w_raw < awr_config.zero_below, 0.0, w_raw)
        w_policy = jnp.where(current_step < awr_config.warmup_steps, jnp.ones_like(w_raw), w_raw)


        losses = model.compute_fpf_losses(
            sample_rng, obs_p, actions, progress_targets, kv_cache, prefix_mask,
            num_flow_samples=awr_config.num_flow_samples,
            action_weights=w_policy,
        )
        total_loss = (jnp.mean(losses["action_loss"])
                      + awr_config.progress_loss_weight * jnp.mean(losses["progress_loss"]))

        info = {
            "action_loss": jnp.mean(losses["action_loss"]),
            "action_loss_unweighted": jnp.mean(losses["action_loss_unweighted"]),
            "progress_loss": jnp.mean(losses["progress_loss"]),
            "awr/advantage_mean": jnp.mean(advantage),
            "awr/advantage_std": jnp.std(advantage),
            "awr/advantage_critic_mean": jnp.mean(advantage_critic),
            "awr/advantage_critic_std": jnp.std(advantage_critic),
            "awr/weight_mean": jnp.mean(w_policy),
            "awr/weight_min": jnp.min(w_policy),
            "awr/weight_max": jnp.max(w_policy),
            "awr/frac_zero": jnp.mean(w_policy == 0),
            "awr/frac_suppressed": jnp.mean(w_policy < 0.1),
            "awr/s_hat_mean": jnp.mean(s_hat),
            "awr/s_target_mean": jnp.mean(progress_targets),
            "awr/expert_frac": jnp.mean(is_expert),
            "awr/in_warmup": jnp.float32(current_step < awr_config.warmup_steps),
        }
        return total_loss, info

    diff_state = nnx.DiffState(0, nnx.Param)
    (loss, info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, loss_rng, observation, actions, progress_targets, is_expert
    )

    params = state.params.filter(nnx.Param)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params, new_params,
            ),
        )

    info["loss"] = loss
    info["grad_norm"] = optax.global_norm(grads)
    return new_state, info


def clean_critic_objective(
    logits,
    progress_targets,
    is_expert=None,
    *,
    balance_classes: bool = False,
    sample_mask=None,
):
    """Balanced clean-chunk BCE and diagnostics aligned with MC inference."""
    chunk_logits = jnp.mean(logits, axis=(1, 2))
    chunk_targets = jnp.mean(progress_targets, axis=(1, 2))
    per_chunk_loss = optax.sigmoid_binary_cross_entropy(
        chunk_logits,
        chunk_targets,
    )

    label_positive = chunk_targets >= 0.5
    if sample_mask is None:
        valid = jnp.ones_like(label_positive, dtype=jnp.bool_)
    else:
        if sample_mask.shape != label_positive.shape:
            raise ValueError(
                "sample_mask must have shape "
                f"{label_positive.shape}, got {sample_mask.shape}"
            )
        valid = sample_mask.astype(jnp.bool_)
    positive = jnp.logical_and(label_positive, valid)
    negative = jnp.logical_and(jnp.logical_not(label_positive), valid)

    def masked_mean(values, mask):
        mask_f = mask.astype(values.dtype)
        return jnp.sum(values * mask_f) / jnp.maximum(jnp.sum(mask_f), 1.0)

    positive_logit = masked_mean(chunk_logits, positive)
    negative_logit = masked_mean(chunk_logits, negative)
    positive_loss = masked_mean(per_chunk_loss, positive)
    negative_loss = masked_mean(per_chunk_loss, negative)
    if balance_classes:
        has_positive = jnp.any(positive).astype(chunk_logits.dtype)
        has_negative = jnp.any(negative).astype(chunk_logits.dtype)
        clean_loss = (
            has_positive * positive_loss + has_negative * negative_loss
        ) / jnp.maximum(has_positive + has_negative, 1.0)
    else:
        clean_loss = masked_mean(per_chunk_loss, valid)
    predictions = chunk_logits >= 0.0
    logit_mean = masked_mean(chunk_logits, valid)
    logit_std = jnp.sqrt(
        masked_mean(jnp.square(chunk_logits - logit_mean), valid)
    )
    metrics = {
        "critic/clean_loss": clean_loss,
        "critic/clean_positive_loss": positive_loss,
        "critic/clean_negative_loss": negative_loss,
        "critic/clean_logit_mean": logit_mean,
        "critic/clean_logit_std": logit_std,
        "critic/clean_positive_logit": positive_logit,
        "critic/clean_negative_logit": negative_logit,
        "critic/clean_logit_gap": positive_logit - negative_logit,
        "critic/clean_accuracy": masked_mean(
            (predictions == label_positive).astype(chunk_logits.dtype),
            valid,
        ),
        "critic/clean_positive_frac": masked_mean(
            label_positive.astype(chunk_logits.dtype),
            valid,
        ),
    }
    if is_expert is not None:
        rollout = jnp.logical_and(is_expert < 0.5, valid)
        rollout_positive = jnp.logical_and(rollout, positive)
        rollout_negative = jnp.logical_and(rollout, negative)
        rollout_positive_logit = masked_mean(
            chunk_logits,
            rollout_positive,
        )
        rollout_negative_logit = masked_mean(
            chunk_logits,
            rollout_negative,
        )
        metrics.update(
            {
                "critic/rollout_accuracy": masked_mean(
                    (predictions == label_positive).astype(
                        chunk_logits.dtype
                    ),
                    rollout,
                ),
                "critic/rollout_positive_logit": rollout_positive_logit,
                "critic/rollout_negative_logit": rollout_negative_logit,
                "critic/rollout_logit_gap": (
                    rollout_positive_logit - rollout_negative_logit
                ),
            }
        )
    return clean_loss, metrics


def clean_critic_ensemble_objective(
    ensemble_logits,
    progress_targets,
    is_expert=None,
    *,
    balance_classes: bool = False,
    bootstrap_masks=None,
):
    """Average independent per-head bootstrap BCE objectives."""
    if ensemble_logits.ndim != 4:
        raise ValueError(
            "ensemble_logits must have shape [M, B, H, 1], got "
            f"{ensemble_logits.shape}"
        )
    ensemble_size, batch_size = ensemble_logits.shape[:2]
    if bootstrap_masks is None:
        bootstrap_masks = jnp.ones(
            (ensemble_size, batch_size),
            dtype=jnp.bool_,
        )
    elif bootstrap_masks.shape != (ensemble_size, batch_size):
        raise ValueError(
            "bootstrap_masks must have shape "
            f"{(ensemble_size, batch_size)}, got {bootstrap_masks.shape}"
        )

    def one_head(logits, sample_mask):
        loss, _ = clean_critic_objective(
            logits,
            progress_targets,
            is_expert,
            balance_classes=balance_classes,
            sample_mask=sample_mask,
        )
        return loss

    head_losses = jax.vmap(one_head)(ensemble_logits, bootstrap_masks)
    clean_loss = jnp.mean(head_losses)
    mean_logits = jnp.mean(ensemble_logits, axis=0)
    mean_logit_loss, metrics = clean_critic_objective(
        mean_logits,
        progress_targets,
        is_expert,
        balance_classes=balance_classes,
    )
    head_chunk_logits = jnp.mean(ensemble_logits, axis=(2, 3))
    disagreement = jnp.std(head_chunk_logits, axis=0)
    metrics.update(
        {
            "critic/clean_loss": clean_loss,
            "critic/ensemble_mean_logit_loss": mean_logit_loss,
            "critic/ensemble_head_loss_mean": clean_loss,
            "critic/ensemble_head_loss_std": jnp.std(head_losses),
            "critic/ensemble_disagreement_mean": jnp.mean(disagreement),
            "critic/ensemble_disagreement_max": jnp.max(disagreement),
            "critic/bootstrap_fraction": jnp.mean(
                bootstrap_masks.astype(ensemble_logits.dtype)
            ),
        }
    )
    return clean_loss, metrics


def train_critic_step_separate(awr_config: AWRConfig, rng, state, critic_batch):
    """Update the critic head and, when enabled, its shared backbone path."""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    train_rng = jax.random.fold_in(rng, state.step)
    _, critic_rng, baseline_rng = jax.random.split(train_rng, 3)
    critic_observation, critic_actions, progress_targets, is_expert = critic_batch

    def critic_loss_fn(model: Pi0FPFV31):
        preprocess_rng, loss_rng, bootstrap_rng = jax.random.split(
            critic_rng, 3
        )
        observation = _model.preprocess_observation(
            preprocess_rng, critic_observation, train=True
        )
        kv_cache, prefix_mask = model.compute_prefix_forward(observation)
        if not awr_config.critic_update_backbone:


            kv_cache = jax.tree.map(jax.lax.stop_gradient, kv_cache)
        total_loss = jnp.zeros((), dtype=critic_actions.dtype)
        chunk_positive = (
            jnp.mean(progress_targets, axis=(1, 2)) >= 0.5
        )
        positive_count = jnp.maximum(jnp.sum(chunk_positive), 1)
        negative_count = jnp.maximum(
            jnp.sum(jnp.logical_not(chunk_positive)),
            1,
        )
        info = {
            "critic/s_target_mean": jnp.mean(progress_targets),
            "critic/label_positive_frac": jnp.mean(progress_targets),
            "critic/expert_frac": jnp.mean(is_expert),
            "critic/positive_expert_frac": (
                jnp.sum(is_expert * chunk_positive) / positive_count
            ),
            "critic/negative_expert_frac": (
                jnp.sum(is_expert * jnp.logical_not(chunk_positive))
                / negative_count
            ),
        }

        if awr_config.progress_loss_weight > 0:
            losses = model.compute_fpf_losses(
                loss_rng,
                observation,
                critic_actions,
                progress_targets,
                kv_cache,
                prefix_mask,
                num_flow_samples=awr_config.critic_num_flow_samples,
                action_weights=None,
            )
            per_chunk_progress_loss = losses["progress_loss"]
            if awr_config.critic_balance_progress_loss:
                positive = chunk_positive.astype(per_chunk_progress_loss.dtype)
                negative = 1.0 - positive
                positive_mean = (
                    jnp.sum(per_chunk_progress_loss * positive)
                    / jnp.maximum(jnp.sum(positive), 1.0)
                )
                negative_mean = (
                    jnp.sum(per_chunk_progress_loss * negative)
                    / jnp.maximum(jnp.sum(negative), 1.0)
                )
                progress_loss = 0.5 * (positive_mean + negative_mean)
            else:
                progress_loss = jnp.mean(per_chunk_progress_loss)
            s_hat = jax.lax.stop_gradient(
                model.estimate_s_hat_nfe1(
                    baseline_rng,
                    observation,
                    kv_cache,
                    prefix_mask,
                    k_samples=awr_config.s_hat_k_samples,
                )
            )
            info.update(
                {
                    "progress_loss": progress_loss,
                    "critic/progress_loss_unbalanced": jnp.mean(per_chunk_progress_loss),
                    "critic/s_hat_mean": jnp.mean(s_hat),
                    "critic/residual_mean": jnp.mean(
                        progress_targets - s_hat
                    ),
                    "critic/residual_std": jnp.std(
                        progress_targets - s_hat
                    ),
                }
            )
            total_loss = (
                total_loss + awr_config.progress_loss_weight * progress_loss
            )

        if awr_config.critic_clean_loss_weight > 0:
            if awr_config.critic_ensemble_size <= 1:
                critic_logits = model.predict_critic_logits(
                    critic_actions,
                    kv_cache,
                    prefix_mask,
                )
                clean_loss, clean_metrics = clean_critic_objective(
                    critic_logits,
                    progress_targets,
                    is_expert,
                    balance_classes=awr_config.critic_balance_clean_loss,
                )
            else:
                ensemble_logits = model.predict_critic_ensemble_logits(
                    critic_actions,
                    kv_cache,
                    prefix_mask,
                )
                bootstrap_masks = jax.random.bernoulli(
                    bootstrap_rng,
                    p=awr_config.critic_bootstrap_prob,
                    shape=(
                        awr_config.critic_ensemble_size,
                        critic_actions.shape[0],
                    ),
                )
                clean_loss, clean_metrics = (
                    clean_critic_ensemble_objective(
                        ensemble_logits,
                        progress_targets,
                        is_expert,
                        balance_classes=(
                            awr_config.critic_balance_clean_loss
                        ),
                        bootstrap_masks=bootstrap_masks,
                    )
                )
            info.update(clean_metrics)
            total_loss = (
                total_loss
                + awr_config.critic_clean_loss_weight * clean_loss
            )

        return total_loss, info

    selected_critic_filter = critic_param_filter(
        awr_config.progress_loss_weight,
        awr_config.critic_clean_loss_weight,
        awr_config.critic_update_backbone,
    )
    (critic_loss_weighted, info), critic_grads = nnx.value_and_grad(
        critic_loss_fn,
        argnums=nnx.DiffState(0, selected_critic_filter),
        has_aux=True,
    )(model)
    params = state.params.filter(selected_critic_filter)
    updates, critic_opt_state = state.critic_tx.update(
        critic_grads,
        state.critic_opt_state,
        params,
    )
    nnx.update(model, optax.apply_updates(params, updates))
    new_state = dataclasses.replace(
        state,
        params=nnx.state(model),
        critic_opt_state=critic_opt_state,
    )
    info["critic_loss_weighted"] = critic_loss_weighted
    info["critic_grad_norm"] = optax.global_norm(critic_grads)
    return new_state, info


def train_actor_awr_micro_step_separate(
    awr_config: AWRConfig,
    rng,
    state,
    actor_batch,
    micro_index,
):
    """Update only actor leaves with stage-labeled AWR on an independent batch."""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    train_rng = jax.random.fold_in(rng, state.step)
    train_rng = jax.random.fold_in(train_rng, micro_index)
    preprocess_rng, flow_rng, baseline_rng = jax.random.split(train_rng, 3)
    observation, actions, progress_targets, is_expert = actor_batch
    current_step = state.step

    def actor_loss_fn(model: Pi0FPFV31):
        observation_p = _model.preprocess_observation(
            preprocess_rng,
            observation,
            train=True,
        )
        kv_cache, prefix_mask = model.compute_prefix_forward(observation_p)
        chunk_targets = jnp.mean(progress_targets, axis=(1, 2))

        if awr_config.baseline == "direct_critic":
            critic_logits = jax.lax.stop_gradient(
                model.predict_critic_logits(
                    actions,
                    jax.tree.map(jax.lax.stop_gradient, kv_cache),
                    prefix_mask,
                )
            )
            s_hat = jax.nn.sigmoid(jnp.mean(critic_logits, axis=(1, 2)))
            advantage = chunk_targets - s_hat
        elif awr_config.baseline == "critic":
            s_hat_full = jax.lax.stop_gradient(
                model.estimate_s_hat_nfe1(
                    baseline_rng,
                    observation_p,
                    kv_cache,
                    prefix_mask,
                    k_samples=awr_config.s_hat_k_samples,
                )
            )
            s_hat = jnp.mean(s_hat_full, axis=(1, 2))
            advantage = chunk_targets - s_hat
        else:
            s_hat = jnp.full_like(chunk_targets, jnp.nan)
            advantage = chunk_targets - jnp.mean(chunk_targets)

        if awr_config.baseline == "binary":
            w_raw = chunk_targets / jnp.maximum(jnp.mean(chunk_targets), 1e-6)
        elif awr_config.baseline == "expert":
            w_raw = is_expert / awr_config.expert_frame_ratio
        else:
            w_raw = jnp.clip(
                jnp.exp(advantage / awr_config.tau),
                0.0,
                awr_config.clip_max,
            )
        if awr_config.zero_below > 0:
            w_raw = jnp.where(w_raw < awr_config.zero_below, 0.0, w_raw)
        weights = jnp.where(
            current_step < awr_config.warmup_steps,
            jnp.ones_like(w_raw),
            w_raw,
        )

        losses = model.compute_fpf_losses(
            flow_rng,
            observation_p,
            actions,
            progress_targets,
            kv_cache,
            prefix_mask,
            num_flow_samples=awr_config.num_flow_samples,
            action_weights=weights[:, None, None],
        )
        actor_loss = jnp.mean(losses["action_loss"])
        info = {
            "action_loss": actor_loss,
            "action_loss_unweighted": jnp.mean(losses["action_loss_unweighted"]),
            "actor/awr_advantage_mean": jnp.mean(advantage),
            "actor/awr_advantage_std": jnp.std(advantage),
            "actor/awr_weight_mean": jnp.mean(weights),
            "actor/awr_weight_min": jnp.min(weights),
            "actor/awr_weight_max": jnp.max(weights),
            "actor/awr_frac_zero": jnp.mean(weights == 0),
            "actor/awr_frac_suppressed": jnp.mean(weights < 0.1),
            "actor/awr_target_mean": jnp.mean(chunk_targets),
            "actor/awr_baseline_mean": jnp.nanmean(s_hat),
            "actor/expert_frac": jnp.mean(is_expert),
            "actor/awr_in_warmup": jnp.float32(
                current_step < awr_config.warmup_steps
            ),
        }
        return actor_loss, info

    (actor_loss, info), actor_grads = nnx.value_and_grad(
        actor_loss_fn,
        argnums=nnx.DiffState(0, ACTOR_PARAM_FILTER),
        has_aux=True,
    )(model)
    params = state.params.filter(ACTOR_PARAM_FILTER)
    updates, actor_opt_state = state.actor_tx.update(
        actor_grads,
        state.actor_opt_state,
        params,
    )
    nnx.update(model, optax.apply_updates(params, updates))
    new_state = dataclasses.replace(
        state,
        params=nnx.state(model),
        actor_opt_state=actor_opt_state,
    )
    info["actor_micro_grad_norm"] = optax.global_norm(actor_grads)
    return new_state, info


def train_actor_micro_step_separate(rng, state, actor_batch, micro_index):
    """Accumulate one inherited-Pi0 actor microbatch with Optax MultiSteps."""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    train_rng = jax.random.fold_in(rng, state.step)
    train_rng = jax.random.fold_in(train_rng, micro_index)
    actor_rng, _, _ = jax.random.split(train_rng, 3)
    actor_observation, actor_actions = actor_batch

    def actor_loss_fn(model: Pi0FPFV31):


        action_loss = model.compute_loss(
            actor_rng,
            actor_observation,
            actor_actions,
            train=True,
        )
        return jnp.mean(action_loss)

    actor_loss, actor_grads = nnx.value_and_grad(
        actor_loss_fn,
        argnums=nnx.DiffState(0, ACTOR_PARAM_FILTER),
    )(model)
    params = state.params.filter(ACTOR_PARAM_FILTER)
    updates, actor_opt_state = state.actor_tx.update(
        actor_grads,
        state.actor_opt_state,
        params,
    )
    nnx.update(model, optax.apply_updates(params, updates))
    new_params = nnx.state(model)
    new_state = dataclasses.replace(
        state,
        params=new_params,
        actor_opt_state=actor_opt_state,
    )
    info = {
        "action_loss": actor_loss,
        "actor_micro_grad_norm": optax.global_norm(actor_grads),
    }
    return new_state, info


def finalize_separate_step(state):
    """Advance the global step and EMA once per effective separate-mode step."""
    new_state = dataclasses.replace(state, step=state.step + 1)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                state.params,
            ),
        )
    return new_state


def init_checkpoint_manager(checkpoint_dir: str, *, overwrite: bool):
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    if checkpoint_dir.exists() and overwrite:
        checkpoint_dir.rmtree()
        logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": _checkpoints.CallbackHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=None,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )


def save_params_only(manager, state: training_utils.TrainState, norm_stats, asset_id: str, step: int):
    """只保存推理所需内容: params(优先 EMA)+ norm stats assets。"""
    params = state.ema_params if state.ema_params is not None else state.params

    def save_assets(directory: epath.Path):
        _normalize.save(directory / asset_id, norm_stats)

    with at.disable_typechecking():
        manager.save(step, {"assets": save_assets, "params": {"params": params}})


def main():
    args = parse_args()
    explicit_save_steps = {
        int(value.strip()) for value in args.save_steps.split(",") if value.strip()
    }
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info(f"Running on: {platform.node()}")

    if args.smoke_test:


        smoke_batch = (
            (max(8, jax.device_count()) + jax.device_count() - 1)
            // jax.device_count()
            * jax.device_count()
        )
        args.batch_size = smoke_batch
        args.actor_batch_size = smoke_batch
        args.actor_grad_accum_steps = 1
        args.critic_updates_per_actor_step = 1
        args.num_train_steps = 3
        args.num_flow_samples = 2
        args.num_workers = 0
        args.wandb_enabled = False
        args.save_interval = 2
        args.high_warmup_steps = 1
        args.backbone_warmup_delay = 1
        args.backbone_warmup_ramp = 1
        args.awr_warmup_steps = 1

    if args.separate_actor_critic_batches:
        if not args.critic_rollout_only and not args.expert_data_root:
            raise ValueError(
                "--separate_actor_critic_batches requires --expert_data_root "
                "unless --critic_rollout_only is enabled"
            )
        if not args.stop_progress_gradient:
            raise ValueError(
                "--separate_actor_critic_batches requires --stop_progress_gradient"
            )
        if args.stop_prefix_gradient:
            raise ValueError(
                "--separate_actor_critic_batches requires --no-stop_prefix_gradient"
            )
    if args.actor_only_control and not args.separate_actor_critic_batches:
        raise ValueError(
            "--actor_only_control requires --separate_actor_critic_batches"
        )
    if args.critic_only and not args.separate_actor_critic_batches:
        raise ValueError(
            "--critic_only requires --separate_actor_critic_batches"
        )
    if args.actor_only_control and args.critic_only:
        raise ValueError(
            "--actor_only_control and --critic_only are mutually exclusive"
        )
    if args.critic_updates_per_actor_step <= 0:
        raise ValueError("--critic_updates_per_actor_step must be positive")
    if (
        args.critic_updates_per_actor_step != 1
        and not args.separate_actor_critic_batches
    ):
        raise ValueError(
            "--critic_updates_per_actor_step != 1 requires "
            "--separate_actor_critic_batches"
        )
    if args.actor_only_control and args.critic_updates_per_actor_step != 1:
        raise ValueError(
            "--actor_only_control does not execute critic updates; use "
            "--critic_updates_per_actor_step 1"
        )
    if args.critic_clean_loss_weight > 0 and not args.separate_actor_critic_batches:
        raise ValueError(
            "--critic_clean_loss_weight requires "
            "--separate_actor_critic_batches"
        )
    if (
        args.separate_actor_critic_batches
        and not args.actor_only_control
        and args.progress_loss_weight == 0
        and args.critic_clean_loss_weight == 0
    ):
        raise ValueError(
            "critic training requires a positive --progress_loss_weight or "
            "--critic_clean_loss_weight"
        )
    if args.batch_size % jax.device_count() != 0:
        raise ValueError(f"batch_size {args.batch_size} 必须能被设备数 {jax.device_count()} 整除")
    if args.separate_actor_critic_batches and not args.critic_only:
        if args.actor_grad_accum_steps <= 0:
            raise ValueError("--actor_grad_accum_steps must be positive")
        if args.actor_batch_size % args.actor_grad_accum_steps != 0:
            raise ValueError(
                f"actor_batch_size {args.actor_batch_size} 必须能被 "
                f"actor_grad_accum_steps {args.actor_grad_accum_steps} 整除"
            )
        actor_microbatch_size = (
            args.actor_batch_size // args.actor_grad_accum_steps
        )
        if actor_microbatch_size % jax.device_count() != 0:
            raise ValueError(
                f"actor microbatch_size {actor_microbatch_size} 必须能被设备数 "
                f"{jax.device_count()} 整除"
            )

    checkpoint_dir = os.path.join(args.checkpoint_base_dir, args.exp_name)
    critic_label_mode = args.critic_label_mode or args.label_mode
    logging.info(
        "Label modes: actor=%s, critic=%s; train steps=%d; LR schedule steps=%d",
        args.label_mode,
        critic_label_mode,
        args.num_train_steps,
        args.lr_schedule_steps or args.num_train_steps,
    )

    rng = jax.random.key(args.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


    if args.actor_only_control:
        model_config = dataclasses.replace(
            _config.get_config("pi05_libero").model,
            action_horizon=args.action_horizon,
        )
        logging.info(
            "Actor-only matched BC control: ordinary Pi0.5 model, no progress parameters"
        )
    else:
        model_config = Pi0FPFV31Config(
            action_horizon=args.action_horizon,
            separate_progress_in_proj=args.separate_progress_in_proj,
            stop_progress_gradient=args.stop_progress_gradient,
            critic_adapter_hidden_dim=0,
            critic_adapter_depth=1,
            critic_update_backbone=False,
            critic_feature_layer=0,
            critic_ensemble_size=1,
            v31_flow_critic_hidden_dim=args.v31_flow_critic_hidden_dim,
            v31_flow_critic_depth=args.v31_flow_critic_depth,
        )


    norm_stats = _checkpoints.load_norm_stats(args.norm_stats_dir, args.asset_id)
    transform_fns = [

        libero_policy.LiberoInputs(model_type=model_config.model_type),
        _transforms.Normalize(
            norm_stats, use_quantiles=args.critic_use_quantile_norm
        ),
        _transforms.ResizeImages(224, 224),
        _transforms.TokenizePrompt(
            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
            discrete_state_input=model_config.discrete_state_input,
        ),
        _transforms.PadStatesAndActions(model_config.action_dim),
    ]
    dataset = LiberoLabeledDataset(
        data_root=args.data_root,
        suites=args.suites,
        action_horizon=args.action_horizon,
        transform_fns=transform_fns,
        label_mode=critic_label_mode,
        use_success_only=args.use_success_only,
        expert_root=(
            None
            if args.critic_rollout_only
            else args.expert_data_root or None
        ),
        rollout_manifest=args.rollout_manifest or None,
        drop_post_success_anchors=args.drop_post_success_anchors,
    )

    import torch.utils.data as torch_data
    sampler = None
    shuffle = True
    torch_generator = torch.Generator()
    torch_generator.manual_seed(int(args.seed))
    if args.critic_mixed_task_source_sampling:
        weights = dataset.get_sample_weights_by_task_source_outcome_episode(
            args.expert_frame_ratio,
            args.success_frame_ratio,
            args.critic_rollout_new_failure_ratio,
        )
        sampler = dataset.make_weighted_sampler(weights, torch_generator)
        shuffle = False
    elif args.critic_task_uniform_sampling:
        weights = dataset.get_sample_weights_by_task_outcome_episode(
            args.success_frame_ratio
        )
        sampler = torch_data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
            generator=torch_generator,
        )
        shuffle = False
    elif args.critic_target_positive_ratio is not None:


        weights = dataset.get_sample_weights(
            args.critic_target_positive_ratio
        )
        sampler = torch_data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
            generator=torch_generator,
        )
        shuffle = False
    elif args.expert_data_root:

        weights = dataset.get_sample_weights_by_source(args.expert_frame_ratio, args.success_frame_ratio)
        sampler = torch_data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
            generator=torch_generator,
        )
        shuffle = False
    elif args.success_frame_ratio is not None and args.success_frame_ratio > 0:
        weights = dataset.get_sample_weights_by_outcome(args.success_frame_ratio)
        sampler = torch_data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
            generator=torch_generator,
        )
        shuffle = False
    elif args.target_success_ratio is not None:
        weights = dataset.get_sample_weights(args.target_success_ratio)
        sampler = torch_data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
            generator=torch_generator,
        )
        shuffle = False
    torch_loader = torch_data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        generator=torch_generator,
    )

    def data_iter(loader):
        while True:
            for batch_dict in loader:
                progress_target = jnp.asarray(batch_dict.pop("progress_target"))
                is_expert = jnp.asarray(batch_dict.pop("is_expert"))
                actions = jnp.asarray(batch_dict.pop("actions"))
                obs = _model.Observation.from_dict(batch_dict)
                batch = (obs, actions, progress_target, is_expert)
                yield jax.device_put(batch, data_sharding)

    it = data_iter(torch_loader)
    batch = next(it)
    critic_batch_kind = (
        "mixed-task-source"
        if args.critic_mixed_task_source_sampling
        else "rollout-only"
        if args.critic_rollout_only
        else "mixed"
    )
    logging.info(
        "First critic/%s batch:\n%s",
        critic_batch_kind,
        training_utils.array_tree_to_info(batch),
    )

    actor_it = None
    actor_batch = None
    if args.separate_actor_critic_batches and not args.critic_only:
        actor_microbatch_size = (
            args.actor_batch_size // args.actor_grad_accum_steps
        )
        if args.separate_actor_awr:
            actor_dataset = LiberoLabeledDataset(
                data_root=args.data_root,
                suites=args.suites,
                action_horizon=args.action_horizon,
                transform_fns=transform_fns,
                label_mode=args.label_mode,
                use_success_only=False,
                expert_root=args.expert_data_root,
                rollout_manifest=args.rollout_manifest or None,
                drop_post_success_anchors=args.drop_post_success_anchors,
            )
            actor_generator = torch.Generator()
            actor_generator.manual_seed(int(args.seed) + 1009)
            if args.actor_mixed_task_source_sampling:
                actor_weights = (
                    actor_dataset.get_sample_weights_by_task_source_outcome_episode(
                        args.actor_expert_frame_ratio,
                        args.actor_rollout_success_ratio,
                        None,
                    )
                )
            else:
                actor_weights = actor_dataset.get_sample_weights_by_source(
                    args.actor_expert_frame_ratio,
                    args.actor_rollout_success_ratio,
                )
            actor_sampler = actor_dataset.make_weighted_sampler(actor_weights, actor_generator)
            actor_loader = torch_data.DataLoader(
                actor_dataset,
                batch_size=actor_microbatch_size,
                shuffle=False,
                sampler=actor_sampler,
                num_workers=args.num_workers,
                collate_fn=collate,
                drop_last=True,
                persistent_workers=args.num_workers > 0,
                generator=actor_generator,
            )
            actor_it = data_iter(actor_loader)
            actor_batch = next(actor_it)
            logging.info(
                "First labeled AWR actor microbatch "
                "(expert_ratio=%.3f, rollout_success_ratio=%.3f, "
                "%d x %d = effective batch %d):\n%s",
                args.actor_expert_frame_ratio,
                args.actor_rollout_success_ratio,
                actor_microbatch_size,
                args.actor_grad_accum_steps,
                args.actor_batch_size,
                training_utils.array_tree_to_info(actor_batch),
            )
        else:
            actor_base_config = _config.get_config("pi05_libero")
            actor_data_config = dataclasses.replace(
                actor_base_config.data,
                episodes_per_task=args.actor_episodes_per_task,
                episodes_seed=args.actor_episodes_seed,
            )
            actor_train_config = dataclasses.replace(
                actor_base_config,
                model=model_config,
                data=actor_data_config,
                assets_base_dir=os.path.dirname(args.norm_stats_dir.rstrip("/")),
                batch_size=actor_microbatch_size,
                num_workers=args.num_workers,
                seed=args.seed,
            )
            actor_loader = _data_loader.create_data_loader(
                actor_train_config,
                sharding=data_sharding,
                shuffle=True,
            )
            actor_it = iter(actor_loader)
            actor_batch = next(actor_it)
            logging.info(
                "First official expert actor microbatch "
                "(%d x %d = effective batch %d):\n%s",
                actor_microbatch_size,
                args.actor_grad_accum_steps,
                args.actor_batch_size,
                training_utils.array_tree_to_info(actor_batch),
            )


    import wandb
    if args.wandb_enabled:
        wandb.init(name=args.exp_name, config=vars(args), project=args.project_name)
    else:
        wandb.init(mode="disabled")


    train_state, train_state_sharding = init_train_state(args, model_config, init_rng, mesh)
    jax.block_until_ready(train_state)
    logging.info("Train state initialized from checkpoint: " + args.init_checkpoint)


    manager = None
    if not args.skip_checkpoint_save:
        manager = init_checkpoint_manager(checkpoint_dir, overwrite=args.overwrite)
        with open(os.path.join(checkpoint_dir, "training_config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)
            f.write("\n")


    awr_config = AWRConfig(
        tau=args.awr_tau,
        clip_max=args.awr_clip_max,
        zero_below=args.awr_zero_below,
        warmup_steps=args.awr_warmup_steps,
        num_flow_samples=args.num_flow_samples,
        critic_num_flow_samples=args.critic_num_flow_samples,
        critic_balance_progress_loss=args.critic_balance_progress_loss,
        stop_prefix_gradient=args.stop_prefix_gradient,
        baseline=args.awr_baseline,
        s_hat_k_samples=args.s_hat_k_samples,
        progress_loss_weight=args.progress_loss_weight,
        critic_clean_loss_weight=args.critic_clean_loss_weight,
        critic_balance_clean_loss=args.critic_balance_clean_loss,
        critic_ensemble_size=args.critic_ensemble_size,
        critic_bootstrap_prob=args.critic_bootstrap_prob,
        critic_update_backbone=args.critic_update_backbone,
        expert_frame_ratio=args.expert_frame_ratio,
    )
    if args.separate_actor_critic_batches:
        pcritic_step = None
        if not args.actor_only_control:
            pcritic_step = jax.jit(
                functools.partial(train_critic_step_separate, awr_config),
                in_shardings=(
                    replicated_sharding,
                    train_state_sharding,
                    data_sharding,
                ),
                out_shardings=(train_state_sharding, replicated_sharding),
                donate_argnums=(1,),
            )
        pactor_micro_step = None
        if not args.critic_only:
            actor_step_fn = (
                functools.partial(train_actor_awr_micro_step_separate, awr_config)
                if args.separate_actor_awr
                else train_actor_micro_step_separate
            )
            pactor_micro_step = jax.jit(
                actor_step_fn,
                in_shardings=(
                    replicated_sharding,
                    train_state_sharding,
                    data_sharding,
                    replicated_sharding,
                ),
                out_shardings=(train_state_sharding, replicated_sharding),
                donate_argnums=(1,),
            )
        pfinalize_step = jax.jit(
            finalize_separate_step,
            in_shardings=(train_state_sharding,),
            out_shardings=train_state_sharding,
            donate_argnums=(0,),
        )
    else:
        ptrain_step = jax.jit(
            functools.partial(train_step_awr, awr_config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )


    pbar = tqdm.tqdm(range(args.num_train_steps), dynamic_ncols=True)
    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            if args.separate_actor_critic_batches:
                critic_info = None
                if not args.actor_only_control:
                    critic_infos = []
                    for critic_update_index in range(
                        args.critic_updates_per_actor_step
                    ):
                        critic_rng = jax.random.fold_in(
                            train_rng, critic_update_index
                        )
                        train_state, one_critic_info = pcritic_step(
                            critic_rng, train_state, batch
                        )
                        critic_infos.append(one_critic_info)
                        if (
                            critic_update_index + 1
                            < args.critic_updates_per_actor_step
                        ):
                            batch = next(it)
                    critic_info = jax.tree.map(
                        jnp.mean,
                        common_utils.stack_forest(critic_infos),
                    )
                    critic_info["critic/updates_per_actor_step"] = jnp.float32(
                        args.critic_updates_per_actor_step
                    )
                if args.critic_only:
                    train_state = pfinalize_step(train_state)
                    info = {
                        **critic_info,
                        "loss": critic_info["critic_loss_weighted"],
                    }
                else:
                    actor_micro_infos = []
                    for micro_index in range(args.actor_grad_accum_steps):
                        train_state, actor_micro_info = pactor_micro_step(
                            train_rng,
                            train_state,
                            actor_batch,
                            jnp.int32(micro_index),
                        )
                        actor_micro_infos.append(actor_micro_info)
                        actor_batch = next(actor_it)
                    train_state = pfinalize_step(train_state)
                    actor_info = jax.tree.map(
                        jnp.mean,
                        common_utils.stack_forest(actor_micro_infos),
                    )
                    if args.actor_only_control:
                        info = {
                            **actor_info,
                            "loss": actor_info["action_loss"],
                        }
                    else:
                        info = {**critic_info, **actor_info}
                        info["loss"] = (
                            actor_info["action_loss"]
                            + critic_info["critic_loss_weighted"]
                        )
            else:
                train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % args.log_interval == 0:
            stacked = common_utils.stack_forest(infos)
            reduced = jax.device_get(jax.tree.map(jnp.mean, stacked))
            pbar.write(f"Step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in reduced.items()))
            wandb.log(reduced, step=step)
            infos = []
        if not args.actor_only_control:
            batch = next(it)

        if manager is not None and (
            (
                step > 0
                and (
                    step % args.save_interval == 0
                    or step in explicit_save_steps
                )
            )
            or step == args.num_train_steps - 1
        ):
            save_params_only(manager, train_state, norm_stats, args.asset_id, step)
            logging.info(f"Saved params-only checkpoint at step {step}")

    if manager is not None:
        manager.wait_until_finished()
    logging.info("Training finished successfully!")


if __name__ == "__main__":
    main()
