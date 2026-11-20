"""Pi0.5 action flow with a detached progress-flow critic for LIBERO.

The actor predicts action velocity. The critic reads detached action features,
noisy progress and flow time to predict progress velocity. Inference ranks
action candidates by their generated progress scores."""

import dataclasses

import einops
from flax.core import freeze, unfreeze
import flax.nnx as nnx
from flax.nnx.bridge import variables as nnx_bridge_variables
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import Pi0, make_attn_mask
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at


def aggregate_critic_ensemble_scores(
    head_scores: at.Float[at.Array, "m b"],
    uncertainty_beta: float,
) -> at.Float[at.Array, " b"]:
    """Aggregate per-head chunk logits with a lower-confidence-bound score."""
    if uncertainty_beta < 0:
        raise ValueError("uncertainty_beta must be non-negative")
    return (
        jnp.mean(head_scores, axis=0)
        - uncertainty_beta * jnp.std(head_scores, axis=0)
    )


@dataclasses.dataclass(frozen=True)
class Pi0FPFV31Config(pi0_config.Pi0Config):
    """pi05_libero 配置 + FPF 开关。默认值对齐注册的 pi05_libero config。"""

    action_horizon: int = 10
    pi05: bool = True
    discrete_state_input: bool = False


    separate_progress_in_proj: bool = False


    mc_num_candidates: int = 1


    mc_selection_mode: str = "argmax"


    mc_remove_initial_progress_noise: bool = False


    mc_share_progress_noise: bool = False


    mc_use_critic_score: bool = False


    mc_critic_score_horizon: int = 0

    mc_log_diagnostics: bool = False


    critic_adapter_hidden_dim: int = 0


    critic_adapter_depth: int = 1


    critic_update_backbone: bool = False


    critic_feature_layer: int = 0


    critic_ensemble_size: int = 1


    mc_critic_uncertainty_beta: float = 0.0


    stop_progress_gradient: bool = False

    v31_flow_critic_hidden_dim: int = 256
    v31_flow_critic_depth: int = 2

    @property
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05

    def create(self, rng: at.KeyArrayLike) -> "Pi0FPFV31":
        return Pi0FPFV31(self, rngs=nnx.Rngs(rng))


class Pi0FPFV31(Pi0):
    def __init__(self, config: Pi0FPFV31Config, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        assert config.pi05, "FPF-LIBERO only supports pi05-style models"
        self.separate_progress_in_proj = config.separate_progress_in_proj
        self.mc_num_candidates = config.mc_num_candidates
        if config.mc_selection_mode not in {"argmax", "argmin", "random"}:
            raise ValueError(
                "mc_selection_mode must be 'argmax', 'argmin', or 'random', got "
                f"{config.mc_selection_mode!r}"
            )
        self.mc_selection_mode = config.mc_selection_mode
        self.mc_remove_initial_progress_noise = config.mc_remove_initial_progress_noise
        self.mc_share_progress_noise = config.mc_share_progress_noise
        self.mc_use_critic_score = config.mc_use_critic_score
        if config.mc_critic_score_horizon < 0:
            raise ValueError("mc_critic_score_horizon must be non-negative")
        if config.mc_critic_score_horizon > config.action_horizon:
            raise ValueError(
                "mc_critic_score_horizon cannot exceed action_horizon: "
                f"{config.mc_critic_score_horizon} > {config.action_horizon}"
            )
        self.mc_critic_score_horizon = config.mc_critic_score_horizon
        self.mc_log_diagnostics = config.mc_log_diagnostics
        enabled_scorers = sum(
            (
                self.mc_remove_initial_progress_noise,
                self.mc_share_progress_noise,
                self.mc_use_critic_score,
            )
        )
        if enabled_scorers > 1:
            raise ValueError(
                "residual, shared-progress, and direct critic scoring protocols "
                "are mutually exclusive"
            )
        self.stop_progress_gradient = config.stop_progress_gradient
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        if config.critic_adapter_hidden_dim < 0:
            raise ValueError("critic_adapter_hidden_dim must be non-negative")
        self.critic_adapter_hidden_dim = config.critic_adapter_hidden_dim
        if config.critic_adapter_depth < 1:
            raise ValueError("critic_adapter_depth must be at least 1")
        self.critic_adapter_depth = config.critic_adapter_depth
        self.critic_update_backbone = config.critic_update_backbone
        if not 0 <= config.critic_feature_layer <= action_expert_config.depth:
            raise ValueError(
                "critic_feature_layer must be in [0, action expert depth], got "
                f"{config.critic_feature_layer} for depth "
                f"{action_expert_config.depth}"
            )
        self.critic_feature_layer = config.critic_feature_layer
        self.critic_action_expert_depth = action_expert_config.depth
        self.critic_paligemma_variant = config.paligemma_variant
        self.critic_action_expert_variant = config.action_expert_variant
        self.critic_embed_dtype = config.dtype
        if (
            0 < self.critic_feature_layer < self.critic_action_expert_depth
            and self.critic_update_backbone
        ):
            raise ValueError(
                "intermediate critic_feature_layer currently requires a "
                "frozen/detached actor backbone"
            )
        if config.critic_ensemble_size < 1:
            raise ValueError("critic_ensemble_size must be at least 1")
        self.critic_ensemble_size = config.critic_ensemble_size
        if config.mc_critic_uncertainty_beta < 0:
            raise ValueError("mc_critic_uncertainty_beta must be non-negative")
        if (
            config.mc_critic_uncertainty_beta > 0
            and config.critic_ensemble_size <= 1
        ):
            raise ValueError(
                "mc_critic_uncertainty_beta > 0 requires critic_ensemble_size > 1"
            )
        self.mc_critic_uncertainty_beta = config.mc_critic_uncertainty_beta
        if config.separate_progress_in_proj:


            self.progress_in_proj = nnx.Linear(
                1, action_expert_config.width,
                kernel_init=nnx.initializers.zeros_init(), rngs=rngs,
            )
        else:


            self.action_in_proj = nnx.Linear(config.action_dim + 1, action_expert_config.width, rngs=rngs)
        self.progress_out_proj = nnx.Linear(action_expert_config.width, 1, rngs=rngs)
        if self.critic_adapter_hidden_dim > 0:


            for head_index in range(self.critic_ensemble_size):
                head_suffix = "" if head_index == 0 else f"_{head_index}"
                for block_index in range(self.critic_adapter_depth):


                    block_suffix = (
                        head_suffix
                        if block_index == 0
                        else f"_block{block_index}{head_suffix}"
                    )
                    setattr(
                        self,
                        f"critic_adapter_down{block_suffix}",
                        nnx.Linear(
                            action_expert_config.width,
                            self.critic_adapter_hidden_dim,
                            rngs=rngs,
                        ),
                    )
                    setattr(
                        self,
                        f"critic_adapter_up{block_suffix}",
                        nnx.Linear(
                            self.critic_adapter_hidden_dim,
                            action_expert_config.width,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        ),
                    )
                setattr(
                    self,
                    f"critic_score_out_proj{head_suffix}",
                    nnx.Linear(
                        action_expert_config.width,
                        1,
                        kernel_init=nnx.initializers.zeros_init(),
                        bias_init=nnx.initializers.zeros_init(),
                        rngs=rngs,
                    ),
                )
        elif self.mc_use_critic_score:
            raise ValueError(
                "mc_use_critic_score requires critic_adapter_hidden_dim > 0"
            )
        elif self.critic_ensemble_size > 1:
            raise ValueError(
                "critic_ensemble_size > 1 requires critic_adapter_hidden_dim > 0"
            )

        if config.v31_flow_critic_hidden_dim <= 0:
            raise ValueError("v31_flow_critic_hidden_dim must be positive")
        if config.v31_flow_critic_depth < 1:
            raise ValueError("v31_flow_critic_depth must be at least 1")
        self.v31_flow_critic_hidden_dim = config.v31_flow_critic_hidden_dim
        self.v31_flow_critic_depth = config.v31_flow_critic_depth
        width = action_expert_config.width
        self.v31_flow_critic_s_proj = nnx.Linear(1, width, rngs=rngs)
        self.v31_flow_critic_time_proj = nnx.Linear(width, width, rngs=rngs)
        for block_index in range(self.v31_flow_critic_depth):
            suffix = "" if block_index == 0 else f"_block{block_index}"
            setattr(
                self,
                f"v31_flow_critic_adapter_down{suffix}",
                nnx.Linear(width, self.v31_flow_critic_hidden_dim, rngs=rngs),
            )
            setattr(
                self,
                f"v31_flow_critic_adapter_up{suffix}",
                nnx.Linear(
                    self.v31_flow_critic_hidden_dim,
                    width,
                    kernel_init=nnx.initializers.zeros_init(),
                    bias_init=nnx.initializers.zeros_init(),
                    rngs=rngs,
                ),
            )
        self.v31_flow_critic_out_proj = nnx.Linear(width, 1, rngs=rngs)

    def embed_suffix_fpf(
        self,
        noisy_actions: _model.Actions,
        noisy_progress: at.Float[at.Array, "b ah 1"],
        timestep: at.Float[at.Array, " b"],
    ):
        """Embed the action stream without allowing progress to perturb Actor."""
        from openpi.models.pi0 import posemb_sincos

        del noisy_progress
        if not self.separate_progress_in_proj:
            raise ValueError(
                "V31 requires separate_progress_in_proj=True so Actor input "
                "keeps the native Pi0.5 action dimension"
            )
        action_tokens = self.action_in_proj(noisy_actions)

        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        adarms_cond = time_emb

        input_mask = jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_)
        ar_mask = jnp.array([True] + [False] * (self.action_horizon - 1))
        return action_tokens, input_mask, ar_mask, adarms_cond

    def compute_prefix_forward(self, observation: _model.Observation):
        """Prefix (VLM) 前向一次,返回 KV cache。梯度可流经此处。"""
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        return kv_cache, prefix_mask

    def _suffix_features(self, x_t, s_t, time, kv_cache, prefix_mask):
        """Run the shared action suffix and return its H action-token features."""
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_fpf(x_t, s_t, time)
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        return suffix_out[:, -self.action_horizon :]

    def _critic_suffix_features(
        self,
        x_t,
        s_t,
        time,
        kv_cache,
        prefix_mask,
    ):

        layer = getattr(self, "critic_feature_layer", 0)
        full_depth = getattr(
            self,
            "critic_action_expert_depth",
            _gemma.get_config("gemma_300m").depth,
        )
        if layer in (0, full_depth):
            return self._suffix_features(
                x_t,
                s_t,
                time,
                kv_cache,
                prefix_mask,
            )

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = (
            self.embed_suffix_fpf(x_t, s_t, time)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(
            prefix_mask,
            "b p -> b s p",
            s=suffix_tokens.shape[1],
        )
        full_attn_mask = jnp.concatenate(
            [prefix_attn_mask, suffix_attn_mask],
            axis=-1,
        )
        positions = (
            jnp.sum(prefix_mask, axis=-1)[:, None]
            + jnp.cumsum(suffix_mask, axis=-1)
            - 1
        )


        linen_attrs = {
            name: getattr(self.PaliGemma.llm, name)
            for name in self.PaliGemma.llm.linen_attributes
        }
        linen_variables = unfreeze(
            nnx_bridge_variables.nnx_attrs_to_linen_vars(linen_attrs)
        )
        linen_variables["params"]["layers"] = jax.tree.map(
            lambda value: value[:layer],
            linen_variables["params"]["layers"],
        )
        partial_configs = [
            dataclasses.replace(
                _gemma.get_config(variant),
                depth=layer,
            )
            for variant in (
                self.critic_paligemma_variant,
                self.critic_action_expert_variant,
            )
        ]
        partial_llm = _gemma.Module(
            configs=partial_configs,
            embed_dtype=self.critic_embed_dtype,
            adarms=True,
        )
        partial_cache = jax.tree.map(lambda value: value[:layer], kv_cache)
        (_, suffix_out), _ = partial_llm.apply(
            freeze(linen_variables),
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=partial_cache,
            adarms_cond=[None, adarms_cond],
        )
        return suffix_out[:, -self.action_horizon :]

    @staticmethod
    def _critic_head_attribute(base_name: str, head_index: int) -> str:
        if head_index < 0:
            raise ValueError("head_index must be non-negative")
        return base_name if head_index == 0 else f"{base_name}_{head_index}"

    def _critic_head_module(self, base_name: str, head_index: int):
        if head_index >= self.critic_ensemble_size:
            raise IndexError(
                f"critic head {head_index} is outside ensemble size "
                f"{self.critic_ensemble_size}"
            )
        return getattr(
            self,
            self._critic_head_attribute(base_name, head_index),
        )

    def _critic_score_features(self, suffix_features, head_index: int = 0):
        """Return critic features, optionally retaining the backbone path."""
        detached = (
            suffix_features
            if self.critic_update_backbone
            else jax.lax.stop_gradient(suffix_features)
        )
        if self.critic_adapter_hidden_dim <= 0:
            return detached
        features = detached
        depth = getattr(self, "critic_adapter_depth", 1)
        for block_index in range(depth):
            head_suffix = "" if head_index == 0 else f"_{head_index}"
            block_suffix = (
                head_suffix
                if block_index == 0
                else f"_block{block_index}{head_suffix}"
            )
            adapter_down = getattr(
                self, f"critic_adapter_down{block_suffix}"
            )
            adapter_up = getattr(self, f"critic_adapter_up{block_suffix}")
            residual = adapter_down(features)
            residual = nnx.relu(residual)
            residual = adapter_up(residual)
            features = features + residual
        return features

    def _suffix_forward(self, x_t, s_t, time, kv_cache, prefix_mask):
        """One-way joint flow: Actor -> detached progress critic."""
        suffix_features = self._suffix_features(
            x_t, s_t, time, kv_cache, prefix_mask
        )
        v_t = self.action_out_proj(suffix_features)
        if not self.stop_progress_gradient:
            raise ValueError(
                "V31 requires stop_progress_gradient=True; the flow critic "
                "must never update Actor features"
            )
        from openpi.models.pi0 import posemb_sincos

        progress_features = jax.lax.stop_gradient(suffix_features)
        progress_features = (
            progress_features
            + self.v31_flow_critic_s_proj(s_t)
            + self.v31_flow_critic_time_proj(
                posemb_sincos(
                    time,
                    self.action_in_proj.out_features,
                    min_period=4e-3,
                    max_period=4.0,
                )
            )[:, None, :]
        )
        for block_index in range(self.v31_flow_critic_depth):
            suffix = "" if block_index == 0 else f"_block{block_index}"
            down = getattr(self, f"v31_flow_critic_adapter_down{suffix}")
            up = getattr(self, f"v31_flow_critic_adapter_up{suffix}")
            progress_features = progress_features + up(
                nnx.relu(down(progress_features))
            )
        v_s = self.v31_flow_critic_out_proj(progress_features)
        return v_t, v_s

    def predict_critic_ensemble_logits(
        self,
        actions: _model.Actions,
        kv_cache,
        prefix_mask,
    ) -> at.Float[at.Array, "m b ah 1"]:
        """Score clean action chunks with all detached direct-critic branches."""
        if self.critic_adapter_hidden_dim <= 0:
            raise ValueError(
                "direct critic scoring requires critic_adapter_hidden_dim > 0"
            )
        batch_size = actions.shape[0]
        progress = jnp.zeros(
            (batch_size, self.action_horizon, 1),
            dtype=actions.dtype,
        )
        time = jnp.zeros((batch_size,), dtype=actions.dtype)
        suffix_features = self._critic_suffix_features(
            actions, progress, time, kv_cache, prefix_mask
        )
        all_logits = []
        for head_index in range(self.critic_ensemble_size):
            critic_features = self._critic_score_features(
                suffix_features, head_index
            )
            output_head = self._critic_head_module(
                "critic_score_out_proj", head_index
            )
            all_logits.append(output_head(critic_features))
        return jnp.stack(all_logits, axis=0)

    def predict_critic_logits(
        self,
        actions: _model.Actions,
        kv_cache,
        prefix_mask,
    ) -> at.Float[at.Array, "b ah 1"]:
        """Backward-compatible ensemble-mean clean critic logits."""
        return jnp.mean(
            self.predict_critic_ensemble_logits(
                actions,
                kv_cache,
                prefix_mask,
            ),
            axis=0,
        )

    def predict_critic_chunk_scores(
        self,
        actions: _model.Actions,
        kv_cache,
        prefix_mask,
    ) -> at.Float[at.Array, "m b"]:
        """Return one clean-endpoint chunk logit per ensemble head."""
        score_horizon = getattr(self, "mc_critic_score_horizon", 0)
        score_actions = actions
        if 0 < score_horizon < self.action_horizon:


            score_actions = actions.at[:, score_horizon:, :].set(0)
        logits = self.predict_critic_ensemble_logits(
            score_actions,
            kv_cache,
            prefix_mask,
        )
        if score_horizon > 0:
            logits = logits[:, :, :score_horizon, :]
        return jnp.mean(logits, axis=(2, 3))

    def compute_fpf_losses(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        progress_targets: at.Float[at.Array, "b ah 1"],
        kv_cache,
        prefix_mask,
        *,
        num_flow_samples: int = 15,
        action_weights=None,
    ) -> dict:
        """Flow matching loss (action + progress),observation 需已 preprocess。"""
        batch_size = actions.shape[0]

        def process_one_flow_sample(sample_rng):
            noise_rng, time_rng, progress_noise_rng = jax.random.split(sample_rng, 3)
            noise = jax.random.normal(noise_rng, actions.shape)
            time = jax.random.beta(time_rng, 1.5, 1, (batch_size,)) * 0.999 + 0.001
            time_expanded = time[:, None, None]

            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

            progress_noise = jax.random.normal(progress_noise_rng, (batch_size, self.action_horizon, 1))
            s_t = time_expanded * progress_noise + (1 - time_expanded) * progress_targets
            u_s = progress_noise - progress_targets

            v_t, v_s = self._suffix_forward(x_t, s_t, time, kv_cache, prefix_mask)
            action_loss = jnp.square(v_t - u_t)
            progress_loss = jnp.mean(jnp.square(v_s - u_s), axis=(-2, -1))
            return action_loss, progress_loss

        flow_rngs = jax.random.split(rng, num_flow_samples)
        with at.disable_typechecking():
            all_action_losses, all_progress_losses = jax.vmap(process_one_flow_sample)(flow_rngs)

        action_loss = jnp.mean(all_action_losses, axis=0)
        progress_loss = jnp.mean(all_progress_losses, axis=0)

        losses = {"progress_loss": progress_loss}
        if action_weights is not None:
            losses["action_loss"] = jnp.mean(action_weights * action_loss, axis=(-2, -1))
            losses["action_loss_unweighted"] = jnp.mean(action_loss, axis=(-2, -1))
        else:
            losses["action_loss"] = jnp.mean(action_loss, axis=(-2, -1))
        return losses

    def estimate_s_hat_nfe1(
        self, rng: at.KeyArrayLike, observation: _model.Observation, kv_cache, prefix_mask,
        k_samples: int = 1,
    ) -> at.Float[at.Array, "b ah 1"]:

        batch_size = observation.state.shape[0]

        def _single(sample_rng):
            noise_rng, progress_noise_rng = jax.random.split(sample_rng)
            action_noise = jax.random.normal(noise_rng, (batch_size, self.action_horizon, self.action_dim))
            progress_noise = jax.random.normal(progress_noise_rng, (batch_size, self.action_horizon, 1))
            time = jnp.ones((batch_size,))
            _, v_s = self._suffix_forward(action_noise, progress_noise, time, kv_cache, prefix_mask)
            return progress_noise - v_s

        if k_samples <= 1:
            return _single(rng)

        rngs = jax.random.split(rng, k_samples)
        with at.disable_typechecking():
            all_s_hat = jax.vmap(_single)(rngs)
        return jnp.mean(all_s_hat, axis=0)

    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """self.mc_num_candidates>1 时生成 K 个候选，再按配置做 scorer argmax
        或均匀随机选择。"""
        num_candidates = self.mc_num_candidates
        if num_candidates <= 1:
            actions, _ = self.sample_actions_with_progress(rng, observation, num_steps=num_steps, noise=noise)
            return actions

        batch_size = observation.state.shape[0]


        rngs = jax.random.split(rng, num_candidates)
        shared_progress_noise = None
        if self.mc_share_progress_noise:


            _, progress_rng = jax.random.split(rngs[0])
            shared_progress_noise = jax.random.normal(
                progress_rng,
                (batch_size, self.action_horizon, 1),
            )

        def single_sample(sample_rng):
            actions, progress = self.sample_actions_with_progress(
                sample_rng,
                observation,
                num_steps=num_steps,
                progress_noise=shared_progress_noise,
            )
            if self.mc_remove_initial_progress_noise:


                _, progress_rng = jax.random.split(sample_rng)
                progress_initial = jax.random.normal(
                    progress_rng,
                    (batch_size, self.action_horizon, 1),
                )
                progress = progress - progress_initial
            score = jnp.mean(progress, axis=(1, 2))
            return actions, score

        with at.disable_typechecking():
            all_actions, all_scores = jax.vmap(single_sample)(rngs)

        if self.mc_use_critic_score:


            observation_p = _model.preprocess_observation(
                None, observation, train=False
            )
            kv_cache, prefix_mask = self.compute_prefix_forward(observation_p)

            def score_candidate(candidate_actions):
                head_scores = self.predict_critic_chunk_scores(
                    candidate_actions,
                    kv_cache,
                    prefix_mask,
                )
                return aggregate_critic_ensemble_scores(
                    head_scores,
                    self.mc_critic_uncertainty_beta,
                )

            with at.disable_typechecking():
                all_scores = jax.vmap(score_candidate)(all_actions)

        if self.mc_selection_mode == "random":


            selection_rng = jax.random.fold_in(rng, 0x52414E44)
            selected_idx = jax.random.randint(
                selection_rng,
                (batch_size,),
                minval=0,
                maxval=num_candidates,
            )
        elif self.mc_selection_mode == "argmin":
            selected_idx = jnp.argmin(all_scores, axis=0)
        else:
            selected_idx = jnp.argmax(all_scores, axis=0)

        if getattr(self, "mc_log_diagnostics", False):


            sorted_scores = jnp.sort(all_scores[:, 0])
            selected_score = all_scores[selected_idx[0], 0]
            score_horizon = (
                getattr(self, "mc_critic_score_horizon", 0)
                or self.action_horizon
            )
            score_fields = {
                f"score_{candidate_index}": all_scores[candidate_index, 0]
                for candidate_index in range(num_candidates)
            }
            score_format = " ".join(
                f"s{candidate_index}={{score_{candidate_index}}}"
                for candidate_index in range(num_candidates)
            )
            jax.debug.print(
                "FPF_MC_DIAG "
                f"k={num_candidates} score_horizon={score_horizon} "
                "selected_idx={selected_idx} selected_score={selected_score} "
                "score_mean={score_mean} score_std={score_std} "
                "top_margin={top_margin} "
                + score_format,
                selected_idx=selected_idx[0],
                selected_score=selected_score,
                score_mean=jnp.mean(all_scores[:, 0]),
                score_std=jnp.std(all_scores[:, 0]),
                top_margin=sorted_scores[-1] - sorted_scores[-2],
                ordered=True,
                **score_fields,
            )
        return all_actions[selected_idx, jnp.arange(batch_size)]

    def sample_actions_with_progress(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        progress_noise: at.Float[at.Array, "b ah 1"] | None = None,
    ) -> tuple[_model.Actions, at.Float[at.Array, "b ah 1"]]:

        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        action_rng, progress_rng = jax.random.split(rng)
        if noise is None:
            noise = jax.random.normal(action_rng, (batch_size, self.action_horizon, self.action_dim))
        if progress_noise is None:
            progress_noise = jax.random.normal(progress_rng, (batch_size, self.action_horizon, 1))
        elif progress_noise.shape != (batch_size, self.action_horizon, 1):
            raise ValueError(
                "progress_noise must have shape "
                f"{(batch_size, self.action_horizon, 1)}, got {progress_noise.shape}"
            )

        kv_cache, prefix_mask = self.compute_prefix_forward(observation)

        def step(carry):
            x_t, s_t, time = carry
            v_t, v_s = self._suffix_forward(
                x_t, s_t, jnp.broadcast_to(time, batch_size), kv_cache, prefix_mask
            )
            return x_t + dt * v_t, s_t + dt * v_s, time + dt

        def cond(carry):
            _, _, time = carry
            return time >= -dt / 2

        x_0, s_0, _ = jax.lax.while_loop(cond, step, (noise, progress_noise, 1.0))
        return x_0, s_0
