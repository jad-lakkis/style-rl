"""SAC for the driving environment: LayerNorm critics, 3-step returns, and a SAC whose actor and entropy coefficient can be frozen (the staged start).

Staged start (see train.py): the actor is first copied from the scripted driver (behaviour cloning), then the CRITICS are trained on the demonstration data with the actor frozen, then the
online phase starts with the actor still frozen for a while, then the actor is released. The freezing relies on SB3 SAC.train() internals (tested on 2.9.x):
  frozen actor  the actor parameters get requires_grad = False: actor_loss.backward() still works (it depends on the critic), Adam skips parameters without a gradient;
  frozen alpha  ent_coef_optimizer is set to None for the duration of train(), so SB3 takes its fixed-coefficient branch.
"""
import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.utils import update_learning_rate
from stable_baselines3.sac.policies import SACPolicy
from torch import nn


class LayerNormContinuousCritic(ContinuousCritic):
    """SB3's critic with a LayerNorm after every hidden Linear of the Q networks (none after the output)."""

    def __init__(self, observation_space, action_space, net_arch, features_extractor, features_dim, activation_fn=nn.ReLU, normalize_images=True, n_critics=2,
                 share_features_extractor=True):
        super().__init__(observation_space, action_space, net_arch, features_extractor, features_dim, activation_fn=activation_fn, normalize_images=normalize_images,
                         n_critics=n_critics, share_features_extractor=share_features_extractor)
        action_dim = get_action_dim(self.action_space)
        for idx in range(n_critics):
            q = nn.Sequential(*create_mlp(features_dim + action_dim, 1, net_arch, activation_fn, post_linear_modules=[nn.LayerNorm]))
            self.add_module(f"qf{idx}", q)
            self.q_networks[idx] = q


class LNSACPolicy(SACPolicy):
    def make_critic(self, features_extractor=None):
        kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return LayerNormContinuousCritic(**kwargs).to(self.device)


class StagedSAC(SAC):
    def __init__(self, *args, **kwargs):
        self.freeze_actor = False
        self.freeze_ent_coef = False
        self.actor_lr = None
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        super()._setup_model()
        self.set_frozen(self.freeze_actor, self.freeze_ent_coef)       # after load(): requires_grad is not saved

    def set_frozen(self, actor, ent_coef):
        self.freeze_actor, self.freeze_ent_coef = bool(actor), bool(ent_coef)
        for p in self.actor.parameters():
            p.requires_grad_(not actor)

    def _update_learning_rate(self, optimizers):
        super()._update_learning_rate(optimizers)                        # SB3 resets every learning rate at the start of train()
        if self.actor_lr is not None:
            update_learning_rate(self.actor.optimizer, self.actor_lr)

    def train(self, gradient_steps, batch_size=64):
        if not (self.freeze_ent_coef and self.ent_coef_optimizer is not None):
            return super().train(gradient_steps, batch_size)
        opt = self.ent_coef_optimizer
        self.ent_coef_optimizer = None
        self.ent_coef_tensor = th.exp(self.log_ent_coef.detach())
        try:
            super().train(gradient_steps, batch_size)
        finally:
            self.ent_coef_optimizer = opt

    @property
    def alpha(self):
        return float(th.exp(self.log_ent_coef.detach()).item()) if self.log_ent_coef is not None else float(self.ent_coef_tensor.item())


def build_model(vec_env, *, hidden=(256, 256, 256), lr=3e-4, batch_size=512, buffer_size=1_000_000, tau=0.005, gamma=0.992, n_steps=3, gradient_steps=2, ent_init=0.1,
                seed=0, device="cpu", layernorm=True):
    arch = list(hidden)
    kw = dict(net_arch=dict(pi=arch, qf=list(arch)), activation_fn=nn.ReLU, n_critics=2, share_features_extractor=False)
    return StagedSAC(LNSACPolicy if layernorm else "MlpPolicy", vec_env, learning_rate=lr, buffer_size=buffer_size, learning_starts=0, batch_size=batch_size, tau=tau, gamma=gamma,
                     train_freq=(1, "step"), gradient_steps=gradient_steps, n_steps=n_steps, ent_coef=f"auto_{ent_init}", target_entropy="auto", use_sde=False, policy_kwargs=kw,
                     seed=seed, device=device, verbose=0)


def init_output_bias(actor, action=(0.0, -0.275, -0.8), log_std=-1.0, weight_scale=0.01):
    """Before behaviour cloning: the actor's mean output is ~`action` for any observation (straight, light throttle, brake in the dead zone), sigma ~ exp(log_std)."""
    target = np.asarray(action, dtype=np.float64)
    with th.no_grad():
        actor.mu.weight.mul_(weight_scale)
        actor.mu.bias.copy_(th.as_tensor(np.arctanh(target), dtype=actor.mu.bias.dtype))
        actor.log_std.weight.mul_(weight_scale)
        actor.log_std.bias.fill_(log_std)


def snapshot_state(model):
    """In-memory copy of everything the collapse guard must be able to restore: actor, critics, targets, alpha, optimizers."""
    import copy
    pol = model.policy
    snap = {"policy": copy.deepcopy(pol.state_dict()), "actor_opt": copy.deepcopy(model.actor.optimizer.state_dict()), "critic_opt": copy.deepcopy(model.critic.optimizer.state_dict()),
            "critic_target": copy.deepcopy(model.critic_target.state_dict())}
    if model.log_ent_coef is not None:
        snap["log_alpha"] = model.log_ent_coef.detach().clone()
        snap["alpha_opt"] = copy.deepcopy(model.ent_coef_optimizer.state_dict())
    return snap


def restore_state(model, snap):
    model.policy.load_state_dict(snap["policy"])
    model.actor.optimizer.load_state_dict(snap["actor_opt"])
    model.critic.optimizer.load_state_dict(snap["critic_opt"])
    model.critic_target.load_state_dict(snap["critic_target"])
    if "log_alpha" in snap:
        with th.no_grad():
            model.log_ent_coef.copy_(snap["log_alpha"])
        model.ent_coef_optimizer.load_state_dict(snap["alpha_opt"])
