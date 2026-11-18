"""Behaviour cloning of the SAC actor on the demonstration data.

Loss = -mean log pi(clip(a, +-0.995) | s) of the EXECUTED (noisy) actions under the actor's squashed Gaussian, so tanh(mean) fits the clean demonstration action and sigma settles near the
noise level. Own Adam (the actor's RL optimizer stays fresh). 90/10 split by episode, early stopping on the validation loss, the best epoch's weights are kept.
"""
import copy
import time

import numpy as np
import torch as th

from .demos import with_index


def _nll(actor, obs, act, clip):
    mean, log_std, _ = actor.get_action_dist_params(obs)
    dist = actor.action_dist.proba_distribution(mean, log_std)
    return -dist.log_prob(act.clamp(-clip, clip)).mean()


def train_bc(actor, demos, *, seed=0, lr=3e-4, batch_size=512, max_epochs=30, patience=3, val_fraction=0.1, split_seed=0, action_clip=0.995, log=print):
    t0 = time.perf_counter()
    d = with_index(demos)
    n_ep = len(d["ep_start_tr"])
    perm = np.random.default_rng(split_seed).permutation(n_ep)
    n_val = min(max(1, int(round(val_fraction * n_ep))), n_ep - 1)
    val_eps = np.sort(perm[:n_val])
    in_val = np.isin(d["episode"], val_eps)
    obs = th.as_tensor(d["obs"][d["obs_index"]], dtype=th.float32)
    act = th.as_tensor(d["actions"], dtype=th.float32)
    tr_i, va_i = th.as_tensor(np.flatnonzero(~in_val)), th.as_tensor(np.flatnonzero(in_val))
    gen = th.Generator().manual_seed(int(seed))
    opt = th.optim.Adam(actor.parameters(), lr=lr)
    actor.set_training_mode(True)

    def val_nll():
        with th.no_grad():
            return sum(float(_nll(actor, obs[c], act[c], action_clip)) * len(c) for c in va_i.split(8192)) / len(va_i)

    best, best_epoch, bad = val_nll(), 0, 0
    best_state = copy.deepcopy(actor.state_dict())
    history = [{"epoch": 0, "train_nll": None, "val_nll": best}]
    for epoch in range(1, max_epochs + 1):
        losses = []
        for batch in tr_i[th.randperm(len(tr_i), generator=gen)].split(batch_size):
            loss = _nll(actor, obs[batch], act[batch], action_clip)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        v = val_nll()
        history.append({"epoch": epoch, "train_nll": float(np.mean(losses)), "val_nll": v})
        log(f"BC epoch {epoch}: train NLL {np.mean(losses):.4f}, val NLL {v:.4f}")
        if v < best:
            best, best_epoch, bad = v, epoch, 0
            best_state = copy.deepcopy(actor.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    actor.load_state_dict(best_state)
    actor.set_training_mode(False)
    with th.no_grad():
        mean, log_std, _ = actor.get_action_dist_params(obs[va_i])
        pred, sigma = th.tanh(mean).numpy(), log_std.exp().numpy()
    vi = va_i.numpy()
    return {"epochs_run": len(history) - 1, "best_epoch": best_epoch, "best_val_nll": best, "history": history, "n_train": int(len(tr_i)), "n_val": int(len(vi)),
            "val_mse_executed_per_dim": ((pred - d["actions"][vi]) ** 2).mean(0).tolist(), "val_mse_clean_per_dim": ((pred - d["actions_clean"][vi]) ** 2).mean(0).tolist(),
            "mean_sigma_per_dim": sigma.mean(0).tolist(), "wall_s": time.perf_counter() - t0}
