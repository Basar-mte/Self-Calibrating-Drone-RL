"""Soft Actor-Critic (Haarnoja et al., 2018), written out in full.

SAC learns two things at once:
    an actor   pi(a | s)  a Gaussian over actions, squashed by tanh into [-1, 1]
    two critics Q1, Q2    how much (discounted, entropy-regularised) reward follows (s, a)

and tunes a temperature alpha so that the policy keeps a target amount of randomness
(entropy). The pieces, each a few lines below:

    critic target   y = r + gamma (1 - done) [ min(Q1', Q2')(s', a') - alpha log pi(a' | s') ],  a' ~ pi(s')
    critic loss     (Q1(s, a) - y)^2 + (Q2(s, a) - y)^2
    actor loss      alpha log pi(a | s) - min(Q1, Q2)(s, a),     a ~ pi(s) by the reparameterisation trick
    alpha loss      -log(alpha) (log pi(a | s) + target_entropy)
    target critics  slowly track the critics (Polyak averaging with rate tau)

Taking the minimum of two critics counters the over-estimation that a single learned
Q-function suffers from (as in TD3); the entropy term keeps exploring without an
epsilon schedule.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def mlp(sizes, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Squashed Gaussian policy: a = tanh(mu + std * noise)."""

    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.body = mlp([obs_dim, *hidden], out_act=nn.ReLU())
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Linear(hidden[-1], act_dim)

    def forward(self, obs, deterministic=False):
        h = self.body(obs)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        u = mu if deterministic else mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        # log-probability of a: Gaussian density of u, corrected for the tanh squashing
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp -= (2.0 * (np.log(2.0) - u - F.softplus(-2.0 * u))).sum(-1)
        return a, logp


class Critic(nn.Module):
    """Two independent Q-networks, each mapping (s, a) to a number."""

    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, *hidden, 1])
        self.q2 = mlp([obs_dim + act_dim, *hidden, 1])

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class ReplayBuffer:
    """Circular memory of transitions (s, a, r, s', done), stored as float32 NumPy arrays."""

    def __init__(self, obs_dim, act_dim, capacity):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(self.capacity, dtype=np.float32)
        self.done = np.zeros(self.capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add_batch(self, obs, act, rew, next_obs, done):
        k = len(obs)
        idx = (self.ptr + np.arange(k)) % self.capacity
        self.obs[idx] = obs
        self.act[idx] = act
        self.rew[idx] = rew
        self.next_obs[idx] = next_obs
        self.done[idx] = done
        self.ptr = (self.ptr + k) % self.capacity
        self.size = min(self.size + k, self.capacity)

    def sample(self, batch_size, rng, device):
        idx = rng.integers(0, self.size, batch_size)
        to = lambda x: torch.as_tensor(x[idx], device=device)
        return to(self.obs), to(self.act), to(self.rew), to(self.next_obs), to(self.done)


class SACAgent:
    def __init__(self, obs_dim, act_dim, cfg, device):
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.device = device
        self.gamma = float(cfg["gamma"])
        self.tau = float(cfg["tau"])
        hidden = list(cfg["hidden"])
        lr = float(cfg["lr"])
        self.actor = Actor(obs_dim, act_dim, hidden).to(device)
        self.critic = Critic(obs_dim, act_dim, hidden).to(device)
        self.critic_targ = Critic(obs_dim, act_dim, hidden).to(device)
        self.critic_targ.load_state_dict(self.critic.state_dict())
        for p in self.critic_targ.parameters():
            p.requires_grad_(False)
        self.log_alpha = torch.tensor(np.log(float(cfg["init_alpha"])), device=device, requires_grad=True)
        te = cfg.get("target_entropy", "auto")
        self.target_entropy = -float(act_dim) if te == "auto" else float(te)
        # A CUDA graph records the whole update once and replays it with a single launch.
        # Same arithmetic, roughly 10x faster for networks this small (see docs/03).
        cg = cfg.get("cuda_graph", "auto")
        self.use_graph = (device.type == "cuda") and cg in ("auto", True)
        opt = dict(lr=lr, capturable=self.use_graph)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), **opt)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), **opt)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], **opt)
        self.updates = 0
        self._graph = None
        self._static_in = None
        self._stats = None

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        o = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        a, _ = self.actor(o, deterministic=deterministic)
        return a.cpu().numpy()

    def update(self, batch):
        """One gradient step on critics, actor and temperature from a sampled batch."""
        if not self.use_graph:
            self._stats = self._update_tensors(*batch)
        elif self._graph is None:
            self._capture(batch)
        else:
            for static, x in zip(self._static_in, batch):
                static.copy_(x)
            self._graph.replay()
        self.updates += 1

    def stats(self):
        """Losses and temperature of the most recent update, as plain floats."""
        if self._stats is None:
            return {}
        v = self._stats.detach().cpu().numpy()
        return {"loss_q": float(v[0]), "loss_pi": float(v[1]), "alpha": float(v[2]),
                "entropy": float(v[3]), "q": float(v[4])}

    def _capture(self, batch):
        self._static_in = [x.clone() for x in batch]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):                       # warm-up steps, required before capture
            for _ in range(3):
                self._update_tensors(*self._static_in)
        torch.cuda.current_stream().wait_stream(side)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._stats = self._update_tensors(*self._static_in)

    def _update_tensors(self, obs, act, rew, next_obs, done):
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            a2, logp2 = self.actor(next_obs)
            q1t, q2t = self.critic_targ(next_obs, a2)
            y = rew + self.gamma * (1.0 - done) * (torch.min(q1t, q2t) - alpha * logp2)
        q1, q2 = self.critic(obs, act)
        loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_critic.zero_grad(set_to_none=True)
        loss_q.backward()
        self.opt_critic.step()

        for p in self.critic.parameters():
            p.requires_grad_(False)
        a, logp = self.actor(obs)
        q1p, q2p = self.critic(obs, a)
        loss_pi = (alpha * logp - torch.min(q1p, q2p)).mean()
        self.opt_actor.zero_grad(set_to_none=True)
        loss_pi.backward()
        self.opt_actor.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        loss_alpha = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad(set_to_none=True)
        loss_alpha.backward()
        self.opt_alpha.step()

        with torch.no_grad():                               # target critics: Polyak averaging
            torch._foreach_lerp_(list(self.critic_targ.parameters()),
                                 list(self.critic.parameters()), self.tau)
        return torch.stack([loss_q.detach(), loss_pi.detach(), alpha,
                            -logp.detach().mean(), q1.detach().mean()])

    def save(self, path, extra=None):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "critic_targ": self.critic_targ.state_dict(), "log_alpha": self.log_alpha.detach().cpu(),
                    "obs_dim": self.obs_dim, "act_dim": self.act_dim, **(extra or {})}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.critic_targ.load_state_dict(ck["critic_targ"])
        with torch.no_grad():
            self.log_alpha.copy_(ck["log_alpha"].to(self.device))
        return ck


class SACController:
    """Wraps a trained actor in the same act(obs, env) interface as the PID controller."""

    def __init__(self, agent, name="SAC"):
        self.agent = agent
        self.name = name

    def reset(self, n=None):
        pass

    def act(self, obs, env):
        return self.agent.act(obs, deterministic=True)
