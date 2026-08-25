"""
LyTTA -- Lyapunov-regularized test-time adaptation.

TTA as constrained online convex optimization. The unsupervised surrogate
(entropy) is the cost f_t; departure from a certified anchor is the constraint

    g_t(theta) = mean_x KL( f_{theta_0}(x) || f_theta(x) ) - tau ,

and the adaptation minimizes the COCO surrogate

    f_t(theta) + mu_t * g_t^+(theta),        mu_t = Phi'(Q(t)),
    Q(t) = sum_{s<=t} g_s^+(theta_s)         (cumulative constraint violation)

so the pull toward the anchor GROWS with accumulated violation instead of being
a fixed hyperparameter. This is the whole claim: EATA/ROID/CoTTA all apply an
anchored penalty with a CONSTANT weight; ASR guesses lambda_0 * phi_t^2. Here
the weight is derived from the Lyapunov analysis.

ANCHOR (non-negotiable, measured 2026-08-09b):
The anchor is the BN-ADAPTED source, not the raw checkpoint. Entropy-minimization
methods null BN running statistics, so a raw-checkpoint anchor sits on a
stream-dependent ~5.9 nat normalization floor and g(theta_0) != -tau. With the
BN-adapted anchor, g(theta_0) = -tau to machine precision at round 0: assumption
C2 holds exactly, which is what the theory requires.

CONFIG (add to conf.py, see notes/LYTTA.md; env vars override for quick sweeps):
    LYTTA.TAU        constraint margin, nats            (default 1.0)
    LYTTA.PHI        'quad' | 'exp' | 'const'           (default 'quad')
    LYTTA.LAMBDA     Lyapunov scale                     (default 0.01 / 0.05)
    LYTTA.MU_MAX     clamp on mu_t (stability)          (default 50.0)
    LYTTA.Q_DECAY    1.0 = pure cumulative; <1 = EMA    (default 1.0)
    LYTTA.PENALTY    'gplus' | 'g'                      (default 'gplus')
"""
import os
import logging
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base import TTAMethod
from models.model import get_model
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy
from monitor.constraint_monitor import monitor_step

logger = logging.getLogger(__name__)


def _opt(cfg, key, default, cast=float):
    """LYTTA.<key> from cfg, overridden by env LYTTA_<KEY>, else default."""
    env = os.environ.get(f"LYTTA_{key}")
    if env is not None:
        return cast(env)
    node = getattr(cfg, "LYTTA", None)
    if node is not None and hasattr(node, key):
        return cast(getattr(node, key))
    return default


@ADAPTATION_REGISTRY.register()
class LyTTA(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.softmax_entropy = Entropy()

        self.tau = _opt(cfg, "TAU", 1.0)
        self.phi = _opt(cfg, "PHI", "quad", str).lower()
        self.mu_max = _opt(cfg, "MU_MAX", 50.0)
        self.q_decay = _opt(cfg, "Q_DECAY", 1.0)
        self.penalty = _opt(cfg, "PENALTY", "gplus", str).lower()
        default_lam = {"quad": 0.01, "exp": 0.05, "const": 1.0}.get(self.phi, 0.01)
        self.lam = _opt(cfg, "LAMBDA", default_lam)

        # ---- certified anchor: fresh checkpoint, BN forced to batch stats ----
        anchor, preprocess = get_model(cfg, num_classes, self.device)
        anchor.model_preprocess = preprocess
        n_bn = 0
        for m in anchor.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
                n_bn += 1
            elif isinstance(m, nn.BatchNorm1d):
                m.train()
        anchor.eval().requires_grad_(False)
        self.anchor = anchor

        self.Q = 0.0          # cumulative constraint violation
        self.mu = 0.0         # current multiplier (logged)
        self.n_violations = 0

        logger.info(
            f"LyTTA: phi={self.phi} lambda={self.lam} tau={self.tau} "
            f"mu_max={self.mu_max} q_decay={self.q_decay} penalty={self.penalty} "
            f"| anchor: batch stats forced on {n_bn} BN2d layers")

        self.models = [self.model]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

    # ------------------------------------------------------------------ Phi'
    def _mu(self, Q):
        """Phi'(Q): the DERIVED multiplier.

        quad   Phi(x)=lambda x^2      -> Phi'(Q) = 2 lambda Q     (linear response;
                                        matches the quadratic Lyapunov used by the
                                        projection-free COCO algorithm)
        exp    Phi(x)=e^{lambda x}    -> Phi'(Q) = lambda e^{lambda Q}
                                        (the Sinha-Vaze potential; sharper)
        const  Phi'(Q) = lambda       -> ABLATION: same constraint, no schedule.
                                        This is the EATA/ROID-style fixed weight
                                        and it is the control that isolates the
                                        contribution of the schedule itself.
        """
        if self.phi == "const":
            mu = self.lam
        elif self.phi == "exp":
            mu = self.lam * float(torch.exp(torch.tensor(
                min(self.lam * Q, 20.0))))          # clamp exponent, not just mu
        else:
            mu = 2.0 * self.lam * Q
        return float(min(mu, self.mu_max))

    # -------------------------------------------------------------- the loss
    def loss_calculation(self, x):
        imgs_test = x[0]
        outputs = self.model(imgs_test)

        # cost f_t: unsupervised surrogate
        loss_cost = self.softmax_entropy(outputs).mean(0)

        # constraint g_t: KL to the certified anchor, minus the margin.
        # grad wrt logits is (p - p0), i.e. a pull back toward the anchor.
        with torch.no_grad():
            p0 = F.softmax(self.anchor(imgs_test).float(), dim=-1)
        logp = F.log_softmax(outputs.float(), dim=-1)
        kl = (p0 * (p0.clamp_min(1e-12).log() - logp)).sum(-1).mean()
        g = kl - self.tau
        g_pen = F.relu(g) if self.penalty == "gplus" else g

        # Q(t) = Q(t-1) + g_t^+  BEFORE forming mu_t, per the protocol ordering
        gplus_val = float(F.relu(g).detach())
        self.Q = self.q_decay * self.Q + gplus_val
        self.mu = self._mu(self.Q)
        if gplus_val > 0:
            self.n_violations += 1

        loss = loss_cost + self.mu * g_pen
        return outputs, loss

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss = self.loss_calculation(x)
            monitor_step(self, x, outputs)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss = self.loss_calculation(x)
            monitor_step(self, x, outputs)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        if hasattr(self, "monitor"):
            self.monitor.lytta_Q = getattr(self.monitor, "lytta_Q", [])
            self.monitor.lytta_mu = getattr(self.monitor, "lytta_mu", [])
            self.monitor.lytta_Q.append(self.Q)
            self.monitor.lytta_mu.append(self.mu)
        return outputs

    def reset(self):
        super().reset()
        self.Q = 0.0
        self.mu = 0.0
        self.n_violations = 0

    def collect_params(self):
        params, names = [], []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np_, p in m.named_parameters():
                    if np_ in ["weight", "bias"]:
                        params.append(p)
                        names.append(f"{nm}.{np_}")
        return params, names

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
