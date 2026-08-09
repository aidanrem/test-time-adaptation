"""
classification/monitor/constraint_monitor.py

Phase-1 constraint instrumentation for the Marsden TTA codebase.
Observation-only: never touches gradients, never mutates the adapting model.

TWO ANCHORS, LOGGED SIMULTANEOUSLY
----------------------------------
Measurement on 2026-08-09 showed that source-KL against the raw checkpoint
tracks CORRUPTION DIFFICULTY rather than model health: at round 0, before any
gradient step, source and adapting model already disagreed on ~67% of samples
(gaussian_noise kl=5.93, agree=0.33), because Tent's configure_model() nulls BN
running statistics so the adapting model uses TEST-BATCH stats from the first
forward while the raw checkpoint uses SOURCE running stats.

We therefore carry two frozen anchors with IDENTICAL WEIGHTS:

  anchor_src  BN running statistics intact (the pre-trained model as shipped).
              kl_src = normalization gap + parameter drift.

  anchor_bn   BN forced to batch statistics (running_mean/var = None), i.e. the
              `norm_test` / BN-adaptation baseline. At round 0 this model is
              EXACTLY the adapting model, so kl_bn[0] == 0 and agree_bn[0] == 1
              by construction -- restoring C2 (g(theta_0) = -tau) exactly, and
              isolating PARAMETER drift from the normalization gap.

Logging both in one run gives a paired comparison, which matters because the
pipeline is nondeterministic across runs by ~0.8pp (see lab.md 2026-08-09):
two separate runs could not be compared at this resolution.

Cost: one extra frozen forward pass per batch. Set MONITOR_ANCHORS=src or
MONITOR_ANCHORS=bn to log only one if that cost ever matters.
"""
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import get_model


# --------------------------------------------------------------------- stats
@torch.no_grad()
def _pairwise(logits, source_logits):
    """Statistics that compare the adapting model against one anchor."""
    logp = F.log_softmax(logits.float(), dim=-1)
    p0 = F.softmax(source_logits.float(), dim=-1)
    kl = (p0 * (p0.clamp_min(1e-12).log() - logp)).sum(-1).mean()
    agree = (logits.argmax(-1) == source_logits.argmax(-1)).float().mean()
    return kl.item(), agree.item()


@torch.no_grad()
def _self_stats(logits):
    """Statistics of the adapting model alone (anchor-free)."""
    logp = F.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    C = logits.shape[-1]
    pbar = p.mean(0)
    me = np.log(C) + (pbar * pbar.clamp_min(1e-12).log()).sum()     # (b) marginal-entropy deficit
    pz = F.softmax(logits.float().mean(0), -1)
    conc = np.log(C) + (pz * pz.clamp_min(1e-12).log()).sum()       # (c) ASR-style concentration
    ent = -(p * logp).sum(-1).mean()                                # mean entropy
    return me.item(), conc.item(), ent.item()


# ------------------------------------------------------------------- monitor
class ConstraintMonitor:
    KEYS = ['kl_src', 'agree_src', 'kl_bn', 'agree_bn', 'me', 'conc', 'ent']

    def __init__(self, params=None, anchors=('src', 'bn')):
        self.anchors = tuple(anchors)
        self.rows = {k: [] for k in self.KEYS}
        self.batch_acc, self.probe_acc, self.domain = [], [], []
        self.path = [0.0]
        self._prev = None
        self._params = params
        self.block_acc = np.nan   # set by test_time.py before save()

    @torch.no_grad()
    def update(self, logits, src_logits=None, bn_logits=None,
               batch_acc=None, domain=None):
        logits = logits.detach()

        if src_logits is not None:
            kl, ag = _pairwise(logits, src_logits)
        else:
            kl, ag = np.nan, np.nan
        self.rows['kl_src'].append(kl)
        self.rows['agree_src'].append(ag)

        if bn_logits is not None:
            kl, ag = _pairwise(logits, bn_logits)
        else:
            kl, ag = np.nan, np.nan
        self.rows['kl_bn'].append(kl)
        self.rows['agree_bn'].append(ag)

        me, conc, ent = _self_stats(logits)
        self.rows['me'].append(me)
        self.rows['conc'].append(conc)
        self.rows['ent'].append(ent)

        self.batch_acc.append(np.nan if batch_acc is None else float(batch_acc))
        self.domain.append(-1 if domain is None else domain)

        if self._params:
            flat = torch.cat([p.detach().flatten() for p in self._params])
            if self._prev is not None:
                self.path.append(self.path[-1] + (flat - self._prev).norm().item())
            self._prev = flat.clone()

    def save(self, path):
        np.savez(path,
                 batch_acc=np.array(self.batch_acc, dtype=float),
                 probe_acc=np.array(self.probe_acc, dtype=float),
                 domain=np.array(self.domain),
                 P_T=np.array(self.path),
                 block_acc=np.array(self.block_acc, dtype=float),
                 anchors=np.array(list(self.anchors)),
                 **{k: np.array(v) for k, v in self.rows.items()})
        print(f"[monitor] saved {len(self.rows['ent'])} rounds -> {path}")


# -------------------------------------------------------------------- anchors
def _force_batch_stats(model):
    """Make every BatchNorm2d use batch statistics (the `norm_test` baseline).

    In PyTorch, a BatchNorm layer in eval mode uses batch statistics when
    running_mean/running_var are None -- this is exactly the mechanism Tent's
    configure_model() relies on. No gradients are enabled here.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            n += 1
        elif isinstance(m, nn.BatchNorm1d):
            m.train()
    return n


def attach_monitor(method, cfg, num_classes, device="cuda"):
    """Give a TTAMethod instance `.monitor`, `.anchor_src`, `.anchor_bn`.

    Both anchors are built from a FRESHLY loaded checkpoint (never from
    method.model_states, which configure_model() has already stripped of BN
    running statistics) and share identical weights.
    """
    which = os.environ.get("MONITOR_ANCHORS", "src,bn").split(",")
    which = tuple(w.strip() for w in which if w.strip() in ("src", "bn"))
    if not which:
        which = ("src", "bn")

    base, preprocess = get_model(cfg, num_classes, device)
    base.model_preprocess = preprocess

    anchor_src = anchor_bn = None

    if "src" in which:
        anchor_src = base
        anchor_src.eval().requires_grad_(False)
        n_bn = sum(isinstance(m, nn.BatchNorm2d) for m in anchor_src.modules())
        n_stats = sum(isinstance(m, nn.BatchNorm2d) and m.running_mean is not None
                      for m in anchor_src.modules())
        if n_bn and n_stats == 0:
            raise RuntimeError(
                f"anchor_src has {n_bn} BatchNorm2d layers but no running "
                "statistics -- get_model() returned a configured model.")
        print(f"[monitor] anchor_src: running BN stats on {n_stats}/{n_bn} layers")

    if "bn" in which:
        anchor_bn = deepcopy(base) if anchor_src is not None else base
        anchor_bn.eval().requires_grad_(False)
        n = _force_batch_stats(anchor_bn)
        print(f"[monitor] anchor_bn: forced batch stats on {n} BN2d layers "
              "(expect kl_bn[0]==0, agree_bn[0]==1)")

    method.anchor_src = anchor_src
    method.anchor_bn = anchor_bn
    method.monitor = ConstraintMonitor(params=method.params, anchors=which)
    return method


# ------------------------------------------------------------ hook for methods
@torch.no_grad()
def monitor_step(method, x, outputs):
    """Call from a method's forward_and_adapt, BEFORE the update step.

    No-op when the monitor is not attached, so upstream behaviour is unchanged
    whenever MONITOR_DIR is unset.
    """
    if not hasattr(method, "monitor"):
        return
    imgs = x[0] if isinstance(x, (list, tuple)) else x
    src_logits = method.anchor_src(imgs) if method.anchor_src is not None else None
    bn_logits = method.anchor_bn(imgs) if method.anchor_bn is not None else None
    method.monitor.update(outputs, src_logits=src_logits, bn_logits=bn_logits)
