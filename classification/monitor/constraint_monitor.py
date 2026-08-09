"""
classification/monitor/constraint_monitor.py

Phase-1 constraint instrumentation for the Marsden TTA codebase.
Observation-only: never touches gradients, never mutates the adapting model.

WHY THE ANCHOR IS BUILT FROM A FRESH CHECKPOINT
-----------------------------------------------
TTAMethod.__init__ calls configure_model() BEFORE copy_model_and_optimizer(),
and Tent-family configure_model() sets, on every BatchNorm2d:
    m.track_running_stats = False; m.running_mean = None; m.running_var = None
So self.model_states[0] is already stripped of running statistics: a source
copy restored from it would use TEST-BATCH stats, not source stats, and every
KL/agreement number would be silently wrong. We therefore rebuild the anchor
by calling the repo's own get_model(cfg, num_classes, device), which returns a
pristine pre-trained model, and freeze it.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import get_model


@torch.no_grad()
def raw_stats(logits, source_logits):
    """tau-free constraint statistics. logits/source_logits: (n, C)."""
    logp = F.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    p0 = F.softmax(source_logits.float(), dim=-1)
    C = logits.shape[-1]

    kl = (p0 * (p0.clamp_min(1e-12).log() - logp)).sum(-1).mean()      # (a) source-KL
    pbar = p.mean(0)
    me = np.log(C) + (pbar * pbar.clamp_min(1e-12).log()).sum()        # (b) marginal-entropy deficit
    pz = F.softmax(logits.float().mean(0), -1)
    conc = np.log(C) + (pz * pz.clamp_min(1e-12).log()).sum()          # (c) ASR-style concentration
    ent = -(p * logp).sum(-1).mean()                                   # mean entropy
    agree = (logits.argmax(-1) == source_logits.argmax(-1)).float().mean()
    return {'kl': kl.item(), 'me': me.item(), 'conc': conc.item(),
            'ent': ent.item(), 'agree': agree.item()}


class ConstraintMonitor:
    KEYS = ['kl', 'me', 'conc', 'ent', 'agree']

    def __init__(self, params=None):
        self.rows = {k: [] for k in self.KEYS}
        self.batch_acc, self.probe_acc, self.domain = [], [], []
        self.path = [0.0]
        self._prev = None
        self._params = params
        self.block_acc = np.nan   # set by test_time.py before save()

    @torch.no_grad()
    def update(self, logits, source_logits, batch_acc=None, domain=None):
        for k, v in raw_stats(logits, source_logits).items():
            self.rows[k].append(v)
        self.batch_acc.append(np.nan if batch_acc is None else float(batch_acc))
        self.domain.append(domain if domain is not None else -1)
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
                 **{k: np.array(v) for k, v in self.rows.items()})
        print(f"[monitor] saved {len(self.rows['kl'])} rounds -> {path}")


def attach_monitor(method, cfg, num_classes, device="cuda"):
    """Give a TTAMethod instance `.monitor` and `.source_model`.

    The anchor is a FRESHLY loaded pre-trained model (see module docstring):
    pristine BN running statistics, eval mode, gradients disabled.
    """
    src, preprocess = get_model(cfg, num_classes, device)
    src.model_preprocess = preprocess
    src.eval().requires_grad_(False)

    n_bn = n_stats = 0
    for m in src.modules():
        if isinstance(m, nn.BatchNorm2d):
            n_bn += 1
            n_stats += int(m.running_mean is not None)
    if n_bn and n_stats == 0:
        raise RuntimeError(
            f"Anchor has {n_bn} BatchNorm2d layers but no running statistics. "
            "get_model() returned an already-configured model.")

    method.source_model = src
    method.monitor = ConstraintMonitor(params=method.params)
    print(f"[monitor] anchor built from fresh checkpoint "
          f"({n_stats}/{n_bn} BN layers carry running stats)")
    return method
