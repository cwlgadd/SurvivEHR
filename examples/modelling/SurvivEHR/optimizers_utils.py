import math
import numpy as np
from typing import Iterable, Tuple, List, Dict, Any, Optional, Sequence
from torch.nn import Parameter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def split_decay_groups(
    named_params: Iterable[Tuple[str, Parameter]],
    lr: float,
    weight_decay: Optional[float] = None,
    split: bool = True,
) -> List[Dict[str, Any]]:
    """
    Build optimizer parameter groups with an option to split decay vs. no-decay params.

    When `split=True` and `weight_decay` is provided, parameters are split into two groups:
    (1) a decayed group (typical weights) and (2) a non-decayed group (biases, LayerNorms,
    and tensors with ndim < 2). When `split=False`, a single group is returned containing all
    trainable parameters; if `weight_decay` is provided, it is applied uniformly.

    Args:
        named_params (Iterable[Tuple[str, torch.nn.Parameter]]):
            An iterable of (name, parameter) pairs, e.g. `model.named_parameters()`.
        lr (float): Learning rate to assign to all returned groups.
        weight_decay (Optional[float]): Weight decay coefficient. If provided and `split=True`,
            it is applied only to the decayed group; the no-decay group uses 0.0. If `split=False`,
            it is applied to the single combined group.
        split (bool): Whether to create two groups (decay / no-decay). If `False`,
            returns a single group regardless of `weight_decay`.

    Returns:
        List[Dict[str, Any]]: A list of optimizer param-group dicts suitable for torch.optim.

    Notes:
        • Parameters with `requires_grad=False` are skipped.
        • Heuristic for no-decay: parameter names containing "bias", "ln", or "layernorm"
          (case-insensitive), or tensors with `ndim < 2` (e.g., biases, LayerNorm weights).
        • Ensure `weight_decay` is a float when provided (matches torch.optim expectations).

    Example:
        >>> groups = split_decay_groups(model.named_parameters(), lr=3e-4, weight_decay=0.01, split=True)
        >>> optimizer = torch.optim.AdamW(groups)
        # No split (single group, all decayed):
        >>> groups = split_decay_groups(model.named_parameters(), lr=3e-4, weight_decay=0.01, split=False)
        >>> optimizer = torch.optim.AdamW(groups)
    """
    if split and weight_decay is not None:
        decay, no_decay = [], []
        for n, p in named_params:
            if not p.requires_grad:
                continue

            is_norm = any(k in n.lower() for k in ("layernorm", "rmsnorm", "groupnorm", "batchnorm", ".ln", ".norm"))
            if p.ndim < 2 or "bias" in n.lower() or is_norm:
                no_decay.append(p)
            else:
                decay.append(p)
                
        return [
            {"params": decay,    "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ]
    else:
        params = [p for _, p in named_params if p.requires_grad]
        group = {"params": params, "lr": lr}
        if weight_decay is not None:
            group["weight_decay"] = weight_decay
        return [group]
        

def _uniq_params(params):
    """Return params with duplicates (by object identity) removed while preserving order."""
    seen = set()
    out = []
    for p in params:
        pid = id(p)
        if pid not in seen:
            out.append(p)
            seen.add(pid)
    return out
    
class CosineAnnealingWarmRestartsDecay(CosineAnnealingWarmRestarts):
    def __init__(self, 
                 optimizer,
                 T_0, 
                 T_mult=1,
                 eta_min=0, 
                 last_epoch=-1, 
                 verbose=False, 
                 decay=1):
        
        super().__init__(optimizer,
                         T_0, 
                         T_mult=T_mult,
                         eta_min=eta_min, 
                         last_epoch=last_epoch, 
                         verbose=verbose)
        
        self.decay = decay
        self.initial_lrs = self.base_lrs
        self._eta_min = eta_min
        
    def step(self, epoch=None):
        if epoch == None:
            if self.T_cur + 1 == self.T_i:
                if self.verbose:
                    print("multiplying base_lrs by {:.4f}".format(self.decay))
                self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    n = int(epoch / self.T_0)
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
            else:
                n = 0

            new_base_lrs = [np.maximum(self._eta_min, initial_lrs * (self.decay**n)) for initial_lrs in self.initial_lrs]
            
            self.base_lrs = new_base_lrs # [initial_lrs * (self.decay**n) for initial_lrs in self.initial_lrs]

        super().step(epoch)
