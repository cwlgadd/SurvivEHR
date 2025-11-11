import math
import logging
from typing import Iterable, Tuple, List, Dict, Any, Optional, Sequence, Literal
import torch
from torch.optim import Optimizer
from torch.nn import Parameter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, CosineAnnealingLR, LambdaLR, SequentialLR, ChainedScheduler

from SurvivEHR.examples.modelling.SurvivEHR.optimizers_utils import split_decay_groups, _uniq_params, CosineAnnealingWarmRestartsDecay


class ConfigureFTOptimizers():


    @property
    def get_optim_cfg(self):
        """Return a Lightning-compatible dict with optimizer and scheduler config."""
        return {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler_config}
        
    def __init__(self, ft_experiment):
        super().__init__()
        self.experiment = ft_experiment
        
        # ---- LRs ----
        self.backbone_lr = float(ft_experiment.cfg.optim.learning_rate)
        self.head_lr     = float(ft_experiment.cfg.fine_tuning.head.learning_rate)
        
        # ---- Optimizer config ----
        # Weight decay
        wd_cfg = getattr(ft_experiment.cfg.fine_tuning, "weight_decay", None)
        self.weight_decay: Optional[float] = wd_cfg
        # Layer-wise learning rate decay
        llrd_cfg = getattr(ft_experiment.cfg.fine_tuning, "llrd", None)
        self.use_llrd: Optional[float] = llrd_cfg
        
        # ---- Scheduler config ----
        self.scheduler_type = getattr(ft_experiment.cfg.optim, "scheduler", "reduceonplateau").lower()
        self.warmup = bool(getattr(ft_experiment.cfg.optim, "scheduler_warmup", False))

        # ------------------------------
        # Collect head/backbone params
        # Split parameters by those in new fine-tuning head and those in backbone
        # ------------------------------
        self.head_named = []
        if ft_experiment.surv_weight > 0:
            self.head_named += list(ft_experiment.surv_layer.named_parameters())
        if ft_experiment.value_weight > 0:
            self.head_named += list(ft_experiment.value_layer.named_parameters())
        if hasattr(ft_experiment, "reduce_hidden"):
            self.head_named += list(ft_experiment.reduce_hidden.named_parameters())
        
        self.backbone_named = list(ft_experiment.model.transformer.named_parameters())

        # ------------------------------
        # Optimizer param groups
        #   Only parameters set to be trainable at this point will be added to the optimizer
        # ------------------------------
        self.optimizer = self.make_optimizer(llrd_decay_factor=self.use_llrd)
        # Log concise param-group summary
        for i, g in enumerate(self.optimizer.param_groups):
            n = sum(p.numel() for p in g["params"])
            logging.info(f"opt.group[{i}]: lr={g['lr']:.3g}, wd={g.get('weight_decay', 0.0)} (n={n})")

        # ------------------------------
        # Scheduler
        # ------------------------------
        self.lr_scheduler_config = self.make_scheduler(
            optimizer       = self.optimizer,
            scheduler_type  = self.scheduler_type,
            warmup          = self.warmup,
        )
        # Log concise scheduler summary
        logging.info(f"Using {self.scheduler_type} scheduler with config {self.lr_scheduler_config}")

        
    def make_optimizer(
        self,
        llrd_decay_factor: Optional[float] = None,
    ):
        """
        Construct an AdamW optimizer with optional Layer-wise Learning-Rate Decay (LLRD).
    
        This utility builds parameter groups using `split_decay_groups`, which separates
        parameters into decay / no-decay groups (e.g., excluding biases and LayerNorms
        from weight decay). Two modes are supported:
    
        1) LLRD enabled (`llrd_decay_factor` is a float in (0, 1)):
           - The model head uses `self.head_lr`.
           - Transformer blocks are traversed from top to bottom; each lower block uses
             the previous block's LR multiplied by `llrd_decay_factor`, starting at
             `self.backbone_lr`.
           - Non-block transformer params (e.g., token/pos embeddings, final LayerNorm)
             are grouped using the final (smallest) backbone LR.
    
        2) LLRD disabled (`llrd_decay_factor is None`):
           - Head params use `self.head_lr`.
           - All backbone params use `self.backbone_lr`.
    
        Assumptions about `self` (attributes expected):
            - self.weight_decay: float weight decay coefficient for decayed params.
            - self.head_lr: float learning rate for the head.
            - self.backbone_lr: float learning rate for the backbone.
            - self.head_named: Iterable[Tuple[str, Parameter]] for head parameters.
            - self.backbone_named: Iterable[Tuple[str, Parameter]] for backbone parameters (used when LLRD disabled).
            - self.experiment.model.transformer: module with:
                * optional `blocks` iterable, each exposing `.named_parameters()`
                * optional `wte`, `wpe`, `ln_f` submodules.
    
        Args:
            llrd_decay_factor: If provided, enables LLRD and must satisfy 0 < factor < 1.
                               If None, no LLRD is applied.
    
        Returns:
            torch.optim.Optimizer: An AdamW optimizer initialized with the constructed groups.
    
        Raises:
            AssertionError: If `llrd_decay_factor` is provided but not in (0, 1).
            NotImplementedError: If `llrd_decay_factor` is neither a float nor None.
    
        Notes:
            - The top-level optimizer `weight_decay` is set to 0.0 because decay is applied
              per-group via `split_decay_groups`.
            - Parameters with `requires_grad=False` are automatically skipped by
              `split_decay_groups`.
        """

        groups: List[Dict[str, Any]] = []

        if isinstance(llrd_decay_factor, float):
            
            if not (0.0 < llrd_decay_factor < 1.0):
                raise ValueError("llrd_decay_factor must be in (0, 1).")

            # Head
            groups += split_decay_groups(self.head_named, self.head_lr, self.weight_decay)

            # Transformer blocks with LLRD
            transformer = self.experiment.model.transformer
            blocks = list(getattr(transformer, "blocks", []))
            lr = self.backbone_lr
            for block in reversed(blocks):
                named = list(block.named_parameters())
                groups += split_decay_groups([(f"blk.{k}", v) for k, v in named], lr, self.weight_decay)
                lr *= llrd_decay_factor  # decay for next (lower) block
        
            #  Non-blocks (at bottom of network), using smallest decayed backbone LR
            extra_named: List[Tuple[str, Parameter]] = []
            for attr in ("wte", "wpe", "ln_f"):
                module = getattr(transformer, attr, None)
                if module is not None:
                    extra_named += list(module.named_parameters())
            if extra_named:
                groups += split_decay_groups([(f"extra.{k}", v) for k, v in extra_named], lr, weight_decay=None)

            logging.info(
                f"LLRD: backbone_base_lr={self.backbone_lr}, decay={llrd_decay_factor}, "
                f"head_lr={self.head_lr}, final_min_lr={lr}"
            )
            
        elif llrd_decay_factor is None:
            # No LLRD: two groups -- head and backbone -- each still split into decay/no-decay.        
            groups += split_decay_groups(self.head_named, self.head_lr, self.weight_decay)
            groups += split_decay_groups(self.backbone_named, self.backbone_lr, self.weight_decay)
            
            logging.info(f"No LLRD: backbone_lr={self.backbone_lr}, head_lr={self.head_lr}")

        else:
            raise NotImplementedError("llrd_decay_factor must be a float or None.")

        # Ensure params are not duplicated
        for g in groups:
            g["params"] = _uniq_params(g["params"])

        # -----------------------------------------------------------------------------
        # Sanity-check that every trainable parameter in the model has been assigned to
        # exactly one optimizer group.
        #
        # 1) Build `all_trainable`: the set of object IDs for all parameters returned by
        #    `model.named_parameters()` that require gradients.
        # 2) Build `grouped`: the set of object IDs for all parameters that appear in the
        #    optimizer `groups` you constructed (across all param groups).
        # 3) If these sets differ, it means either:
        #    - Some trainable parameters were not placed into any optimizer group
        #    - Some parameters may have been assigned multiple times (potential duplicates).
        # -----------------------------------------------------------------------------
        # all_trainable = {id(p) for _, p in self.experiment.named_parameters() if p.requires_grad}
        # grouped = {id(p) for g in groups for p in g["params"]}
        # if all_trainable != grouped:
        #     missing = all_trainable - grouped
        #     duplicate = [id_ for id_ in grouped if list(grouped).count(id_) > 1]  # paranoia
        #     raise RuntimeError(f"Param grouping mismatch: missing={len(missing)}, duplicates_possible={len(duplicate)}")
            
        # Decay is handled per-group; set global weight_decay to 0.0.
        return torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8)

    
    def make_scheduler(
        self,
        optimizer: Optimizer,
        scheduler_type: Literal["ca", "cawarmrestarts", "cawarmrestartsdecay", "reduceonplateau"],
        warmup: bool = False,
        warmup_steps: Optional[int] = None,
        monitor: str = "val_loss",
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Build a PyTorch-Lightning LR-scheduler config for the given optimizer.
    
        Supports:
          • "ca"                  → CosineAnnealingLR (stepped every train step)
          • "cawarmrestarts"      → CosineAnnealingWarmRestarts (stepped every train step)
          • "cawarmrestartsdecay" → CosineAnnealingWarmRestarts + multiplicative decay per restart (custom)
          • "reduceonplateau"     → ReduceLROnPlateau (stepped every epoch, monitors `monitor`)
    
        Warmup:
          If `warmup=True` and the scheduler is step-based, a linear warmup (LambdaLR) is
          prepended using `SequentialLR`. By default, warmup length = `steps_per_epoch`,
          unless `warmup_steps` is provided.
    
        Returns:
          A dict compatible with Lightning’s `configure_optimizers()` lr_scheduler config.
    
        Notes:
          - Uses `self.backbone_lr` for `eta_min` in cosine schedules.
          - Expects `self.experiment.trainer` to define `estimated_stepping_batches` and `max_epochs`.
          - For "reduceonplateau", Lightning expects `{"reduce_on_plateau": True}` in the config.
        """
        # ---- derive step/epoch counts safely ----
        trainer = self.experiment.trainer
        est_steps = int(getattr(trainer, "estimated_stepping_batches", 100))
        max_epochs = int(getattr(trainer, "max_epochs", 1) or 1)
        steps_per_epoch = max(1, math.ceil(est_steps / max_epochs))
        freq = 1  # call scheduler every interval

        # ---- choose base scheduler ----
        reduce_on_plateau_flag = False
        eta_min = self.backbone_lr / 5.0
        
        if scheduler_type == "ca":
            sch = CosineAnnealingLR(optimizer, T_max=est_steps, eta_min=eta_min)
            interval, mon = "step", None
        
        elif scheduler_type == "cawarmrestarts":
            sch = CosineAnnealingWarmRestarts(optimizer, T_0=steps_per_epoch, T_mult=2, eta_min=eta_min)
            interval, mon = "step", None
        
        elif scheduler_type == "cawarmrestartsdecay":
            sch = CosineAnnealingWarmRestartsDecay(
                optimizer, T_0=steps_per_epoch, T_mult=2, eta_min=eta_min, decay=0.9
            )
            interval, mon = "step", None
        
        elif scheduler_type == "reduceonplateau":
            sch = ReduceLROnPlateau(optimizer, factor=0.9, min_lr=1e-6, patience=3, cooldown=1)
            interval, mon = "epoch", monitor
            reduce_on_plateau_flag = True
        
        else:
            raise ValueError(f"Invalid scheduler_type: {scheduler_type}.")
        
        # ---- optional warmup (only for step-based schedulers) ----
        if warmup and interval == "step":
            
            wu_steps = warmup_steps or steps_per_epoch
            logging.info(f"Using linear warmup for {wu_steps} steps before {scheduler_type}.")

            warm = LambdaLR(optimizer, lr_lambda=lambda s: min(1.0, s / max(1, wu_steps)))
            sch = SequentialLR(optimizer, schedulers=[warm, sch], milestones=[wu_steps])

            scheduler_type = f"warmup+{scheduler_type}"
            logging.info(f"Using linear warmup for {warmup_steps} steps.")
            
        elif warmup and interval != "step":
            logging.info("Warmup requested but scheduler is epoch-based; skipping warmup.")

        # ---- assemble Lightning config ----
        lr_scheduler_config = {
            "scheduler": sch,                                                           # The scheduler instance
            "interval": interval,                                                       # The unit of the scheduler's step size
            "monitor": mon,                                                             # Metric to monitor for scheduler, if needed
            "frequency": freq,                                                          # How many epochs/steps should pass between calls to `scheduler.step()`
            "strict": strict,                                                           # Enforce that "val_loss" is available when the scheduler is updated
            "name": scheduler_type,                                                     # For `LearningRateMonitor`, specify a custom logged name
        }

        if reduce_on_plateau_flag:
            lr_scheduler_config["reduce_on_plateau"] = True

        return lr_scheduler_config
