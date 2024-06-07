# Create custom callbacks for our pytorch-lightning model

import numpy as np
from pytorch_lightning import Callback
import torch
from sklearn.manifold import TSNE
import umap
import wandb
import matplotlib.pyplot as plt
from WaveLSTM.custom_callbacks.base import BaseCallback
from pycox.evaluation import EvalSurv
import pandas as pd
import seaborn as sns
# from plotly import offline
import copy

class PerformanceMetrics(Callback, BaseCallback):
    """
    Record metrics for survival model.
    """

    def get_mae_rmse(self):
        # Get Mean Absolute Error and Root Mean Square Error
        raise NotImplementedError

    def __init__(self, val_samples=None, test_samples=None, label_dictionary=None):
        Callback.__init__(self)
        BaseCallback.__init__(self, val_samples=val_samples, test_samples=test_samples, label_dict=label_dictionary)

    def run_callback(self, features, labels, log_name, _trainer, _pl_module, **val_surv):
        labels = labels.cpu().detach().numpy()
        t_eval = np.linspace(start=0, stop=_pl_module.time_scale, num=300)

        # Push features through the model
        prediction, _ = _pl_module.predict(features, val_surv["c"], t=t_eval)
        surv = pd.DataFrame(np.transpose((1 - prediction)), index=t_eval)

        t = val_surv["t"].cpu().detach().numpy() # / _pl_module.time_scale
        k = val_surv["k"].cpu().detach().numpy()
        ev = EvalSurv(surv, t, k, censor_surv='km')

        time_grid = np.linspace(start=t.min(), stop=0.9 * _pl_module.time_scale, num=300)
        ctd = ev.concordance_td()                           # Time-dependent Concordance Index
        ibs = ev.integrated_brier_score(time_grid)          # Integrated Brier Score
        inbll = ev.integrated_nbll(time_grid)               # Integrated Negative Binomial LogLikelihood
        # mae, rmse = self.get_mae_rmse()
        
        # Log all
        self.log_dict({log_name + "ctd": ctd,
                       log_name + "ibs": ibs,
                       log_name + "inbll": inbll,
                       # log_name + "mae": mae,
                       # log_name + "rmse": rmse
                       })

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.val_features is not None:
            # Send to device
            features = self.val_features.to(device=pl_module.device)
            val_surv = {k: v.to(device=pl_module.device, non_blocking=True) for k, v in
                         self.val_surv.items()}
            # Run callback
            self.run_callback(features, self.val_labels, "Val:", trainer, pl_module, **val_surv)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_features is not None:
            # Send to device
            features = self.test_features.to(device=pl_module.device)
            test_surv = {k: v.to(device=pl_module.device, non_blocking=True) for k, v in
                         self.test_surv.items()}  # possibly empty surv dictionary
            # Run callback
            self.run_callback(features, self.test_labels, "Test:", trainer, pl_module, **test_surv)


class KaplanMeier(Callback, BaseCallback):

    def __init__(self,
                 val_samples=None, test_samples=None,
                 label_dictionary=None, group_by=["label"],
                 error_bars=True, samples=True):
        """
        @param: group_by
            Cancer type: "label"
            Multi-resolution quantile: "quant"
        @param: error bars (bool)
            Whether we make the KM-plot with error bars
        @param: samples (bool)
            Whether we make the KM-plot with individual samples

        """
        Callback.__init__(self)
        BaseCallback.__init__(self, val_samples=val_samples, test_samples=test_samples, label_dict=label_dictionary)
        self.group_by = group_by
        self.error_bars = error_bars
        self.samples = samples

    def run_callback(self, features, labels, log_name, _trainer, _pl_module, **val_surv):
        labels = labels.cpu().detach().numpy()
        if self.label_dict is not None:
            labels = [self.label_dict[l] for l in labels]
        t_eval = np.linspace(0, _pl_module.time_scale, 100)

        # Push features through the model
        prediction, meta_data = _pl_module.predict(features, val_surv["c"], t=t_eval)
        prediction = 1 - prediction

        # Put into df (#TODO: vectorise)
        surv = []
        for idx_n in range(prediction.shape[0]):
            for idx_t in range(prediction.shape[1]):
                d = {'survival_prob' : prediction[idx_n, idx_t],
                     'time' : t_eval[idx_t],
                     'sample_id' : f"s{idx_n}",
                     'cancer type': labels[idx_n],
                     'event': int(val_surv["k"][idx_n].detach().cpu().numpy())
                     }
                surv.append(d)
        surv = pd.DataFrame(surv)

        # Plot KM-curves
        wandb_images = []
        if "label" in self.group_by:
            if self.error_bars:
                fig, ax = plt.subplots(1, 1)
                fig.suptitle(f"Mean with 95% CI of estimator")
                sns.lineplot(data=surv, x="time", y="survival_prob", hue="cancer type", ax=ax,
                             estimator="mean")   # style="event",
                plt.xlim((0, _pl_module.time_scale))
                plt.ylim((0, 1))
                plt.xlabel("Time")
                plt.ylabel("Probability of survival")
                wandb_images.append(wandb.Image(fig))
            if self.samples:
                for cancer in surv["cancer type"].unique():
                    group_surv = surv[surv["cancer type"] == cancer]
                    fig, ax = plt.subplots(1, 1)
                    # fig.suptitle(f"label: {cancer}")
                    sns.lineplot(data=group_surv, x="time", y="survival_prob", hue="cancer type", units="sample_id",
                                 estimator=None, lw=1, alpha=0.2, ax=ax)
                    plt.xlim((0, _pl_module.time_scale))
                    plt.ylim((0, 1))
                    plt.xlabel("Time")
                    plt.ylabel("Probability of survival")
                    wandb_images.append(wandb.Image(fig))

        _trainer.logger.experiment.log({
            log_name + "_label": wandb_images
        })

        if  "quant" in self.group_by:
                pass



            # for lbl in np.unique(labels):
            #     idx_lbl = np.where(labels == lbl)[0]
            #     for k, idx_lbl_k in enumerate([np.where(_k[idx_lbl] == 0)[0], np.where(_k[idx_lbl] != 0)[0]]):
            #         idx_lbl_k = idx_lbl_k[:40] if len(idx_lbl_k) > 40 else idx_lbl_k
            #         # idx_lbl = idx_lbl[:5] if len(idx_lbl) > 5 else idx_lbl
            #         # for i in range(len(idx_lbl)):
            #         if len(idx_lbl_k) > 0:
            #             ev[idx_lbl_k].plot_surv()
            #             plt.title(
            #                 f"Cancer {cancer_types[lbl]} and {'event' if k == 1 else 'censored'}")  # at normalised time {_t[i]:.2f},
            #             plt.ylim((0, 1))
        # elif self.group_by == "quant":
        #     pass
        # else:
        #     raise NotImplementedError


        # _trainer.logger.experiment.log({
        #     log_name: wandb_images
        # })

        plt.close()

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.val_features is not None:
            # Send to device
            features = self.val_features.to(device=pl_module.device)
            val_surv = {k: v.to(device=pl_module.device, non_blocking=True) for k, v in
                         self.val_surv.items()}
            # Run callback
            self.run_callback(features, self.val_labels, f"Val:KM", trainer, pl_module,
                              **val_surv)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_features is not None:
            # Send to device
            features = self.test_features.to(device=pl_module.device)
            test_surv = {k: v.to(device=pl_module.device, non_blocking=True) for k, v in
                         self.test_surv.items()}  # possibly empty surv dictionary
            # Run callback
            self.run_callback(features, self.test_labels, f"Test:KM", trainer, pl_module,
                              **test_surv)