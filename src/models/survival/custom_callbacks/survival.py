# Create custom callbacks for our pytorch-lightning model

import numpy as np
from pytorch_lightning import Callback
import torch
from sklearn.manifold import TSNE
import umap
import wandb
import matplotlib.pyplot as plt
from CPRD.src.models.base_callback import BaseCallback
from pycox.evaluation import EvalSurv
import pandas as pd
import seaborn as sns
import logging
import copy

class PerformanceMetrics(Callback, BaseCallback):
    """
    Record metrics for survival model.
    """

    def get_mae_rmse(self):
        # Get Mean Absolute Error and Root Mean Square Error
        raise NotImplementedError

    def __init__(self, val_batch=None, test_batch=None):
        Callback.__init__(self)
        BaseCallback.__init__(self, val_batch=val_batch, test_batch=test_batch)

    def plot_km(self,
                _trainer,
                _pl_module,
                predictions,
                ks,
                t_eval,
                log_name,
               ):
        # Put into df (#TODO: vectorise)
        surv = []
        for ode_idx, (prediction, k) in enumerate(zip(predictions, ks)):
            for idx_n in range(10):#prediction.shape[0]):
                for idx_t in range(prediction.shape[1]):
                    d = {'survival_prob' : prediction[idx_n, idx_t],
                         'time' : t_eval[idx_t],
                         'sample_id' : f"s{idx_n}",
                         'ODE': ode_idx,
                         'event': int(k[idx_n])
                         }
                    surv.append(d)
        surv = pd.DataFrame(surv)

        # Plot KM-curves
        wandb_images = []
        # ... plot mean estimator of each ODE group
        if True:
            logging.debug("Plotting ODE KM means")
            fig, ax = plt.subplots(1, 1)
            fig.suptitle(f"Mean with 95% CI of estimator")
            sns.lineplot(data=surv, x="time", y="survival_prob", hue="ODE", ax=ax,
                         estimator="mean")   # style="event",
            plt.xlim((0, _pl_module.model.surv_layer._time_scale))
            # plt.ylim((0, 1))
            plt.xlabel(f"Time ({int(_pl_module.model.surv_layer._time_scale/365)} years)")
            plt.ylabel(f"Risk of token idx {ode_idx}")
            wandb_images.append(wandb.Image(fig))

        # ... plot individual samples of each ODE curve
        if False:
            for ode_idx in surv["ODE"].unique():
                logging.debug(f"Plotting ODE KM for ODE {ode_idx+1}")
                group_surv = surv[surv["ODE"] == ode_idx]
                fig, ax = plt.subplots(1, 1)
                # fig.suptitle(f"label: {cancer}")
                sns.lineplot(data=group_surv, x="time", y="survival_prob", hue="ODE", units="sample_id",
                             estimator=None, lw=1, alpha=0.2, ax=ax)
                plt.xlim((0, _pl_module.model.surv_layer._time_scale))
                # plt.ylim((0, 1))
                plt.xlabel(f"Time ({int(_pl_module.model.surv_layer._time_scale/365)} years)")
                plt.ylabel(f"Risk of token idx {ode_idx}")
                wandb_images.append(wandb.Image(fig))

        _trainer.logger.experiment.log({
            log_name + "_km": wandb_images
        })
        
    def run_callback(self,
                     _trainer,
                     _pl_module,
                     batch,
                     log_name:               str='Embedding',
                    ):
        
        # Push features through the model. The forward method of the model has two modes decided by is_generation. 
        # ... When true we forward only the last step (which saves compute for appliation of the model), whilst 
        # ... when false we forward all, but only compute the loss.
        # ... We need a third option here, as we want to generate, but we want to do this on all steps - i.e., we
        # ... want to evaluate the metrics across all the sequence (just as we do for the loss)

        # ... is_generation = False ensures we forward all hidden states (not just the last)
        # ... whilst return_cdf = True ensures we do calculate the CDF surves, as these are not required to train the model
        all_outputs, _, _ = _pl_module(batch, is_generation=False, return_cdf=True)
        outputs = all_outputs["surv"]
        # print(outputs)
        
        # the real time deltas 
        t = outputs["tte_deltas"].cpu().detach().numpy() 
        # t += np.abs(np.random.normal(size=t.shape))/10
        # the real outcomes, split as needed across each survival ODE model
        k = [_k.cpu().detach().numpy() for _k in outputs["k"]]
        # the predicted KM curve for each survival ODE model
        predictions = outputs["surv_CDF"] # [_surv_cdf.cpu().detach().numpy() for _surv_cdf in outputs["surv_CDF"]]
        surv = [pd.DataFrame(np.transpose((1 - _pred)), index=_pl_module.model.surv_layer.t_eval) for _pred in predictions] 

        if True:
            self.plot_km(_trainer, _pl_module,
                         predictions,
                         k,
                         _pl_module.model.surv_layer.t_eval,
                         log_name + "_first_ode_model")
            
        ctd, ibs, inbll = 0, 0, 0
        for _idx, (_surv, _k) in enumerate(zip(surv, k)):
            _k[0] = 1
            ev = EvalSurv(_surv, t, _k, censor_surv='km')
    
            # The tte_deltas are scaled inside the survival head, we calculate the metrics over a normalised time grid
            time_grid = np.linspace(start=t.min(), stop=0.9 * _pl_module.model.surv_layer._time_scale, num=300)         
            try:
                ctd += ev.concordance_td()                           # Time-dependent Concordance Index
                # ibs += ev.integrated_brier_score(time_grid)          # Integrated Brier Score
                # inbll += ev.integrated_nbll(time_grid)               # Integrated Negative Binomial LogLikelihood
                # mae, rmse = self.get_mae_rmse()
                print("Sucess")
                print(_surv.shape)
                print(_surv)
                print(len(t))
                print([_ for _ in t])
                print(len(k))
                print([_ for _ in _k])
                print(batch.keys())
            except:
                print("Fail")
                print(_surv.shape)              # (121, 3217)
                print(_surv)
                print(len(t))                   # 3217
                print([_ for _ in t])
                print(len(_k))                  # 3217
                print([_ for _ in _k])
                print(batch.keys())
                print(batch["ages"].shape)
                print(batch["ages"])
                print(batch["attention_mask"])
                print(torch.sum(batch["attention_mask"]))
                raise NotImplementedError

        print(f"{ctd}, {ibs}, {inbll}")
            

        # Log all
        self.log_dict({log_name + "ctd": ctd / (_idx + 1),
                       log_name + "ibs": ibs / (_idx + 1),
                       log_name + "inbll": inbll / (_idx + 1),
                       # log_name + "mae": mae,
                       # log_name + "rmse": rmse
                       })

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.do_validation is True:
            # Run callback
            self.run_callback(_trainer=trainer, 
                              _pl_module = pl_module,
                              batch=self.val_batch,
                              log_name = "Val:PerformanceMetrics", 
                              )

    # def on_test_epoch_end(self, trainer, pl_module):
    #     if self.test_features is not None:
    #         # Send to device
    #         features = self.test_features.to(device=pl_module.device)
    #         test_surv = {k: v.to(device=pl_module.device, non_blocking=True) for k, v in
    #                      self.test_surv.items()}  # possibly empty surv dictionary
    #         # Run callback
    #         self.run_callback(features, self.test_labels, "Test:", trainer, pl_module, **test_surv)
