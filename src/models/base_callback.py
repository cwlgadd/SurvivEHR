# Base class for callback classes
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import pandas as pd
import scipy.cluster.hierarchy as hcluster
import torch

import pickle
import sklearn.manifold
from pytorch_lightning import Callback
from sklearn.manifold import TSNE
import umap
import wandb

from typing import Optional

class BaseCallback(object):
    """
    A base class to hold samples in memory for compute intensive callbacks which cant be ran on the entire dataset.
    """

    def __init__(self, val_batch=None, test_batch=None):

        assert (val_batch is not None) or (test_batch is not None), "Must supply a validation or test set"

        # Unpack validation hook set
        if val_batch is not None:
            self.do_validation = True
            self.val_batch = val_batch
        else:
            self.do_validation = False
            
        # Unpack test hook set
        if test_batch is not None:
            self.do_test = True
            self.test_batch = test_batch
        else:
            self.do_test = False

    # def histogram(self, ax, z, labels, xlabel, kde_only=True):
    #     """
    #     Plot a latent histogram on axis 'ax'.
    #         Optionally include labels, and (if included) plot a stacked histogram
    #     """
    #     assert len(z.shape) == 1, z.shape
    #     _df = pd.DataFrame({
    #         "latent": z,
    #         "Cancer type": [self.label_dict[l] for l in labels.numpy()] if self.label_dict is not None else labels
    #     })
    #     if kde_only:
    #         g = sns.kdeplot(data=_df, ax=ax, x="latent", palette="Set2", hue="Cancer type", legend=True, common_norm=False)
    #     else:
    #         g = sns.histplot(data=_df, ax=ax, stat="density", multiple="dodge",    # stat=count,   multiple=stack,
    #                          x="latent", kde=True,
    #                          palette="Set2", hue="Cancer type",
    #                          element="bars", legend=True,
    #                                                                                                                                                                                                                                 # kde_kws={"common_norm": True, "common_grid": True}
    #                          )

    #     ax.set_xlabel(xlabel)
    #     ax.set_ylabel("Kernel Density Esimation")

    #     return ax




class Embedding(Callback, BaseCallback):
    """
    Callback to view latent embedding  of labelled data at each recurrent step,
     plotting the first two principal components of each latent embedding, and the free-energy of each component

    
    """
    def __init__(self, 
                 val_batch=None,
                 test_batch=None,
                 custom_stratification_method=None,):
        """

        KWARGS:
            val_batch:
                            A validation batch to be used for the embedding callback.
            test_batch:
                            A test batch to be used for the embedding callback.
            custom_stratification_method:
                             A function which takes as an argument the batch, and returns a stratification label
                             For example, if we want to stratify by gender then the batch dictionary will be inputted, and the return will be a list 
                             of length equal to the number of samples, of the form ["male", "female", "male",...] etc. and the unique strings will be used
                             to stratify the RMST logging.
        """
        Callback.__init__(self)
        BaseCallback.__init__(self, val_batch=val_batch, test_batch=test_batch)
        self.custom_stratification_method = custom_stratification_method

    def embedding(self, ax, z, labels, label_dict=None):
        """
        Plot a 2 or 3d latent embedding on axis `ax`.
            Optionally include labels, or a metric tied to each sample
        """

        # col_iterator = iter(get_cmap("tab10").colors)
        # col_iterator = iter(sns.color_palette("deep", n_colors=np.unique(labels), as_cmap=True))
        col_iterator = iter(sns.color_palette("Set2"))   # , n_colors=np.unique(labels)
        for lbl in np.unique(labels):
            mask = np.ma.getmask(np.ma.masked_equal(labels, lbl))
            color = next(col_iterator)
            c = label_dict[lbl] if label_dict is not None else lbl
            if z.shape[1] == 3:
                ax.scatter(z[mask, 0], z[mask, 1], z[mask,2], c=np.array([color]), label=c, alpha=0.25, edgecolors='none')
            else:
                ax.scatter(z[mask, 0], z[mask, 1], c=np.array([color]), label=c, alpha=0.25, edgecolors='none')

        ax.legend()
        ax.set_xlabel("Embed dim $1$")
        ax.set_ylabel("Embed dim $2$")
        if z.shape[1] == 3:
            ax.set_zlabel("Embed dim $3$")

        return ax
    
    def run_callback(self, 
                     _trainer,
                     _pl_module,
                     batch,
                     log_name:               str='Embedding',
                     proj:                   str="umap", 
                     proj_3d:                bool=False,
                     **kwargs):

        # Push features through the model to get the hidden dimension from the Transformer output:
        #      hidden_states: torch.Size([bsz, seq_len, hid_dim])
        _, _, hidden_states = _pl_module(batch)

        # Optionally process the batch using a custom method to label each patient into a different stratification group.
        if self.custom_stratification_method is not None and callable(self.custom_stratification_method):
            labels = self.custom_stratification_method(batch)
            assert len(labels) == batch['target_token'].cpu().numpy().shape[0]
        else:
            labels = ["no_stratification" for _ in range(batch['target_token'].cpu().numpy().shape[0])]

        # Plot each resolution-embedding vectors
        wandb_images = []

        hidden_states = np.asarray(hidden_states.detach().cpu()).reshape(-1,hidden_states.shape[-1])

        # Plot depends on shape of latent dimension
        if hidden_states.shape[1] == 1:
            # Histogram for 1-d embedding
            raise NotImplementedError
            # fig, ax = plt.subplots(1, 1)
            # fig.suptitle(f"j={level+1}")
            # self.histogram(ax, hidden_states[:, 0], labels=labels, xlabel=f"Latent resolution embedding $(h_j)$")
        elif (hidden_states.shape[1] == 2):
            # 2d scatter plot for 2-d embedding
            raise NotImplementedError
            # fig, ax = plt.subplots(1, 1)
            # fig.suptitle(f"j={level+1} resolution embedding")
            # self.embedding(ax, hidden_states, labels=labels)
        elif (hidden_states.shape[1] == 3) and (proj_3d is False):
            # 3d scatter plot for 3-d embedding
            raise NotImplementedError
            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # fig.suptitle(f"j-{level+1} resolution embedding")
            # self.embedding(ax, hidden_states, labels=labels)
        else:
            # Else we project with U-MAP to 2-d and do a 2d scatter
            fig, ax = plt.subplots(1, 1)
            fig.suptitle(f"Hidden state embedding ({proj})")                
            if proj.lower() == "umap":
                h_proj = umap.UMAP(n_components=2).fit_transform(hidden_states)  # random_state=42
            elif proj.lower() == "tsne":
                perp = np.max((3, np.min((30, int(0.1 * hidden_states.shape[0])))))
                fig.suptitle(f"j-{level + 1} resolution embedding (t-SNE, perplexity={perp})")
                h_proj = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=perp).fit_transform(hidden_states)
            else:
                raise NotImplementedError
                
            self.embedding(ax, h_proj, labels=labels)

        plt.tight_layout()
        # wandb_images.append(wandb.Image(fig))

        # Log
        _trainer.logger.experiment.log({
            log_name: wandb.Image(fig)
        })
        plt.close()

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.do_validation is True:
            self.run_callback(_trainer=trainer, 
                              _pl_module = pl_module,
                              batch=self.val_batch,
                              log_name = "Val:Embedding", 
                              )

    def on_test_epoch_end(self, trainer, pl_module):
        if self.do_test is True:
            self.run_callback(_trainer=trainer, 
                              _pl_module = pl_module,
                              batch=self.test_batch,
                              log_name = "Test:Embedding", 
                              )

# class SaveOutput(Callback, BaseCallback):
#     """
#     Callback on test epoch end to save outputs for plotting
#     """
#     def __init__(self, val_batch=None, test_batch=None, file_path=None):
#         Callback.__init__(self)
#         BaseCallback.__init__(self, val_batch=val_batch, test_batch=test_batch)
#         self.file_path = file_path if file_path is not None else "output.pkl"

#     def run_callback(self, features, labels, _pl_module, **kwargs):
#         # Push features through the model
#         outputs, _, hidden_states = _pl_module(batch)
#         # meta_result["labels"] = labels

#         with open(self.file_path, 'wb') as file:
#             pickle.dump(meta_result, file)

#     def on_test_epoch_end(self, trainer, pl_module):
#         if self.test_features is not None:
#             # Send to device
#             features = self.test_features.to(device=pl_module.device)
#             test_surv = {k: v.to(device=pl_module.device, non_blocking=True) for k, v in
#                          self.test_surv.items()}  # possibly empty surv dictionary
#             # Run callback
#             self.run_callback(features, self.test_labels, pl_module, **test_surv)