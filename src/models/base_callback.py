# Base class for callback classes
from itertools import compress
from collections import Counter
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize, BoundaryNorm, ListedColormap
import pandas as pd
import scipy.cluster.hierarchy as hcluster
import torch
import logging
import pickle
import sklearn.manifold
from pytorch_lightning import Callback
from sklearn.manifold import TSNE
import umap
import wandb
import io
from PIL import Image

from typing import Optional

class BaseCallback(object):
    """
    A base class to hold samples in memory for compute intensive callbacks which cant be ran on the entire dataset.
    """

    def __init__(self, val_batch=None, test_batch=None):

        # assert (val_batch is not None) or (test_batch is not None), "Must supply a validation or test set"

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


class Embedding(Callback, BaseCallback):
    """
    Callback to view latent embedding  of labelled data at each recurrent step,
     plotting the first two principal components of each latent embedding, and the free-energy of each component

    
    """

    def create_canvas(self, set_axis_ticks=False, set_axis_labels=False):
        """ Create canvas in a way that ensure any plots created with this callback will all be formatted with the same alignment
        """
        
        # fig = plt.figure(figsize=(5, 4), dpi=300, constrained_layout=True)
        # gs = GridSpec(1, 2, figure=fig, width_ratios=[10, 1], wspace=0.3)
        # ax = fig.add_subplot(gs[0, 0])
        # cax = fig.add_subplot(gs[0, 1])

        fig = plt.figure(figsize=(4, 4), dpi=300)
        ax  = fig.add_axes([0.10, 0.12, 0.68, 0.68])  # square
        cax = fig.add_axes([0.82, 0.12, 0.04, 0.68])     

        if not set_axis_ticks:
            ax.set_xticks([])
            ax.set_yticks([])

        # Set axis labels
        if set_axis_labels:
            ax.set_xlabel("Embedding $1$")
            ax.set_ylabel("Embedding $2$")
            if z.shape[1] == 3:
                ax.set_zlabel("Embedding $3$")

        return fig, ax, cax

    def __init__(self, 
                 val_batch=None,
                 test_batch=None,
                 custom_stratification_method=None,
                 proj:str="umap", 
                 last_in_sequence:bool=True,
                 stratification_title:str="",
                 mask_static:bool=False,
                 mask_value:bool=False,
                 topk_labels:int=10,
                 ):
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
                             for labelling the hidden embedding.
                             There are two types of embedding plot, determined by the type of label returned by the ``custom_stratification_method``. If 
                             strings are returned, then we plot a discrete label map, if continuous values are preserved we plot a continuous or ordinal heatmap
            mask_static:
                            Whether the static covariates should be passed to the transformer when generating embedding 
            mask_value:
                            Whether the values should be passed to the transformer when generating embedding
            topk_labels:
                            How many of the most frequent labels should be plotted.
            
        """
        Callback.__init__(self)
        BaseCallback.__init__(self, val_batch=val_batch, test_batch=test_batch)
        self.custom_stratification_method = custom_stratification_method
        self.proj = proj.lower()
        self.last_in_sequence = last_in_sequence
        self.stratification_title = stratification_title
        self.mask_static = mask_static
        self.mask_value = mask_value

        if type(self.val_batch) != list:
            self.val_batch = [self.val_batch]
        if type(self.test_batch) != list:
            self.test_batch = [self.test_batch]


    def _split_background_labels(self, labels):
        """
        Processes a list of labels by separating the 'Other' (background) class from the rest.
    
        This function:
        - Assigns integer IDs to non-'Other' labels.
        - Creates a colormap where 'Other' is shown in grey, and other labels get distinct colors.
        - Defines point sizes and transparency levels, making 'Other' smaller and more transparent.
        - Prepares color indices for plotting.
    
        Args:
            labels (list of str): List of label names, with 'Other' used to mark background points.
    
        Returns:
            tuple:
                cmap (ListedColormap): Colormap with distinct colors for labels, grey for 'Other'.
                norm (BoundaryNorm): Normalization for mapping integer labels to colormap.
                sizes (np.ndarray): Array of point sizes (small for 'Other', large for others).
                alphas (np.ndarray): Array of alpha values (transparent for 'Other', opaque for others).
                colors (np.ndarray): Integer array of label indices shifted so 'Other' = 0.
        """
        
        # Separate "Other" points (background)
        mask_other = [label == "Other" for label in labels]
        mask_not_other = [label != "Other" for label in labels]
    
        # Unique labels excluding "Other"
        labels_not_other = list(compress(labels, mask_not_other))
        unique_labels = np.sort(np.unique(labels_not_other))
        label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Assign int labels: non-"Other" get ints; "Other" can be any placeholder (we’ll mask its color later)
        # int_labels = np.array([label_to_int[label] for label in labels[mask_not_other]])
        int_labels = np.full_like(labels, fill_value=-1, dtype=float)
        int_labels[mask_not_other] = [label_to_int[label] for label in labels_not_other]
    
        # Colormap excluding "Other" which will be grey
        full_palette = sns.color_palette("Set2", n_colors=len(unique_labels)+1)
        filtered_palette = [
            c for c in full_palette 
            if not (np.isclose(c[0], c[1], atol=1e-3) and np.isclose(c[1], c[2], atol=1e-3))
        ]
        filtered_palette = [(0.7, 0.7, 0.7)] + filtered_palette[:len(unique_labels)]
        cmap = ListedColormap(filtered_palette)
        norm = BoundaryNorm(boundaries=np.arange(-0.5, len(unique_labels) + 1.5), ncolors=cmap.N)
    
        # Assign sizes: small for "Other", larger for rest
        sizes = np.where(mask_other, 0.3, 1)
        alphas = np.where(mask_other, 0.2, 1)
        colors = int_labels + 1    # Shift int_labels up by 1 so "Other" = 0, rest = 1..N
    
        return cmap, norm, sizes, alphas, colors

    def _plot_from_continuous_labels(self, z, labels):

        fig, ax, cax = self.create_canvas(set_axis_labels=False, set_axis_ticks=False)
        
        cmap = sns.color_palette("viridis", as_cmap=True)
        if z.shape[1] == 3:
            scatter = ax.scatter(z[:, 0], z[:, 1], z[:, 2], s=0.3, c=labels, alpha=1, cmap=cmap)
        else:
            scatter = ax.scatter(z[:, 0], z[:, 1], s=0.3, c=labels, alpha=1, cmap=cmap)   # 'viridis'
            
        cbar = plt.colorbar(scatter, cax=cax)
        cbar.set_label(self.stratification_title)
        return fig, ax

    def _plot_from_integer_labels(self, z, labels):

        fig, ax, cax = self.create_canvas(set_axis_labels=False, set_axis_ticks=False)
        
        unique_labels = np.sort(np.unique(labels))
        cmap = ListedColormap(sns.color_palette("Set2", n_colors=len(unique_labels)))
        norm = BoundaryNorm(boundaries=np.arange(unique_labels.min() - 0.5, unique_labels.max() + 1.5), ncolors=cmap.N)
        if z.shape[1] == 3:
            scatter = ax.scatter(z[:, 0], z[:, 1], z[:, 2], s=0.3, c=labels, alpha=1, cmap=cmap, norm=norm)
        else:
            scatter = ax.scatter(z[:, 0], z[:, 1], s=0.3, c=labels, alpha=1, cmap=cmap, norm=norm)   # 'viridis'
            
        cbar = plt.colorbar(scatter, cax=cax, ticks=unique_labels)
        cbar.set_label(self.stratification_title)
        return fig, ax

    def _plot_from_str_labels(self, z, labels, top_k=7):

        fig, ax, cax = self.create_canvas(set_axis_labels=False, set_axis_ticks=False)
        
        # Reduce to only the most common labels
        if top_k:
            freq = Counter(labels)
            top_k = {label_key for label_key, _ in freq.most_common(top_k)}
            labels =  [label if label in top_k else "Other" for label in labels]
        
        unique_labels = np.sort(np.unique(labels))
        cmap = ListedColormap(sns.color_palette("Set2", n_colors=len(unique_labels)))
        norm = BoundaryNorm(boundaries=np.arange(-0.5, len(unique_labels) + 0.5), ncolors=cmap.N)

        # Assign int labels: non-"Other" get ints, "Other" gets -1
        label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
        int_labels = [label_to_int[label] for label in labels]

        # Plot all points
        if z.shape[1] == 3:
            scatter = ax.scatter(z[:, 0], z[:, 1], z[:, 2], s=0.3, c=int_labels, alpha=1, cmap=cmap, norm=norm)
        else:
            scatter = ax.scatter(z[:, 0], z[:, 1], s=0.3, c=int_labels, alpha=1, cmap=cmap, norm=norm)

        cbar = plt.colorbar(scatter, cax=cax, ticks=np.arange(len(unique_labels)))
        cbar.ax.set_yticklabels(unique_labels)               
        cbar.set_label(self.stratification_title)
        
        return fig, ax

    def plot_embedding_with_background(self, _trainer, log_name, z, labels, set_axis_label=False):
        """
        Fuzzy plotting - 1 vs Remaining labels, with Remaining labels shown as background
        """

        all_labels = sorted({elem for s in labels for elem in s})
        labels_not_other = np.sort(np.unique([label for label in all_labels if label != "Other" ]))
        
        for element in labels_not_other:
            
            # fig, ax = plt.subplots(1, 1, figsize=self._FIGSIZE)
            fig, ax, cax = self.create_canvas(set_axis_labels=False, set_axis_ticks=False)
            
            sub_labels = [element if element in label_set else "Other" for label_set in labels]
            cmap, norm, sizes, alphas, colors = self._split_background_labels(sub_labels)
            
            # Plot all points
            if z.shape[1] == 3:
                scatter = ax.scatter(z[:, 0], z[:, 1], z[:, 2],
                                     s=sizes, c=colors, alpha=alphas, cmap=cmap, norm=norm)
            else:
                scatter = ax.scatter(z[:, 0], z[:, 1],
                                     s=sizes, c=colors, alpha=alphas, cmap=cmap, norm=norm)

            cbar = plt.colorbar(scatter, cax=cax, ticks=np.arange(2))
            cbar.ax.set_yticklabels(np.array(["Other", element]), rotation=90)
            cbar.set_label(self.stratification_title)

            # render the figure to an in‑memory PNG with tight bounding box, then log it
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            buf.seek(0)   
            pil_img = Image.open(buf)
            wb_img = wandb.Image(pil_img, caption=log_name+f"_{element}")
            _trainer.logger.experiment.log({log_name+f"_{element}": wb_img})

    def plot_embedding_without_background(self, _trainer, log_name, z, labels, set_axis_label=False):
        """
        Normal plotting
        """
        
        # If given boolean labels, convert to string
        if np.issubdtype(labels.dtype, np.bool_):
            labels = labels.astype(str)
            
        is_integer = np.issubdtype(labels.dtype, np.integer)
        is_continuous = np.issubdtype(labels.dtype, np.floating)
        is_string = np.issubdtype(labels.dtype, np.str_)

        if is_continuous:
            fig, ax = self._plot_from_continuous_labels(z, labels)
            
        elif is_integer:
            fig, ax = self._plot_from_integer_labels(z, labels)

        elif is_string:
            fig, ax = self._plot_from_str_labels(z, labels)
            
        else:
            print(labels)
            raise ValueError(f"Labels must be either integers (for ordinal data), floats (for continuous data) or strings (for categorical data). Got {labels.dtype}")

        # render the figure to an in‑memory PNG with tight bounding box, then log it
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)   
        pil_img = Image.open(buf)
        wb_img = wandb.Image(pil_img, caption=log_name)
        _trainer.logger.experiment.log({log_name: wb_img})
        
    def run_callback(self, 
                     _trainer,
                     _pl_module,
                     batches:                list,
                     log_name:               str='Embedding',
                     max_num_labels:         int=10,
                     **kwargs
                    ):

        all_hidden = []
        all_labels = []
        for _idx, batch in enumerate(batches):

            # Optionally process the batch using a custom method to label each patient into a different stratification group.
            if self.custom_stratification_method is not None and callable(self.custom_stratification_method):
                labels, batch = self.custom_stratification_method(batch)     
            else:
                labels = ["no stratification" for _ in range(batch['tokens'].shape[0])]
            labels = np.asarray(labels)
            assert len(labels) == batch['tokens'].shape[0], f"Expected { batch['tokens'].shape[0]} sample labels, got {len(labels)}"
            
            # Push features through the model to get the hidden dimension from the Transformer output:
            #      hidden_states: torch.Size([bsz, seq_len, hid_dim])
            static = None if self.mask_static else batch["static_covariates"].to(_pl_module.device) 
            values = None if self.mask_value else batch["values"].to(_pl_module.device) 
            hidden_states = _pl_module.model.transformer(tokens=batch["tokens"].to(_pl_module.device),
                                                         ages=batch["ages"].to(_pl_module.device),
                                                         values=values,
                                                         covariates=static,
                                                         )        
            hidden_states = np.asarray(hidden_states.detach().cpu())
            if self.last_in_sequence:
                hidden_states = hidden_states[:, [-1], :]
                
            # Cast labels to hidden_states dimension
            #    One sample may have seq_len hidden states, so we must broadcast this.
            #    Padded values will have the same hidden_state as the final observation - so will just be overlaid on top of each other
            labels = np.repeat(np.array(labels)[:, np.newaxis], hidden_states.shape[1], axis=1)
    
            # Flatten both along seq_len
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
            labels = labels.reshape(-1)  # Flatten to match reshaped hidden_states

            all_hidden.append(hidden_states)
            all_labels.append(labels)

        all_hidden = np.concatenate(all_hidden, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Project hidden
        if self.proj == "umap":
            all_hidden_proj = umap.UMAP(n_components=2, 
                                        random_state=42
                                       ).fit_transform(all_hidden)
        elif self.proj == "tsne":
            # perplexity = np.max((3, np.min((30, int(0.1 * all_hidden.shape[0])))))
            all_hidden_proj = TSNE(n_components=2, 
                                   learning_rate='auto',
                                   init="pca", 
                                   random_state=42, 
                                   perplexity=min(100, max(3, all_hidden.shape[0] // 10))
                                  ).fit_transform(all_hidden)
        else:
            raise NotImplementedError

        # Plot
        if np.issubdtype(labels.dtype, np.object_):
            self.plot_embedding_with_background(_trainer, log_name, all_hidden_proj, all_labels)
        else:
            self.plot_embedding_without_background(_trainer, log_name, all_hidden_proj, all_labels)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.do_validation is True:
            with torch.no_grad():
                self.run_callback(_trainer=trainer, 
                                  _pl_module = pl_module,
                                  batches=self.val_batch,
                                  log_name = f"Val:{self.proj}_{self.stratification_title}", 
                                  )

    def on_test_epoch_start(self, trainer, pl_module):
        if self.do_test is True:
            with torch.no_grad():
                self.run_callback(_trainer=trainer, 
                                  _pl_module = pl_module,
                                  batches=self.test_batch,
                                  log_name = f"Test:{self.proj}_{self.stratification_title}", 
                                  )