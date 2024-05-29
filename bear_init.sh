#!/bin/bash
# Original venv
# module load Transformers/4.24.0-foss-2022a-CUDA-11.7.0
# module load polars/0.17.12-foss-2022a
# module load PyTorch-Lightning/1.9.3-foss-2022a-CUDA-11.7.0
# module load Hydra/1.3.2-GCCcore-11.3.0
# module load sklearn-pandas/2.2.0-foss-2022a
# list of modules built into venv
# ???

### PyTorch 2.0 environment
module load PyTorch/2.0.1-foss-2022a-CUDA-11.7.0
module load PyTorch-Lightning/2.1.0-foss-2022a-CUDA-11.7.0
module load sklearn-pandas/2.2.0-foss-2022a
module load Hydra/1.3.2-GCCcore-11.3.0
module load polars/0.17.12-foss-2022a
module load wandb/0.13.6-GCC-11.3.0
module load Seaborn/0.12.1-foss-2022a
# list of modules built into venv
# pip install tdigest
# pip install transformers


# Misc  (?)
# module load wandb/0.13.6-GCC-11.3.0
# module load Seaborn/0.12.1-foss-2022a
# module load umap-learn/0.5.3-foss-2022a
# module load plotly.py/5.12.0-GCCcore-11.3.0