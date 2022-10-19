# scCobra: Contrastive cell embedding learning with domain adaptation for single-cell data integration 
  
scCobra is used for integrating single-cell data from different batches. scCobra is a two-step model. In phase 1,  we let the original data enter the random dropout layer twice independently, so each cell will get two augmented views. Then a weight-sharing encoder is employed to encode the cell inputs in two augmented views into reduced latent embeddings. Here we employ a set of domain-specific batch normalization layers to normalize the embeddings from various sources (experimental batches). Following that, the projection head maps the latent cell embeddings nonlinearly to the same joint reduced space (h) for contrastive learning. In phase 2, we used the generative adversarial networks (GAN) to fuse different experiment batches of cells. Specifically, we use an adversarial training strategy to alternately train the discriminator and encoder, aligning the gene expression distribution of datasets from different experimental batches. After training, we use the trained encoder to encode cells directly, and the resulting cell embeddings (z) will be used for downstream tasks, such as cell clustering or trajectory inference of the integrated dataset. See the supplementary method for the specific parameters of scCobra.

![Workflow](https://raw.githubusercontent.com/GlancerZ/scCobra/main/Figure/single-cell-model.png)


## Installation

**Step 1**: Create a conda environment for scCobra

```bash
# Recommend you to use python above 3.8
conda create -n scCobra python=3.8
conda activate scCobra

# Install scanpy
pip install scanpy==1.9.1

# Install pytorch, choose profer version pytorch
# pytorch installation reference: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# Install R4.1, rpy2
conda install r-base=4.1.3
conda install -c conda-forge rpy2

# Install scib
pip install scib
conda install -c bioconda bioconductor-scran # Use for normalization
# You can install addtional packages: https://scib.readthedocs.io/en/latest/index.html

# Install ipykernel, if you want use jupyer notebook
conda install ipykernel

# (option) 
python -m ipykernel install --user --name scCobra --display-name "scCobra"

``` 

**Step 2**: Clone This Repo


## Data resources

* [Data](https://figshare.com/articles/dataset/Benchmarking_atlas-level_data_integration_in_single-cell_genomics_-_integration_task_datasets_Immune_and_pancreas_/12420968) used in the study

## How to use scCobra
* [Tutorial](https://github.com/GlancerZ/scCobra/blob/main/pancreas_demo.ipynb)

