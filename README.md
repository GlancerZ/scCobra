# scCobra: Contrastive cell embedding learning with domain adaptation for single-cell data integration 
  
This repositorty contains the implementation of scCobra. scCobra is a method combined contrastive learning and domain adaptation for single-cell data integration. 

![Workflow](https://raw.githubusercontent.com/GlancerZ/scCobra/main/Figure/single-cell-model.png)


# Installation

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
# You can install addtional packages: https://scib.readthedocs.io/en/latest/index.html

# Install ipykernel, if you want use jupyer notebook
conda install ipykernel

# (option) 
python -m ipykernel install --user --name scCobra --display-name "scCobra"

``` 

**Step 2**: Clone This Repo


# Resources

* [Data](https://figshare.com/articles/dataset/Benchmarking_atlas-level_data_integration_in_single-cell_genomics_-_integration_task_datasets_Immune_and_pancreas_/12420968) used in the study

## Compared methods

- [Seurat v3](https://github.com/satijalab/seurat) (default method: CCA) 
- [Scanorama](https://github.com/brianhie/scanorama)
- [scVI](https://github.com/YosefLab/scVI)
- [Harmony](https://github.com/slowkow/harmonypy) 

