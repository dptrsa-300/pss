# pss
Using predicted protein structure data from DeepMind's AlphaFold to explore structural similarities among all 23K+ proteins in the human proteome.

## Environment Setup
### Conda
#### Local
1. Install miniconda: [https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html)
2. ```conda env create -f environment.yml --force```  
3. ```conda activate pss```
4. [upon changes to `environment.yml`] `conda env update -f environment.yml` 


#### Colab/JupyterLab
Pull in `environment.yml` into directory

In top cell add 
```
%%bash
MINICONDA_INSTALLER_SCRIPT=Miniconda3-4.5.4-Linux-x86_64.sh
MINICONDA_PREFIX=/usr/local
wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT
chmod +x $MINICONDA_INSTALLER_SCRIPT
./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX
```
```
conda install --channel conda-forge -f environment.yml
```


