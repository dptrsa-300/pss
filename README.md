# pss
Using predicted protein structure data from DeepMind's AlphaFold to explore structural similarities among all 23K+ proteins in the human proteome.

## Conventions

* When referring to proteins in code or in datasets, use the full filename (ex.: AF-Q7Z5P9-F10-model_v1).

## Environment Setup

### Local

1. Install miniconda: [https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html)
2. ```conda env create -f environment.yml --force```  
3. ```conda activate pss```
4. [upon changes to `environment.yml`] `conda env update -f environment.yml` 

### Colab/JupyterLab

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

### Both: Install BLAST+

1. Download the [BLAST+ utilities](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/) from the NIH. Make sure you download the right executable based on your architecture (ex.: Linux, Colab or Google Cloud JupyterLab should use the [Linux X64 tarball](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.12.0+-x64-linux.tar.gz)).
2. Unpack the archive into blast/.
3. Run `chmod 777 -R blast/` to change permissions on the executables such that you can run them.

