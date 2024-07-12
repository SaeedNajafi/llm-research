# LLM Research

This repo is going to contain code related to my Ph.D. research with large language models.

# Installation

The current installation guide is optimized for the slurm cluster such as narval cluster from computecanada or the vcluster from the vector institute. We prefer conda to install the python and related cuda libraries.

## Install miniconda

Visit the link https://docs.anaconda.com/miniconda/ and install the miniconda installer.

```sh
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```

## Install Module
Install the cuda drivers and the module itself. As an argument give the llm-env as the name for the environment.

```sh
bash setup.sh llm-env
```
