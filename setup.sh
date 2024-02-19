#!/bin/bash

ENV_NAME="llm"


function load_python () {
	module load python/3.10.12
}


function install_env () {
	python3 -m venv $ENV_NAME-env
	source $ENV_NAME-env/bin/activate
	mkdir -p $HOME/tmp
	export TMPDIR=$HOME/tmp
	python3 -m pip install --upgrade --no-cache-dir pip
	export PATH=$HOME/.local/bin:$PATH
}


function install_package () {
	python3 -m pip install --no-cache-dir setuptools
	python3 -m pip install --no-cache-dir wheel
	python3 -m pip install torch --no-cache-dir --index-url https://download.pytorch.org/whl/cu118
	python3 -m pip install -e .'[dev]'
	python3 -m pip install --no-cache-dir packaging
	python3 -m pip uninstall -y ninja && python3 -m pip install --no-cache-dir ninja
	export CUDA_HOME=/pkgs/cuda-11.8
	MAX_JOBS=7 python3 -m pip install --no-cache-dir flash-attn --no-build-isolation

}

load_python
install_env
install_package
