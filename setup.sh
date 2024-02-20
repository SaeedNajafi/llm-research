#!/bin/bash

set -e

for ARGUMENT in "$@"
do
	KEY=$(echo $ARGUMENT | cut -f1 -d=)
	KEY_LENGTH=${#KEY}
	VALUE="${ARGUMENT:$KEY_LENGTH+1}"
	export "$KEY"="$VALUE"
done

function install_python () {
	if [ "$OS" = "mac" ]; then
		brew install python@3.10
		brew install python-tk@3.10
		python3_command="python3.10"
	elif [ "$OS" = "vcluster" ]; then
		module load python/3.10.12
		export PATH=$HOME/.local/bin:$PATH
		python3_command="python3"
	fi
}

ENV_NAME="llm"

function install_env () {
	${python3_command} -m venv $ENV_NAME-env
	source $ENV_NAME-env/bin/activate
	pip3 install --upgrade pip
}


function install_package () {
	pip3 install --no-cache-dir setuptools
	pip3 install --no-cache-dir wheel
	if [ "$OS" = "mac" ]; then
		pip3 install --pre torch torchvision torchaudio torchtext \
			--extra-index-url https://download.pytorch.org/whl/nightly/cpu

		pip3 install --no-cache-dir tensorflow tensorboard
		pip3 install --no-cache-dir tensorflow-macos
		pip3 install -e .'[dev]'

	elif [ "$OS" = "vcluster" ]; then
		pip3 install torch torchvision torchaudio torchtext \
			--no-cache-dir --index-url https://download.pytorch.org/whl/cu118
		pip3 install --no-cache-dir tensorflow tensorboard
		export CUDA_HOME=/pkgs/cuda-11.8
		pip3 install -e .'[dev]'
		pip3 install --no-cache-dir packaging
		pip3 uninstall -y ninja && python3 -m pip install --no-cache-dir ninja
		MAX_JOBS=7 pip3 install --no-cache-dir flash-attn --no-build-isolation
	fi

}

install_python
install_env
install_package
