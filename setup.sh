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
		module load cuda11.8+cudnn8.9.6
		python3_command="python3"
	elif [ "$OS" = "colab" ]; then
		python3_command="python3"
	fi
}

ENV_NAME="llm"

function install_env () {
	${python3_command} -m venv $ENV_NAME-env
	source $ENV_NAME-env/bin/activate
	export PATH=${PWD}/$ENV_NAME-env/bin:$PATH
	pip3 install --upgrade pip
}


function install_package () {
	pip3 install --no-cache-dir setuptools
	pip3 install --no-cache-dir wheel
	if [ "$OS" = "mac" ]; then
		pip3 install --pre torch torchvision torchaudio torchtext \
			--extra-index-url https://download.pytorch.org/whl/nightly/cpu
		pip3 install tensorflow[and-cuda]
		pip3 install --no-cache-dir tensorboard
		pip3 install --no-cache-dir tensorflow-macos
		pip3 install -e .'[dev]'
		pip3 install -U sentence-transformers
		pip3 install git+https://github.com/huggingface/transformers

	elif [ "$OS" = "vcluster" ]; then
		pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

		# compatible with python3.10 and cuda 11.8
		pip3 install --no-cache-dir tensorflow==2.12.0
		pip3 install --no-cache-dir tensorflow_hub==0.12.0 tensorflow_text==2.12.0
		pip3 install -e .'[dev]'
		pip3 uninstall -y ninja && pip3 install --no-cache-dir ninja
		MAX_JOBS=7 pip3 install --no-cache-dir flash-attn --no-build-isolation
		pip3 install -U sentence-transformers
		pip3 install git+https://github.com/huggingface/transformers
		export TRITON_PTXAS_PATH=/pkgs/cuda-11.8/bin/ptxas
		pip3 install llm2vec fire wandb bitsandbytes

	elif [ "$OS" = "lambda" ]; then
		pip3 install --no-cache-dir torch torchvision torchaudio torchtext
		pip3 install --no-cache-dir tensorflow tensorboard
		pip3 install -e .'[dev]'
		pip3 install --no-cache-dir packaging
		pip3 uninstall -y ninja && pip3 install --no-cache-dir ninja
		MAX_JOBS=8 pip3 install --no-cache-dir flash-attn --no-build-isolation
		pip3 install llm2vec wandb bitsandbytes sentence_transformers
	fi


}

install_python
install_env
install_package
