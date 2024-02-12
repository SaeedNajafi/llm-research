#!/bin/bash

function install_env () {
	python3 -m venv $ENV_NAME-env
	source $ENV_NAME-env/bin/activate
	mkdir -p $HOME/tmp
	export TMPDIR=$HOME/tmp
	python3 -m pip install --upgrade pip
}


function install_package () {
	# Installs pre-commit tools as well.
	python3 -m pip install -e .'[dev]'
}

install_env
install_package