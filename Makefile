name ?= $(notdir $(shell pwd))

# some makefile magic commands
.DEFAULT_GOAL := help
.PHONY: help

new-notebook:  ## creates a new notebook from the template. set name variable to give a non-default name
	@cp -r notebooks/.template.ipynb notebooks/$(name).ipynb

env:  ## sets everything up when running on a renkulab session
	@pip3 install uv
	@uv sync

mlflow-ui: env  ## runs an mlflow ui server ensuring that env is set up
	uv run mlflow ui

renku-tunnel:  ## opens a tunnel from the renkulab session
	@bash src/start_tunnel.sh

jupyter-kernel:  ## creates a jupyter-notebook kernel for this project
	@uv sync
	@uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=$(name)

runs:  ## runs the hydra multirun script
	@uv run bash notebooks/run-hydra-multirun.sh

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
