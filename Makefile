name ?= $(notdir $(shell pwd))

# some makefile magic commands
.DEFAULT_GOAL := help
.PHONY: help

new-notebook:  ## creates a new notebook from the template. set name variable to give a non-default name
	@cp -r notebooks/.template.ipynb notebooks/$(name).ipynb

jupyter-kernel:  ## creates a jupyter-notebook kernel for this project
	@uv sync
	@uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=$(name)

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
