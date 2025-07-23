name ?= new_notebook

# some makefile magic commands
.DEFAULT_GOAL := help
.PHONY: help

new-notebook:  ## creates a new notebook from the template. set name variable to give a non-default name
	@cp -r notebooks/.template.ipynb notebooks/$(name).ipynb

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
