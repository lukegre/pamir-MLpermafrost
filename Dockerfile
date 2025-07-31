FROM renku/renkulab-vscode-python:py-3.12.7-5423b62

USER vscode

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv venv --python=python3.12 --venv-dir=/home/vscode/work/.venv-python3.12
#
