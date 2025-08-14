#!/bin/bash

set -e

RENKU_WORKING_DIR="${RENKU_WORKING_DIR:-${HOME}}"
RENKU_MOUNT_DIR="${RENKU_MOUNT_DIR:-${RENKU_WORKING_DIR}}"
BIN_FOLDER="${RENKU_MOUNT_DIR}/.local/bin"
CODE_CLI_PATH="${BIN_FOLDER}/code"

# Detect if code CLI is already installed
INSTALL_CODE_CLI="1"
if [ -f "${CODE_CLI_PATH}" ]; then
    INSTALL_CODE_CLI=$("${CODE_CLI_PATH}" --version >/dev/null && echo "0" || echo "1")
    if [ "${INSTALL_CODE_CLI}" = "1" ]; then
        echo "WARNING: code CLI will overwite file at ${CODE_CLI_PATH}"
    else
        echo "Found existing code CLI at ${CODE_CLI_PATH}"
    fi
fi

# Download code CLI
if [ "${INSTALL_CODE_CLI}" = "1" ]; then
    PLATFORM=$(uname -m)
    if [ "${PLATFORM}" = "x86_64" ]; then
        DOWNLOAD_URL="https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64"
    elif [ "${PLATFORM}" = "aarch64" ]; then
        DOWNLOAD_URL="https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-arm64"
    else
        echo "Unsupported platform: ${PLATFORM}"
        exit 1
    fi

    TMP="$(mktemp -d)"
    mkdir -p "${BIN_FOLDER}"

    echo "Downloading code CLI from ${DOWNLOAD_URL}..."
    curl "${DOWNLOAD_URL}" -Lo "${TMP}/code_cli.tar.gz"

    echo "Extracting code CLI..."
    tar xf "${TMP}/code_cli.tar.gz" -C "${TMP}"

    echo "Moving code CLI to ${CODE_CLI_PATH}..."
    cp "${TMP}/code" "${CODE_CLI_PATH}"
    chmod a+x "${CODE_CLI_PATH}"

    rm -r "${TMP}"
fi

# Update code CLI
"${CODE_CLI_PATH}" update || true
"${CODE_CLI_PATH}" --version

# Generate stable tunnel name if possible
if [ -n "${RENKU_BASE_URL_PATH}" ]; then
    TUNNEL_NAME=$(echo -n "${RENKU_BASE_URL_PATH}" | md5sum)
elif [ -n "${HOSTNAME}" ]; then
    TUNNEL_NAME=$(echo -n "${HOSTNAME}" | md5sum)
else
    echo "Warning: could not find session name or hostname, using random string"
    TUNNEL_NAME=$(LC_ALL=C tr -dc "a-z0-9" </dev/urandom 2>/dev/null | head -c 20)
fi
TUNNEL_NAME=$(echo "renku-${TUNNEL_NAME}" | cut -c 1-20)
echo "TUNNEL_NAME: ${TUNNEL_NAME}"

"${CODE_CLI_PATH}" tunnel \
    --name "${TUNNEL_NAME}" \
    --server-data-dir "${RENKU_MOUNT_DIR}/.vscode_tunnel" \
    --extensions-dir "${RENKU_MOUNT_DIR}/.vscode_tunnel/extensions" \
    --cli-data-dir "${RENKU_MOUNT_DIR}/.vscode_tunnel/cli"