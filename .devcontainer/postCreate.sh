#!/usr/bin/env bash
set -euo pipefail

# Determine whether privileged package install commands can use sudo.
if command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
else
  SUDO=""
fi

# Prepare user-writable cache paths for installers that rely on ~/.cache.
mkdir -p "$HOME/.cache"
if [ ! -w "$HOME/.cache" ]; then
  if [ -n "$SUDO" ]; then
    ${SUDO} chown -R "$(id -u):$(id -g)" "$HOME/.cache"
  else
    echo "Error: $HOME/.cache is not writable and sudo is unavailable."
    exit 1
  fi
fi

# Install core CLI tools needed by project workflows and shell setup.
if ! command -v ssh >/dev/null 2>&1; then
  echo "Installing OpenSSH client"
  ${SUDO} apt-get update
  ${SUDO} apt-get install -y openssh-client
fi

if ! command -v direnv >/dev/null 2>&1; then
  echo "Installing direnv"
  ${SUDO} apt-get update
  ${SUDO} apt-get install -y direnv
fi

if ! command -v task >/dev/null 2>&1; then
  echo "Installing go-task"
  if ! command -v curl >/dev/null 2>&1; then
    ${SUDO} apt-get update
    ${SUDO} apt-get install -y curl
  fi
  curl -fsSL https://taskfile.dev/install.sh | ${SUDO} sh -s -- -d -b /usr/local/bin
fi

# Install Rye (or reuse existing shims), then load Rye env for this script run.
if ! command -v rye >/dev/null 2>&1; then
  if [ -x "$HOME/.rye/shims/rye" ]; then
    export PATH="$HOME/.rye/shims:$PATH"
  else
    echo "Installing Rye"
    if ! command -v curl >/dev/null 2>&1; then
      ${SUDO} apt-get update
      ${SUDO} apt-get install -y curl
    fi
    curl -fsSL https://rye.astral.sh/get | RYE_INSTALL_OPTION="--yes" bash
  fi
fi

if [ -f "$HOME/.rye/env" ] && ! grep -q 'source "$HOME/.rye/env"' "$HOME/.bashrc"; then
  echo 'source "$HOME/.rye/env"' >> "$HOME/.bashrc"
fi

if [ -f "$HOME/.rye/env" ]; then
  . "$HOME/.rye/env"
fi

# Define workspace paths used for ownership repairs and env syncing.
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$WORKSPACE_DIR/.venv"
LOCKFILE_PATHS=(
  "$WORKSPACE_DIR/requirements.lock"
  "$WORKSPACE_DIR/requirements-dev.lock"
)

if [ -d "$VENV_DIR" ] && [ ! -w "$VENV_DIR" ]; then
  if [ -n "$SUDO" ]; then
    echo "Fixing ownership for existing virtual environment"
    ${SUDO} chown -R "$(id -u):$(id -g)" "$VENV_DIR"
  else
    echo "Error: $VENV_DIR is not writable and sudo is unavailable."
    exit 1
  fi
fi

if [ -L "$VENV_DIR/bin/python" ]; then
  VENV_PY_TARGET="$(readlink "$VENV_DIR/bin/python" || true)"
  if [[ "$VENV_PY_TARGET" == /root/* ]]; then
    echo "Removing stale virtual environment linked to root interpreter"
    rm -rf "$VENV_DIR"
  fi
fi

for LOCKFILE in "${LOCKFILE_PATHS[@]}"; do
  if [ -f "$LOCKFILE" ] && [ ! -w "$LOCKFILE" ]; then
    if [ -n "$SUDO" ]; then
      echo "Fixing ownership for $(basename "$LOCKFILE")"
      ${SUDO} chown "$(id -u):$(id -g)" "$LOCKFILE"
    else
      echo "Error: $LOCKFILE is not writable and sudo is unavailable."
      exit 1
    fi
  fi
done

# Ensure git repository is writable by current user for commits/operations.
GIT_DIR="$WORKSPACE_DIR/.git"
if [ -d "$GIT_DIR" ] && [ ! -w "$GIT_DIR" ]; then
  if [ -n "$SUDO" ]; then
    echo "Fixing ownership for .git directory"
    ${SUDO} chown -R "$(id -u):$(id -g)" "$GIT_DIR"
  else
    echo "Error: $GIT_DIR is not writable and sudo is unavailable."
    exit 1
  fi
fi

# Install shell prompt and shell integrations (starship, direnv, task completion).
if ! command -v starship >/dev/null 2>&1; then
  echo "Installing Starship"
  if ! command -v curl >/dev/null 2>&1; then
    ${SUDO} apt-get update
    ${SUDO} apt-get install -y curl
  fi
  curl -fsSL https://starship.rs/install.sh | ${SUDO} sh -s -- -y -b /usr/local/bin
fi

if ! grep -q 'eval "$(starship init bash)"' "$HOME/.bashrc"; then
  echo 'eval "$(starship init bash)"' >> "$HOME/.bashrc"
fi

if command -v direnv >/dev/null 2>&1 && ! grep -q 'eval "$(direnv hook bash)"' "$HOME/.bashrc"; then
  echo 'eval "$(direnv hook bash)"' >> "$HOME/.bashrc"
fi

if command -v task >/dev/null 2>&1 && ! grep -q 'eval "$(task --completion bash)"' "$HOME/.bashrc"; then
  echo 'eval "$(task --completion bash)"' >> "$HOME/.bashrc"
fi

# Trust repo-local .envrc so environment variables load on shell entry.
if [ -f "$WORKSPACE_DIR/.envrc" ] && command -v direnv >/dev/null 2>&1; then
  echo "Allowing direnv for workspace .envrc"
  direnv allow "$WORKSPACE_DIR"
fi

# Create/update the project virtual environment and dependency set.
echo "Syncing Rye-managed project environment"
rye sync --features preprocessing --features training-core --features training-onnx --features streamlit

echo "Dev container setup complete."