#!/usr/bin/env bash
set -euo pipefail

# Helper functions to reduce repetition.
SUDO=""; command -v sudo >/dev/null 2>&1 && SUDO="sudo"

ensure_apt() {
  [ ! "$#" -gt 0 ] && return
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Installing $1"
    ${SUDO} apt-get update
    ${SUDO} apt-get install -y "$1"
  fi
}

ensure_curl_and_install() {
  local name=$1 install_cmd=$2
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "Installing $name"
    ensure_apt curl
    eval "$install_cmd"
  fi
}

add_bashrc_line() {
  local line=$1
  grep -q "$line" "$HOME/.bashrc" || echo "$line" >> "$HOME/.bashrc"
}

fix_ownership() {
  local path=$1 desc=$2
  if [ -e "$path" ] && [ ! -w "$path" ]; then
    if [ -n "$SUDO" ]; then
      echo "Fixing ownership for $desc"
      ${SUDO} chown -R "$(id -u):$(id -g)" "$path"
    else
      echo "Error: $path not writable; sudo unavailable."
      exit 1
    fi
  fi
}

# Prepare cache and install core tools.
mkdir -p "$HOME/.cache"
mkdir -p "$HOME/.cache/kaggle"
mkdir -p "$HOME/.cache/pip"
fix_ownership "$HOME/.cache" "cache directory"
fix_ownership "$HOME/.cache/kaggle" "Kaggle cache"
fix_ownership "$HOME/.cache/pip" "pip cache"

ensure_apt openssh-client
ensure_apt direnv
ensure_curl_and_install task 'curl -fsSL https://taskfile.dev/install.sh | ${SUDO} sh -s -- -d -b /usr/local/bin'

# Install Rye.
ensure_curl_and_install rye 'curl -fsSL https://rye.astral.sh/get | RYE_INSTALL_OPTION="--yes" bash'

if [ -x "$HOME/.rye/shims/rye" ]; then
  export PATH="$HOME/.rye/shims:$PATH"
fi

add_bashrc_line 'source "$HOME/.rye/env"'
[ -f "$HOME/.rye/env" ] && . "$HOME/.rye/env"

# Setup workspace paths and fix permissions.
# Ensure all workspace files are owned by the current (vscode) user to prevent
# permission issues across container rebuilds and different execution contexts.
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CURRENT_UID=$(id -u)
CURRENT_GID=$(id -g)

echo "Fixing workspace permissions for UID:GID $CURRENT_UID:$CURRENT_GID"

# Fix ownership of the entire workspace recursively
if [ "$CURRENT_UID" -ne 0 ]; then
  # Running as non-root; use sudo to fix ownership
  ${SUDO} find "$WORKSPACE_DIR" \! -path '*/.venv/*' \! -path '*/__pycache__/*' \! -path '*/.git/*' -type f -print0 2>/dev/null | ${SUDO} xargs -0 chown "$CURRENT_UID:$CURRENT_GID" 2>/dev/null || true
  ${SUDO} find "$WORKSPACE_DIR" \! -path '*/.venv/*' \! -path '*/__pycache__/*' \! -path '*/.git/*' -type d -print0 2>/dev/null | ${SUDO} xargs -0 chown "$CURRENT_UID:$CURRENT_GID" 2>/dev/null || true
else
  # Running as root; no sudo needed
  find "$WORKSPACE_DIR" \! -path '*/.venv/*' \! -path '*/__pycache__/*' \! -path '*/.git/*' -type f -exec chown "$CURRENT_UID:$CURRENT_GID" {} \; 2>/dev/null || true
  find "$WORKSPACE_DIR" \! -path '*/.venv/*' \! -path '*/__pycache__/*' \! -path '*/.git/*' -type d -exec chown "$CURRENT_UID:$CURRENT_GID" {} \; 2>/dev/null || true
fi

# Handle .venv and .git separately in case they exist
fix_ownership "$WORKSPACE_DIR/.venv" "virtual environment"
fix_ownership "$WORKSPACE_DIR/.git" ".git directory"

for lockfile in "$WORKSPACE_DIR/requirements.lock" "$WORKSPACE_DIR/requirements-dev.lock"; do
  fix_ownership "$lockfile" "$(basename "$lockfile")"
done

# Handle stale venv linked to root interpreter.
if [ -L "$WORKSPACE_DIR/.venv/bin/python" ]; then
  target=$(readlink "$WORKSPACE_DIR/.venv/bin/python" || true)
  [[ "$target" == /root/* ]] && rm -rf "$WORKSPACE_DIR/.venv"
fi

# Fix permissions on data and outputs directories (including named volumes).
# These are often created by different processes and may have root ownership.
for dir in "$WORKSPACE_DIR/data" "$WORKSPACE_DIR/outputs"; do
  if [ -d "$dir" ]; then
    echo "Fixing permissions for $dir"
    ${SUDO} find "$dir" -type f -exec chown "$CURRENT_UID:$CURRENT_GID" {} \; 2>/dev/null || true
    ${SUDO} find "$dir" -type d -exec chown "$CURRENT_UID:$CURRENT_GID" {} \; 2>/dev/null || true
    ${SUDO} find "$dir" -type f -exec chmod 644 {} \; 2>/dev/null || true
    ${SUDO} find "$dir" -type d -exec chmod 755 {} \; 2>/dev/null || true
  fi
done

# Ensure data directories exist with correct permissions
for subdir in "data/raw" "outputs/models" "outputs/processed"; do
  mkdir -p "$WORKSPACE_DIR/$subdir"
  ${SUDO} chown "$CURRENT_UID:$CURRENT_GID" "$WORKSPACE_DIR/$subdir" 2>/dev/null || true
  chmod 755 "$WORKSPACE_DIR/$subdir"
done

# Install shell integrations.
ensure_curl_and_install starship 'curl -fsSL https://starship.rs/install.sh | ${SUDO} sh -s -- -y -b /usr/local/bin'
add_bashrc_line 'eval "$(starship init bash)"'
add_bashrc_line 'eval "$(direnv hook bash)"'
add_bashrc_line 'eval "$(task --completion bash)"'

# Trust repo-local .envrc.
[ -f "$WORKSPACE_DIR/.envrc" ] && command -v direnv >/dev/null 2>&1 && direnv allow "$WORKSPACE_DIR"

source ~/.bashrc

echo "Dev container setup complete."
