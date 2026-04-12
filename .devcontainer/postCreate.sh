#!/usr/bin/env bash
set -euo pipefail

if command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
else
  SUDO=""
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

if ! command -v rye >/dev/null 2>&1; then
  echo "Installing Rye"
  if ! command -v curl >/dev/null 2>&1; then
    ${SUDO} apt-get update
    ${SUDO} apt-get install -y curl
  fi
  curl -fsSL https://rye.astral.sh/get | RYE_INSTALL_OPTION="--yes" bash
fi

if [ -f "$HOME/.rye/env" ] && ! grep -q 'source "$HOME/.rye/env"' "$HOME/.bashrc"; then
  echo 'source "$HOME/.rye/env"' >> "$HOME/.bashrc"
fi

if [ -f "$HOME/.rye/env" ]; then
  . "$HOME/.rye/env"
fi

echo "Syncing Rye-managed project environment"
rye sync --features preprocessing --features training-core --features training-onnx --features streamlit

python3 - <<'PY'
try:
    from cuml.ensemble import RandomForestClassifier  # noqa: F401
except Exception as exc:
    print(f"cuML import check failed: {exc}")
else:
    print("cuML import check passed.")
PY

echo "Dev container setup complete."