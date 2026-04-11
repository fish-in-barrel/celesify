#!/usr/bin/env bash
set -euo pipefail

if ! command -v direnv >/dev/null 2>&1; then
  echo "Installing direnv"
  sudo apt-get update
  sudo apt-get install -y direnv
fi

if ! command -v task >/dev/null 2>&1; then
  echo "Installing go-task"
  if ! command -v curl >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y curl
  fi
  curl -fsSL https://taskfile.dev/install.sh | sudo sh -s -- -d -b /usr/local/bin
fi

python3 -m pip install --upgrade pip

# Install requirements used by each service so scripts run directly in the dev container.
for req in services/*/requirements.txt; do
  if [ -f "$req" ]; then
    echo "Installing dependencies from $req"
    pip3 install -r "$req"
  fi
done

echo "Dev container setup complete."