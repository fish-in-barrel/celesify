#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip

# Install requirements used by each service so scripts run directly in the dev container.
for req in services/*/requirements.txt; do
  if [ -f "$req" ]; then
    echo "Installing dependencies from $req"
    pip3 install -r "$req"
  fi
done

echo "Dev container setup complete."