# Kaggle Credentials Setup with `.envrc`

This project can auto-download the SDSS17 dataset in the preprocessing container.
To enable this, set Kaggle credentials in a local `.envrc` file.

## 0. Install `direnv` (optional but recommended)

Ubuntu / Debian:

```bash
sudo apt-get update
sudo apt-get install -y direnv
```

Enable the shell hook (bash):

```bash
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
source ~/.bashrc
```

If you use zsh, use:

```bash
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
source ~/.zshrc
```

## 1. Create your local `.envrc`

From the repository root, create `.envrc` from the template:

```bash
cp .envrc.template .envrc
```

Edit `.envrc` and set your Kaggle credentials:

```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_kaggle_api_key"
```

## 2. Load env vars into your shell

If you use `direnv`:

```bash
direnv allow
```

If you do not use `direnv`, source the file manually:

```bash
set -a
. ./.envrc
set +a
```

## 3. Run preprocessing

```bash
docker compose run --rm --build preprocessing
```

The preprocessing service reads `KAGGLE_USERNAME` and `KAGGLE_KEY` from your shell and uses the Kaggle API when no CSV is found under `data/raw`.

## Notes

- `.envrc` files are ignored by git to prevent secret leaks.
- `.envrc.template` is committed as the safe template.
- You can verify vars are loaded with:

```bash
echo "$KAGGLE_USERNAME"
```
