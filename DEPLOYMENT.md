# Streamlit Community Cloud Deployment Guide

Deploy celesify to celesify.space using Streamlit Community Cloud.

## Prerequisites

- ✅ GitHub repo is public: `fish-in-barrel/celesify`
- ✅ Streamlit account (free) at https://streamlit.io
- ✅ celesify.space domain registered and DNS access available

## Deployment Steps

### 1. Generate Training Artifacts Locally (Recommended for Development)

**Primary Approach: Manual Build via `task deploy`**

```bash
task deploy
```

This single command:
1. Runs the full training pipeline (preprocessing → training → ONNX export)
2. Creates `celesify-artifacts.zip` containing all model artifacts
3. Uploads to GitHub Releases as `latest` (auto-creates/updates release)

**Expected output:**
```
Training complete...
✓ Created celesify-artifacts.zip (XX MB)
✓ Released to GitHub
```

Troubleshooting:
- `gh: not found` → Install GitHub CLI: `brew install gh` (or package manager for your OS)
- `gh auth: not authenticated` → Run `gh auth login` and paste your personal access token
- Training fails → Check `data/raw/star_classification.csv` exists and has data

**Alternative: Build Locally Without Release**

```bash
rye run train
zip -r celesify-artifacts.zip outputs/
gh release create latest celesify-artifacts.zip --latest --title "Training Artifacts ($(date +%Y-%m-%d))"
```

**Optional: GitHub Actions Automation (Disabled by Default)**

To enable automatic rebuilds on every push to `main`, edit `.github/workflows/build-artifacts.yml` and uncomment the `on: push:` section. See that file for details. This workflow is currently disabled for flexibility during active development.

### 2. Connect to Streamlit Community Cloud

1. Go to https://share.streamlit.io
2. Log in with GitHub account
3. Click "New app"
4. Select repository: `fish-in-barrel/celesify`
5. Set branch: `main`
6. Set main file path: `streamlit_app.py`
7. Click "Deploy"

Streamlit will:
- Install dependencies from `requirements.txt`
- Run `streamlit_app.py` (which calls `setup_artifacts.py`)
- Download latest artifacts from GitHub Releases
- Start the dashboard on a `*.streamlit.app` URL

### 3. Set Up DNS

Point `celesify.space` to Streamlit's managed domain:

1. In Streamlit dashboard settings, copy your app's URL (e.g., `celesify-xyz.streamlit.app`)
2. Add CNAME record in your domain DNS:
   ```
   celesify.space CNAME celesify-xyz.streamlit.app
   ```
3. Wait for DNS propagation (~5-30 minutes)

### 4. Verify Deployment

- Visit https://celesify.space
- Check that the app loads
- Test data exploration, model evaluation, and inference tabs
- Monitor logs: Streamlit dashboard > App > Logs

## Updating the Deployment

### When Code Changes

After any changes to code or hyperparameters, update the production app:

1. **Rebuild artifacts locally:**
   ```bash
   task deploy
   ```
   This rebuilds the trained model and updates GitHub Releases.

2. **Streamlit Auto-Redeploys**
   - Every push to `main` triggers a Streamlit redeploy (1–2 minutes)
   - The app auto-downloads the latest artifacts from GitHub Releases at startup
   - Refresh your browser to see the update

**Timeline:**
- `task deploy` runs (10–30 min depending on training time)
- `git push` to upload changes
- Streamlit detects push → redeploys (2–3 min)
- App refreshes and pulls new artifacts (automatic)

### Testing Locally Before Deployment

```bash
# Test the app on localhost before public release
task run-streamlit
```

Visit http://localhost:8501 to verify:
- ✅ Data Explorer tab works
- ✅ Performance Metrics tab shows results
- ✅ Upload & Infer returns predictions
- ✅ No error messages

## Environment Variables

No special environment variables needed for Streamlit Community Cloud. The app automatically detects the cloud environment and handles artifact downloads.

## Troubleshooting

### "`task deploy` fails with "gh: command not found"

Install GitHub CLI:
```bash
# macOS
brew install gh

# Linux (Debian/Ubuntu)
sudo apt-get install gh

# Or: https://cli.github.com/manual/installation
```

Then authenticate:
```bash
gh auth login
# Paste your personal access token when prompted
```

### Artifacts Not Downloading in Streamlit Cloud

Check that GitHub release exists:
```bash
gh release view latest --json assets
```

**Expected output:** Should list `celesify-artifacts.zip` file

**If missing:**
1. Run `task deploy` locally to create the release
2. Verify release is public: `gh release view latest`
3. Restart the Streamlit app (Streamlit dashboard > Manage app > Reboot)

### "Streamlit app is loading forever"

Check app logs in Streamlit dashboard:
1. Go to https://share.streamlit.io
2. Click your app
3. Click **Manage app** → **Logs**
4. Look for errors from `setup_artifacts.py`

**Common causes:**
- GitHub release doesn't exist → run `task deploy` locally
- Network blocked → GitHub API unreachable from Streamlit → check repository is public
- Corrupted ZIP → rebuild: `task deploy`

### Artifacts Downloaded But Models Won't Load

Check `outputs/models/` folder exists with these files:
- `model.joblib` — trained Random Forest model
- `baseline_metrics.json` — baseline evaluation
- `tuned_metrics.json` — tuned model evaluation
- `clean_tuned_metrics.json` — clean tuned evaluation
- `best_params.json` — hyperparameters
- `feature_importance.json` — feature rankings

**To fix:**
1. Run training: `rye run train`
2. Verify outputs: `ls -la outputs/models/`
3. Rebuild release: `task deploy`
4. Restart Streamlit app from dashboard

### DNS Not Resolving

Verify CNAME record is set:
```bash
dig celesify.space
# Should show: celesify.space CNAME celesify-xyz.streamlit.app
```

If not set:
1. Log into your domain registrar
2. Add CNAME record: `celesify.space` → your Streamlit app URL
3. Wait 5–30 minutes for propagation
4. Test with: `dig celesify.space`

### Local Training Takes Too Long

For quick iteration/testing:
```bash
TRAINING_N_ITER=1 TRAINING_CV_SPLITS=2 rye run train
```

This does minimal tuning (~2–3 min). For full training, use: `rye run train` or `task deploy`

## Quick Start Checklist

✅ **Local Setup:**
- Install `rye` and authenticate with `gh auth login`
- Run `task run-streamlit` and verify app works locally

✅ **First Deployment:**
- Run `task deploy` to build artifacts and create GitHub release
- Connect repo to Streamlit Community Cloud dashboard
- Verify app deploys successfully

✅ **DNS Setup:**
- Add CNAME record in your registrar: `celesify.space` → Streamlit app URL
- Wait for propagation and verify with `dig celesify.space`

✅ **Public Access:**
- Visit https://celesify.space once DNS is live

## Cost

- **Streamlit Community Cloud**: $0/month (free tier)
- **Domain celesify.space**: ~$12-15/year depending on registrar
- **Total**: ~$15/year

## Monitoring

Access from Streamlit dashboard:
- **Health**: View app status and uptime
- **Logs**: Real-time logs for debugging
- **Settings**: Configure Python version, secrets, etc.

---

For issues or questions, see [README.md](README.md) or create a GitHub Issue.

