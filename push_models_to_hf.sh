#!/usr/bin/env bash


# Automatically handle branch detection and creation.
# Use the safe, universal --include-ref=HEAD for git lfs migrate import.
# Automatically create or update Hugging Face model repos.
# Keep your main GitHub repo completely untouched.

# 🧠 What this version does

# Detects all subfolders inside /models/.
# For each one:
# Ensures binary files (*.keras, *.h5, *.png, *.jpg, *.jpeg, *.csv, *.json) are tracked by Git LFS.
# Creates a temporary mini-repo just for that model.
# Runs git lfs migrate import --include-ref=HEAD so all relevant files are properly stored via LFS.
# Pushes to Hugging Face — creating the repo if needed, or force-updating it if it already exists.


# ✅ What You’ll Get

# Each folder inside /models → separate Hugging Face repo.
# Repos created automatically if missing.
# Existing repos updated (force-pushed).
# All binary/model files correctly stored via LFS.
# Main GitHub project stays untouched.


# usage: 
# Run it from your project root: ~/DataScience/ds_aug24_lung_desease_classification (master)
# $ ./push_models_to_hf.sh 

set -e  # Exit on error

# --------------------------------------
# CONFIGURATION
# --------------------------------------
HF_USERNAME="valste"          # 🔧 Your Hugging Face username
MODEL_ROOT="models"           # Folder containing all model subdirectories
LFS_PATTERNS=("*.keras" "*.h5" "*.png" "*.jpg" "*.jpeg" "*.csv" "*.json")

# --------------------------------------
# CHECK DEPENDENCIES
# --------------------------------------
for cmd in git git-lfs hf curl; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "❌ Missing dependency: $cmd"
    exit 1
  fi
done

# --------------------------------------
# INITIALIZE GIT LFS (global)
# --------------------------------------
git lfs install
for pattern in "${LFS_PATTERNS[@]}"; do
  git lfs track "$pattern" || true
done
git add .gitattributes
git commit -m "Ensure LFS tracking for binary files" || true

# --------------------------------------
# MAIN LOOP
# --------------------------------------
cd "$MODEL_ROOT" || exit 1

for folder in */; do
  folder="${folder%/}"
  repo_name="$folder"
  repo_full="$HF_USERNAME/$repo_name"
  remote_url="https://huggingface.co/$repo_full"

  echo ""
  echo "🚀 Processing model: $repo_full"
  echo "--------------------------------------"

  # Check if repo exists
  if curl --silent --fail "https://huggingface.co/api/models/$repo_full" >/dev/null; then
    echo "ℹ️ Repo already exists on Hugging Face."
    repo_exists=true
  else
    echo "🆕 Creating Hugging Face repo: $repo_full"
    hf repo create "$repo_name" --repo-type model || {
      echo "❌ Failed to create repo $repo_full"
      continue
    }
    repo_exists=false
  fi

  # --------------------------------------
  # CREATE TEMPORARY MINI-REPO
  # --------------------------------------
  TMP_DIR=$(mktemp -d)

  # 🧹 Remove nested .git directories before copying
  find "$folder" -type d -name ".git" -exec rm -rf {} + 2>/dev/null || true

  # ✅ Copy only the contents (not the folder itself)
  cp -r "$folder"/* "$TMP_DIR"/

  pushd "$TMP_DIR" >/dev/null

  git init -q
  git lfs install
  for pattern in "${LFS_PATTERNS[@]}"; do
    git lfs track "$pattern" || true
  done
  git add .
  git commit -m "Update model: $repo_name" || true
  git branch -M main

  # --- Safe LFS migration for all tracked binary types ---
  echo "🔄 Migrating existing files to Git LFS..."
  git lfs migrate import --include="*.keras,*.h5,*.png,*.jpg,*.jpeg,*.csv,*.json" --include-ref=HEAD || true

  # Add and push
  git remote add origin "$remote_url"
  if [ "$repo_exists" = true ]; then
    echo "📤 Updating existing repo on Hugging Face..."
  else
    echo "📤 Pushing new repo to Hugging Face..."
  fi

  git push origin main -f
  echo "✅ Synced → https://huggingface.co/$repo_full"

  popd >/dev/null
  rm -rf "$TMP_DIR"
done

echo ""
echo "🎉 All model folders have been created or updated on Hugging Face!"