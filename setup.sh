#!/usr/bin/env bash
set -e  # exit on first error

# --- CONFIG ---
PYTHON_VERSION="python3.11"   # required Python version
VENV_NAME="venv_dsCovid"              # name of the virtual environment folder
# ---------------

# Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$SCRIPT_DIR"     # project root is where setup.sh lives
VENV_DIR="${SCRIPT_DIR}/../${VENV_NAME}"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"

# Helper function: print step messages
step () {
  echo -e "\n👉 $1"
}

# --- PYTHON VERSION CHECK ---
step "Checking Python version..."
CURRENT_VERSION=$($PYTHON_VERSION -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") || {
  echo "❌ Could not find $PYTHON_VERSION on your system."
  echo "Please install Python 3.11 before continuing."
  exit 1
}

if [[ "$CURRENT_VERSION" != "3.11" ]]; then
  echo "⚠️ WARNING: This project requires Python 3.11, but you are using $CURRENT_VERSION"
  echo "Please install Python 3.11 to avoid compatibility issues."
  echo
  read -rp "Continue anyway? [y/N]: " cont
  if [[ ! "$cont" =~ ^[Yy]$ ]]; then
    echo "Exiting setup."
    exit 1
  fi
else
  echo "✅ Python version check passed ($CURRENT_VERSION)"
fi

# --- PROMPT USER ---
echo
echo "Do you want to:"
echo "  [c] Create new virtual environment"
echo "  [s] Skip venv and just add the project directory to PYTHONPATH"
read -rp "Select an option [c/s]: " choice

if [[ "$choice" == "c" ]]; then
    step "Creating virtual environment at $VENV_DIR ..."
    if [ ! -d "$VENV_DIR" ]; then
        $PYTHON_VERSION -m venv "$VENV_DIR"
    else
        echo "⚠️ Virtual environment already exists, skipping creation."
    fi

    # Detect OS for correct venv activation path
    if [[ "$OS" == "Windows_NT" ]]; then
        ACTIVATE_PATH="${VENV_DIR}/Scripts/activate"
    else
        ACTIVATE_PATH="${VENV_DIR}/bin/activate"
    fi

    step "Activating virtual environment ..."
    # shellcheck disable=SC1090
    source "$ACTIVATE_PATH"

    if [ -f "$REQ_FILE" ]; then
        step "Installing requirements from $REQ_FILE ..."
        python -m pip install --upgrade pip
        python -m pip install -r "$REQ_FILE"
    else
        echo "⚠️ No requirements.txt found in $SCRIPT_DIR"
    fi

    step "Adding project folder to PYTHONPATH ..."
    export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
    echo "export PYTHONPATH=\"$PROJECT_DIR:\$PYTHONPATH\"" >> "${VENV_DIR}/bin/activate"

    echo -e "\n✅ Setup complete!"
    echo "To activate the virtual environment, run:"
    echo "   • On Windows \(Git Bash\): source ../venv_rag/Scripts/activate"
    echo "   • On macOS/Linux:        source ../venv_rag/bin/activate"
    

elif [[ "$choice" == "s" ]]; then
    step "Skipping venv creation. Adding project folder to PYTHONPATH ..."
    export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
    echo "export PYTHONPATH=\"$PROJECT_DIR:\$PYTHONPATH\"" >> ~/.bashrc

    echo -e "✅ Project folder added to PYTHONPATH."
    echo "You may need to restart your shell or run: source ~/.bashrc"
    echo "Afterwards, run src/streamlit/Home.py in the python shell inside of an appropriate environment to explore the demo app."

else
    echo "❌ Invalid option. Please run again and choose \[c\] or \[s\]."
    exit 1
fi

# --- PROMPT TO DOWNLOAD MODELS ---

# Check for dry-run flag
DRY_RUN=false
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN=true
    fi
done

echo "Would you like to download the models right away [d] or skip this step [s]?"
echo "--->Selecting [d] will overwrite any existing models in the /models directory!<---"
read -rp "[d/s]: " USER_CHOICE

if [[ "$USER_CHOICE" == "d" ]]; then
    echo "🚀 Starting model download..."
    if [[ "$DRY_RUN" == true ]]; then
        bash ./download_models.sh --dry-run
    else
        bash ./download_models.sh
    fi
else
    echo "❌ Skipping model download."
fi

