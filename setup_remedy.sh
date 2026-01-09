#!/bin/bash -i
set -e

# ==========================================
# 0. CONFIGURATION
# ==========================================
# Your huge project path
PROJECT_DIR="/project/home/p201026"
# Where the environment will live
ENV_PATH="$PROJECT_DIR/remedy_metric"
# Where Miniconda will live (or lives already)
CONDA_PATH="$PROJECT_DIR/miniconda"

# ==========================================
# 1. Redirect ALL Cache (Vital for Quota)
# ==========================================
export PIP_CACHE_DIR="$PROJECT_DIR/.cache_pip"
export CONDA_PKGS_DIRS="$PROJECT_DIR/.cache_conda_pkgs"
mkdir -p "$PIP_CACHE_DIR"
mkdir -p "$CONDA_PKGS_DIRS"

# ==========================================
# 2. Locate or Install Miniconda
# ==========================================
if [ ! -f "$CONDA_PATH/bin/conda" ]; then
    echo "Miniconda not found in project storage. Installing it now..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -u -p "$CONDA_PATH"
    rm miniconda.sh
    echo "Miniconda installed at $CONDA_PATH"
fi

# Source Conda
source "$CONDA_PATH/etc/profile.d/conda.sh"

# ==========================================
# 3. Create Environment (Python >= 3.12)
# ==========================================
if [ -d "$ENV_PATH" ]; then
    echo "Environment already exists at: $ENV_PATH"
    echo "Skipping creation..."
else
    echo "Creating environment at: $ENV_PATH"
    # Using python 3.12 as required
    conda create -p "$ENV_PATH" python=3.12 -c conda-forge -y
fi

# Activate
conda activate "$ENV_PATH"

# ==========================================
# 4. Install Libraries
# ==========================================
echo "Installing dependencies..."

# Auto-accept ToS for Conda
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=true

# 1. Install PyTorch >= 2.6.0 (Using nightly or stable depending on availability for 3.12)
# Since 2.6.0 is very new/nightly, we target the specific requirements.
# If 2.6.0 isn't on the main channel yet, we usually use the nightly channel (pytorch-nightly).
# However, vllm 0.9.2 is quite specific.
# Let's try installing dependencies via PIP to match your setup.py strictness exactly,
# because Conda channels might lag behind on "vllm==0.9.2" and "torch>=2.6.0".

echo "Installing vLLM, PyTorch and other requirements via pip..."

# Installing build tools first just in case
pip install --upgrade pip setuptools wheel

# Install the strict list provided
pip install "vllm==0.9.2" \
            "torch>=2.6.0" \
            "transformers<4.54.0" \
            "datasets>=3.1.0" \
            "matplotlib>=3.10.0" \
            "hf-transfer>=0.1.8" \
            "trl>=0.12.0" \
            "scikit-learn>=1.6.0" \
            "openpyxl" \
            "pandas"

# ==========================================
# 5. Install Remedy Package (Current Dir)
# ==========================================
# Assuming this script is run where setup.py is located
if [ -f "setup.py" ]; then
    echo "Found setup.py, installing current package in editable mode..."
    pip install -e .
else
    echo "Warning: setup.py not found in current directory. "
    echo "You can install it later by running 'pip install -e .' inside the folder."
fi

echo "=========================================="
echo " Setup Complete!"
echo " To activate this environment, run:"
echo " source $CONDA_PATH/bin/activate"
echo " conda activate $ENV_PATH"
echo "=========================================="