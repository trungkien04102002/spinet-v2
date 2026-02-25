#!/bin/bash
set -e

echo "=== SpineNet Vast.ai Setup Script ==="

# Update system packages
echo "Updating system packages..."
apt-get update && apt-get install -y git wget

# Clone repository if not exists
REPO_DIR="SpineNetV2"
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning SpineNetV2 repository..."
    git clone https://github.com/rwindsor1/SpineNetV2.git
else
    echo "Repository already exists, skipping clone..."
fi

cd $REPO_DIR

# Create and activate virtual environment
echo "Creating virtual environment..."
if [ ! -d "spinenet-venv" ]; then
    python3 -m venv spinenet-venv
fi
source spinenet-venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Set PYTHONPATH permanently
echo "Setting PYTHONPATH..."
SPINENET_PATH=$(pwd)
export PYTHONPATH=$PYTHONPATH:$SPINENET_PATH

# Add to bashrc if not already there
if ! grep -q "PYTHONPATH.*SpineNetV2" ~/.bashrc; then
    echo "export PYTHONPATH=\$PYTHONPATH:$SPINENET_PATH" >> ~/.bashrc
fi

# Verify CUDA availability
echo ""
echo "=== Checking CUDA availability ==="
python3 << EOF
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device count: {torch.cuda.device_count()}')
    print(f'GPU device name: {torch.cuda.get_device_name(0)}')
else:
    print('WARNING: CUDA not available! SpineNet will run on CPU (much slower)')
EOF

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To run Python scripts:"
echo "  cd $REPO_DIR"
echo "  source spinenet-venv/bin/activate"
echo "  python test_spinenet.py"
echo ""
echo "The test script is configured to use CUDA. Develop with PyCharm/VSCode!"
