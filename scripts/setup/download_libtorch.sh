#!/bin/bash
# check if third-party/libtorch exists and download if not
if [ -d "third-party/libtorch" ]; then
    echo "libtorch already exists in third-party"
    exit 0
fi

echo "Checking if PyTorch uses CXX11 ABI"
if python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)" | grep -q "True"; then
    echo "[DEBUG] PyTorch uses CXX11 ABI"
    LIBTORCH_URL="https://download.pytorch.org/libtorch/nightly/cu124/libtorch-cxx11-abi-shared-with-deps-latest.zip"
else
    echo "[DEBUG] PyTorch does not use CXX11 ABI"
    LIBTORCH_URL="https://download.pytorch.org/libtorch/nightly/cu124/libtorch-shared-with-deps-latest.zip"
fi

echo "[INFO] Downloading libtorch from $LIBTORCH_URL to third-party"

mkdir -p third-party
wget -O third-party/libtorch.zip $LIBTORCH_URL

echo "Extracting libtorch to third-party"
unzip -q third-party/libtorch.zip -d third-party


