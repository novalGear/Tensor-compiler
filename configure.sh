#!/bin/bash
# configure.sh - запускать в корне проекта

# Определяем версию WSL
if grep -q Microsoft /proc/version; then
    echo "Running in WSL"
fi

# Поиск LLVM
if [ -d "/usr/lib/llvm-17" ]; then
    LLVM_DIR="/usr/lib/llvm-17/lib/cmake/llvm"
    MLIR_DIR="/usr/lib/llvm-17/lib/cmake/mlir"
    echo "Using LLVM 17 from apt"
elif [ -d "/usr/lib/llvm-16" ]; then
    LLVM_DIR="/usr/lib/llvm-16/lib/cmake/llvm"
    MLIR_DIR="/usr/lib/llvm-16/lib/cmake/mlir"
    echo "Using LLVM 16 from apt"
elif [ -d "/usr/local/llvm" ]; then
    LLVM_DIR="/usr/local/llvm/lib/cmake/llvm"
    MLIR_DIR="/usr/local/llvm/lib/cmake/mlir"
    echo "Using LLVM from source"
else
    echo "ERROR: LLVM not found. Please install:"
    echo "sudo apt install llvm-17 llvm-17-dev mlir-17-tools libmlir-17-dev"
    exit 1
fi

# Создание build директории
mkdir -p build
cd build

# Конфигурация CMake
cmake .. \
    -DLLVM_DIR=${LLVM_DIR} \
    -DMLIR_DIR=${MLIR_DIR} \
    -DCMAKE_BUILD_TYPE=Debug \
    "$@"

echo ""
echo "Configured with:"
echo "  LLVM_DIR: ${LLVM_DIR}"
echo "  MLIR_DIR: ${MLIR_DIR}"
