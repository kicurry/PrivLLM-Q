#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Path Agnostic Setup ---
# Get the real directory of the script, resolving any symlinks.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# The project root is one level above the script directory.
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." &> /dev/null && pwd)

# --- Default Configuration ---
# Phantom-FHE build options
FP64_MM_ARITH_OPT="OFF"
RNS_POLY_BATCH_OPT="ON"

# Main project build options
ENABLE_VALIDATION_OPT="OFF"

# --- Function Definitions ---
# Display script usage
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "This script builds the PrivLLM-Q-MatMul project and its dependencies."
    echo "It can be run from any directory."
    echo ""
    echo "Options:"
    echo "  --fp64                Enable FP64 arithmetic for modular multiplication in phantom-fhe."
    echo "                        (Sets -DFP64_MM_ARITH=ON, default: OFF)"
    echo "  --no-rns-batch        Disable RNS polynomial batch processing in phantom-fhe."
    echo "                        (Sets -DRNS_POLY_BATCH=OFF, default: ON)"
    echo "  --enable-validation   Enable matrix multiplication validation in the main project."
    echo "                        (Sets -DENABLE_VALIDATION=ON, default: OFF)"
    # [ADDED] Help text for the new performance option.
    echo "  --perf                Performance profile: Enables --fp64, enables RNS batch, disables validation."
    echo "  -h, --help            Display this help message and exit."
    echo ""
    echo "Example:"
    echo "  # Build with default settings from project root"
    echo "  ./scripts/build.sh"
    echo ""
    echo "  # Build with performance profile"
    echo "  ./scripts/build.sh --perf"
    echo ""
    echo "  # Build with validation enabled and phantom-fhe using FP64"
    echo "  ./scripts/build.sh --enable-validation --fp64"
}

# --- Argument Parsing ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --fp64)
            FP64_MM_ARITH_OPT="ON"
            shift
            ;;
        --no-rns-batch)
            RNS_POLY_BATCH_OPT="OFF"
            shift
            ;;
        --enable-validation)
            ENABLE_VALIDATION_OPT="ON"
            shift
            ;;
        # [ADDED] Case for the --perf option.
        # It sets the three configuration variables to the desired values for performance testing.
        --perf)
            FP64_MM_ARITH_OPT="ON"
            RNS_POLY_BATCH_OPT="ON"
            ENABLE_VALIDATION_OPT="OFF"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage
            exit 1
            ;;
    esac
done

# --- Dynamically Generate Build Directory Name ---
BUILD_SUFFIX="_fp64_$(echo "$FP64_MM_ARITH_OPT" | tr '[:upper:]' '[:lower:]')"
BUILD_SUFFIX+="_rnsbatch_$(echo "$RNS_POLY_BATCH_OPT" | tr '[:upper:]' '[:lower:]')"
BUILD_SUFFIX+="_validation_$(echo "$ENABLE_VALIDATION_OPT" | tr '[:upper:]' '[:lower:]')"
BUILD_DIR="build${BUILD_SUFFIX}"

# --- Main Script Body ---

echo "========================================"
echo "Project Root: ${PROJECT_ROOT}"
echo "Build Directory: ${BUILD_DIR}"
echo "========================================"

# Determine the number of cores for parallel compilation
if command -v nproc &> /dev/null; then
    NUM_CORES=$(nproc)
elif command -v sysctl &> /dev/null; then
    NUM_CORES=$(sysctl -n hw.ncpu)
else
    NUM_CORES=4
    echo "Warning: Cannot determine number of cores. Defaulting to ${NUM_CORES}."
fi
echo "Using ${NUM_CORES} cores for parallel build."

# Step 1: Initialize and update submodules
echo ""
echo "========================================"
echo "Step 1: Initializing and updating submodules..."
echo "========================================"
git -C "${PROJECT_ROOT}" submodule update --init --recursive

# Step 2: Apply patch to deps/phantom-fhe
echo ""
echo "========================================"
echo "Step 2: Applying patch to deps/phantom-fhe..."
echo "========================================"
PATCH_FILE="${PROJECT_ROOT}/deps/patch/phantom.patch"
PHANTOM_DIR="${PROJECT_ROOT}/deps/phantom-fhe"

if git -C "${PHANTOM_DIR}" apply --reverse --check "${PATCH_FILE}" &> /dev/null; then
    echo "Patch seems to be already applied. Skipping."
else
    echo "Applying ${PATCH_FILE}..."
    git -C "${PHANTOM_DIR}" apply "${PATCH_FILE}"
    echo "Patch applied successfully."
fi

# Step 3: Build and install phantom-fhe
echo ""
echo "========================================"
echo "Step 3: Building and installing phantom-fhe"
echo "========================================"
(
    cd "${PHANTOM_DIR}"
    
    if [ -d "build" ]; then
        echo "Removing old phantom-fhe build directory..."
        rm -rf build
    fi
    if [ -d "build_phantom" ]; then
        echo "Removing old phantom-fhe install directory..."
        rm -rf build_phantom
    fi
    
    echo "Configuring phantom-fhe with:"
    echo "  - FP64_MM_ARITH      : ${FP64_MM_ARITH_OPT}"
    echo "  - RNS_POLY_BATCH     : ${RNS_POLY_BATCH_OPT}"

    cmake -S . -B build \
          -DCMAKE_INSTALL_PREFIX=build_phantom \
          -DFP64_MM_ARITH=${FP64_MM_ARITH_OPT} \
          -DRNS_POLY_BATCH=${RNS_POLY_BATCH_OPT}

    echo "Building and installing phantom-fhe (target: install)..."
    cmake --build build --target install -j${NUM_CORES}
)
echo "phantom-fhe build complete."


# Step 4: Build the main project (PrivLLM-Q-MatMul)
echo ""
echo "========================================"
echo "Step 4: Building the main project (PrivLLM-Q-MatMul)"
echo "========================================"

if [ -d "${PROJECT_ROOT}/${BUILD_DIR}" ]; then
    echo "Removing old project build directory: ${BUILD_DIR}"
    rm -rf "${PROJECT_ROOT}/${BUILD_DIR}"
fi

echo "Configuring PrivLLM-Q-MatMul with:"
echo "  - ENABLE_VALIDATION  : ${ENABLE_VALIDATION_OPT}"

cmake -S "${PROJECT_ROOT}" -B "${PROJECT_ROOT}/${BUILD_DIR}" \
      -DENABLE_VALIDATION=${ENABLE_VALIDATION_OPT}

echo "Building PrivLLM-Q-MatMul..."
cmake --build "${PROJECT_ROOT}/${BUILD_DIR}" -j${NUM_CORES}

echo ""
echo "========================================"
echo "Build finished successfully!"
echo "The final executable(s) can be found in the '${BUILD_DIR}' directory."
echo "========================================"
