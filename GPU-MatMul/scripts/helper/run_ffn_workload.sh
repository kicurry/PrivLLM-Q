#!/bin/bash

# Generic helper script to run FFN workloads.
# It receives all parameters, runs tests, and summarizes the results.

# --- Path Agnostic Setup ---
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." &> /dev/null && pwd)

# --- Argument Validation ---
# This script MUST receive exactly FOUR arguments from the main script.
if [ "$#" -ne 4 ]; then
    echo "Internal Error: Helper script received an incorrect number of arguments ($# instead of 4)."
    echo "Usage (internal helper): $0 <executable_path> <poly_degree> <plain_width> <rns_moduli>"
    exit 1
fi

EXECUTABLE_REL_PATH="$1"
POLY_DEGREE="$2"
PLAIN_WIDTH="$3"
RNS_MODULI="$4"
EXECUTABLE="${PROJECT_ROOT}/${EXECUTABLE_REL_PATH}"

# Check if executable exists and is executable
if [ ! -x "$EXECUTABLE" ]; then
    echo "Error: Executable '$EXECUTABLE' not found or not executable!"
    exit 1
fi

# --- Test Configurations ---
declare -a TEST_CONFIGS=(
    "128,4096,4096"
    "128,4096,12288"
    "128,12288,4096"
    "128,4096,16384"
    "128,16384,4096"
)

declare -A TIME_RESULTS
declare -A NOISE_RESULTS

echo "------------------------------------------------------"
echo "Running FFN Workload..."
echo "  - Using Params: PolyDegree=${POLY_DEGREE}, PlainWidth=${PLAIN_WIDTH}, RNS=${RNS_MODULI}"
echo "------------------------------------------------------"

for config in "${TEST_CONFIGS[@]}"; do
    echo -n "Testing FFN dim: $config ... "
    
    COMMAND="$EXECUTABLE -nn --poly-degree $POLY_DEGREE --plain-width $PLAIN_WIDTH --rns-moduli $RNS_MODULI -ffn $config"
    OUTPUT=$($COMMAND 2>&1)
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "FAILED!"
        echo "Error running test for configuration $config. Output:"
        echo "$OUTPUT"
        exit $EXIT_CODE
    fi
    
    NOISE_BUDGET=$(echo "$OUTPUT" | grep -oP "\+ Noise budget after PCMM: \K\d+" || echo "N/A")
    NOISE_RESULTS["$config"]="$NOISE_BUDGET bits"

    MATMUL_TIME=$(echo "$OUTPUT" | grep -oP "Ciphertext activation-Plaintext weight matrix multiplication - Complete. Time: \K\d+" || echo "N/A")
    TIME_RESULTS["$config"]="$MATMUL_TIME ms"
    
    echo "Done."
done

echo ""
echo "======================================================"
echo "Test Summary"
echo "======================================================"
printf "%-20s | %-20s | %-20s\n" "FFN Dimension" "MatMul Time" "Noise Budget"
printf "%s\n" "-----------------------------------------------------------------"
for config in "${TEST_CONFIGS[@]}"; do
    printf "%-20s | %-20s | %-20s\n" "$config" "${TIME_RESULTS[$config]}" "${NOISE_RESULTS[$config]}"
done
echo "======================================================"
