#!/bin/bash

# Main performance test runner.
# It takes a profile (affecting runtime parameters) and a direct path
# to the executable, making it easy to use with shell tab-completion.

set -e

# --- Path Agnostic Setup ---
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." &> /dev/null && pwd)

# --- Function Definitions ---
usage() {
    echo "Usage: $0 <profile> <path_to_executable>"
    echo ""
    echo "Arguments:"
    echo "  <profile>             Test profile to run. Affects runtime parameters."
    echo "                        Available: noquant, privllm-q"
    echo ""
    echo "  <path_to_executable>  Relative path from the project root to the executable."
    echo ""
    echo "Example (using tab-completion to find the executable):"
    echo "  $0 noquant build_fp64_on_rnsbatch_on_validation_off/fastFFN/test_ffn_less_ntt_nobsgs"
    echo ""
    echo "  $0 privllm-q build_fp64_on_rnsbatch_on_validation_off/fastFFN/test_ffn_ntt_pmult"
}

# --- Argument Validation ---
if [ "$#" -ne 2 ]; then
    echo "Error: Exactly two arguments are required."
    usage
    exit 1
fi

PROFILE="$1"
EXECUTABLE_REL_PATH="$2"
FULL_EXECUTABLE_PATH="${PROJECT_ROOT}/${EXECUTABLE_REL_PATH}"

# Check if the provided executable path is valid
if [ ! -f "${FULL_EXECUTABLE_PATH}" ]; then
    echo "Error: Executable file not found at '${EXECUTABLE_REL_PATH}'"
    echo "Please ensure the path is correct and relative to the project root."
    exit 1
fi
if [ ! -x "${FULL_EXECUTABLE_PATH}" ]; then
    echo "Error: File '${EXECUTABLE_REL_PATH}' is not executable. Please check permissions."
    exit 1
fi

# --- Set parameters based on profile ---
POLY_DEGREE=8192

case "$PROFILE" in
    noquant)
        PLAIN_WIDTH=60
        RNS_MODULI="40,40,40,40,40"
        ;;
    privllm-q)
        # Using a smaller plainWidth to simulate quantization
        PLAIN_WIDTH=20
        RNS_MODULI="34,34,34"
        ;;
    *)
        echo "Error: Unknown profile '$PROFILE'. Available: noquant, privllm-q"
        usage
        exit 1
        ;;
esac

echo "======================================================"
echo "Starting Performance Test"
echo "======================================================"
echo "  - Profile:      ${PROFILE}"
echo "  - Executable:   ${EXECUTABLE_REL_PATH}"
echo ""

# --- Execute the test ---
HELPER_SCRIPT="${SCRIPT_DIR}/helper/run_ffn_workload.sh"
bash "${HELPER_SCRIPT}" "${EXECUTABLE_REL_PATH}" "${POLY_DEGREE}" "${PLAIN_WIDTH}" "${RNS_MODULI}"

echo ""
echo "Profile '$PROFILE' test for '${EXECUTABLE_REL_PATH}' completed."
echo "======================================================"
