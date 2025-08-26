# PrivLLM-Q: Build and Test Instructions

This document provides instructions for reviewers to build the project and reproduce the performance results presented in our paper. We have provided convenient scripts to automate the entire process.

## Prerequisites

Before you begin, please ensure you have the following software installed:

*   A modern C++ compiler (e.g., g++ 9 or newer, clang++ 10 or newer)
*   CMake (version 3.10 or newer)
*   Git (version 1.8.5 or newer)

We have tested the build process on Ubuntu 20.04.

## 1. Building the Project

The build process, including fetching dependencies, patching, and compilation, is handled by a single script.

### Step 1: Clone the Repository

First, clone the repository and all its submodules using the `--recursive` flag. Run this command from the toplevel of the working tree:

```sh
git submodule update --init --recursive
```

### Step 2: Use the Build Script

We provide a powerful build script located at `scripts/build.sh`. This script handles all dependencies (`SEAL`, `phantom-fhe`) and compiles the project.

#### A) Performance-Optimized Build (Recommended)

To reproduce the performance results from our paper, please use the `--perf` flag. This creates a release build with all performance-related compiler flags enabled for our dependencies.

```sh
# Grant execution permissions to the script first
chmod +x scripts/build.sh

# Run the performance build
./scripts/build.sh --perf
```

This command will:
1.  Initialize and update all submodules.
2.  Apply a necessary patch to the `phantom-fhe` library.
3.  Build `phantom-fhe` with performance optimizations (`FP64_MM_ARITH=ON`, `RNS_POLY_BATCH=ON`).
4.  Build the main project in **Release** mode with validation disabled.

The build artifacts will be located in a directory named `build_fp64_on_rnsbatch_on_validation_off/`.

#### B) Other Build Options

The build script supports other configurations. For a full list of options, you can run:

```sh
./scripts/build.sh --help
```

## 2. Running Performance Tests

After successfully building the project, you can use the `run_perf_tests.sh` script to execute the benchmark workloads.

### Test Script Usage

The script takes two arguments: a **profile** and the **path to the executable**.

```sh
./scripts/run_perf_tests.sh <profile> <path_to_executable>
```

*   **`<profile>`**: Determines the runtime parameters for the test.
    *   `noquant`: Represents the **non-quantized baseline**. It uses a larger plaintext bit-width (`plainWidth=60`) and RNS components {40,40,40,40,40}, resulting in higher precision but slower performance.
    *   `privllm-q`: Represents **our proposed optimized solution**. It uses a smaller, quantization-aware plaintext bit-width (`plainWidth=20`) and RNS components {34,34,34}, leading to significant performance gains.

*   **`<path_to_executable>`**: The relative path to the test program you want to run. You can easily find these binaries in `build_fp64_on_rnsbatch_on_validation_off/fastFFN`.


---

### **Mapping Executables to Paper Contributions**

To help reviewers connect our experimental results with the contributions described in the paper, this section explains what each key executable in the `fastFFN/` directory represents.

*   `test_ffn_baseline`
    *   **Corresponds to: Baseline**
    *   This executable represents our **Baseline** implementation. It improves upon the native `phantom` library by pre-transforming all ciphertext-plaintext multiplication inputs into the NTT domain. In the context of our paper, this corresponds to the baseline large-scale MatMul computation flow that reuses the results of the Baby-Step Rotation (BS-HRot) stage.

*   `test_ffn_gsd` & `test_ffn_gsd_nobsgs`
    *   **Corresponds to: Giant-Step Deferral (GSD)**
    *   These executables implement the **Giant-Step Deferral (GSD)** optimization. The core idea is to reduce the number of costly HRot by merging the Giant-Step Rotation (GS-HRot) calculations from different input tiles that contribute to the same output tile.
    *   The `test_ffn_gsd_nobsgs` variant is a special case where the 'giant-step' size is set to 1. This program is crucial as it isolates the structural benefits of the 'less-NTT' approach. **We recommend using `test_ffn_gsd` for evaluating the core GSD improvement.**

*   `test_ffn_gsd_falr`
    *   **Corresponds to: GSD + Fused HE-MAC (FALR)**
    *   This executable incorporates the **Fused HE-MAC Reduction (FALR)** optimization on top of GSD. It further boosts performance by fusing all Homomorphic Multiply-Accumulate (HE-MAC) operations into a single, large computational kernel.

*   **A Note on HIFA Optimization**
    *   The **Hybrid INT32-FP64 Arithmetic (HIFA)** optimization is a hardware-level enhancement that is not tied to a specific executable. Instead, it is a **compile-time option** controlled by the `--fp64` flag in the `build.sh` script and is automatically enabled by the recommended `--perf` build. It accelerates the underlying arithmetic for **all** executables when the project is compiled with it.

---

### Walkthrough: Reproducing Performance Results

Here is a step-by-step guide to reproduce the key performance figures.

**Step 1: Build the project for performance (if you haven't already).**

```sh
./scripts/build.sh --perf
```

**Step 2: Run the tests.**

We recommend using `test_ffn_gsd` as a representative example of our matrix multiplication algorithm.

#### A) Test the Baseline (`noquant`) Performance

Run the following command. You can use the `Tab` key to auto-complete the long executable path after typing `build_...`:

```sh
# Grant execution permissions to the test script
chmod +x scripts/run_perf_tests.sh
chmod +x scripts/helper/run_ffn_workload.sh

# Run the test
./scripts/run_perf_tests.sh noquant build_fp64_on_rnsbatch_on_validation_off/fastFFN/test_ffn_gsd
```

#### B) Test Our Optimized (`privllm-q`) Performance

Now, run the same test but with the `privllm-q` profile. Notice that we use the *exact same executable*; the performance difference comes from the change in runtime parameters managed by the profile.

```sh
./scripts/run_perf_tests.sh privllm-q build_fp64_on_rnsbatch_on_validation_off/fastFFN/test_ffn_gsd
```

### Understanding the Output

The script will execute the test for several matrix dimensions and produce a summary table at the end, which looks like this:

```
======================================================
Test Summary
======================================================
FFN Dimension        | MatMul Time          | Noise Budget
-----------------------------------------------------------------
128,4096,4096        | xxx ms              | x bits
128,4096,12288       | xxx ms              | x bits
...
======================================================
```

By comparing the `MatMul Time` column from the `noquant` and `privllm-q` runs, you can directly observe the performance improvement of our method.

**A Note on `Noise Budget` column**: *N/A bits* when using `run_perf_tests.sh` or `run_ffn_workload.sh`
