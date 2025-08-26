# Artifact for USENIX Security 2026 Submission PrivLLM-Q

This repository provides the artifact for our USENIX Security 2026 paper.  


## Overview

We release two main modules in this repository:

1. **Quantization** (`Quantization/`)  
   - Includes scripts and configurations for reproducing the accuracy results from the paper.  
   - Supports training, inference, and evaluation with different bit-width settings.  

2. **GPU-Accelerated CT-PT MatMul** (`GPU-MatMul/`)  
   - Provides our optimized GPU implementation of low-bitwidth ciphertext–plaintext matrix multiplication.   
   - Includes benchmarks and example usage scripts.

Each directory contains its own **detailed usage guide** (see the `README.md` inside each folder).
