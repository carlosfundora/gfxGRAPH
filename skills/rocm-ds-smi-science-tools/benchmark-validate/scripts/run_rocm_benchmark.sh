#!/usr/bin/env bash
set -euo pipefail

# run_rocm_benchmark.sh
# A helper script to time and compare a CPU baseline script vs a ROCm script.

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <cpu_script.py> <rocm_script.py>"
  echo "Runs both scripts, times them, and saves their outputs for comparison."
  exit 1
fi

CPU_SCRIPT="$1"
ROCM_SCRIPT="$2"

echo "=== ROCm-DS Benchmark Wrapper ==="
echo "CPU Baseline:  ${CPU_SCRIPT}"
echo "ROCM Script:   ${ROCM_SCRIPT}"
echo "---------------------------------"

# Run CPU Baseline
echo "[1/2] Running CPU Baseline..."
CPU_START=$(date +%s.%N)
python3 "${CPU_SCRIPT}" > cpu_output.txt 2> cpu_error.txt || echo "CPU Script returned non-zero exit code"
CPU_END=$(date +%s.%N)
CPU_TIME=$(echo "$CPU_END - $CPU_START" | bc)
echo "CPU Execution Time: ${CPU_TIME} seconds"

# Run ROCm Script
echo ""
echo "[2/2] Running ROCm Script..."
ROCM_START=$(date +%s.%N)
python3 "${ROCM_SCRIPT}" > rocm_output.txt 2> rocm_error.txt || echo "ROCm Script returned non-zero exit code"
ROCM_END=$(date +%s.%N)
ROCM_TIME=$(echo "$ROCM_END - $ROCM_START" | bc)
echo "ROCm Execution Time: ${ROCM_TIME} seconds"

echo "---------------------------------"
echo "Benchmark Complete."
echo "CPU Output saved to: cpu_output.txt (Errors: cpu_error.txt)"
echo "ROCm Output saved to: rocm_output.txt (Errors: rocm_error.txt)"
echo ""
echo "To verify parity, manually compare the outputs or run a parity testing script."
