#!/bin/bash
# Helper to find pandas usage for potential hipDF porting
TARGET_DIR=${1:-"."}
echo "Scanning for pandas imports in $TARGET_DIR..."
grep -rn "import pandas as pd" "$TARGET_DIR" || echo "No pandas imports found."
