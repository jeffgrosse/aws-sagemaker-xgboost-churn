#!/usr/bin/env bash
# Fetches the IBM Telco Customer Churn dataset (7,043 rows, 21 columns) fresh
# from IBM's own GitHub mirror. Not committed to this repo - see README.md's
# Dataset section for why.
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/raw"
SOURCE_URL="https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DEST="${DATA_DIR}/Telco-Customer-Churn.csv"

mkdir -p "${DATA_DIR}"

echo "Downloading Telco Customer Churn dataset..."
curl -fsSL "${SOURCE_URL}" -o "${DEST}"

ROWS=$(($(wc -l < "${DEST}") - 1))
echo "Saved to ${DEST} (${ROWS} data rows)."

if [ "${ROWS}" -ne 7043 ]; then
  echo "WARNING: expected 7043 rows, got ${ROWS} - upstream file may have changed." >&2
fi
