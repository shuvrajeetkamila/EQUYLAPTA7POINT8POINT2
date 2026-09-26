#!/usr/bin/env bash
set -e

echo "================================================================================"
echo "REPRODUCING EQUYLAPTA 7.8.2 EXPERIMENTAL SUITE & EVIDENCE INTEGRITY"
echo "================================================================================"

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$WORKSPACE/EQUYLAPTA7POINT8_2" ]; then
    cd "$WORKSPACE/EQUYLAPTA7POINT8_2"
    WORKSPACE="$WORKSPACE/EQUYLAPTA7POINT8_2"
else
    cd "$WORKSPACE"
fi

export PYTHONPATH="$WORKSPACE:$WORKSPACE/..:$WORKSPACE/../ai-model-fusion-lab:$PYTHONPATH"

echo -e "\n[1/3] Executing Master Experimental Harness (run_7_8_2.py)..."
python3 run_7_8_2.py

echo -e "\n[2/3] Executing Automated Pytest Suite across all 8 verification test suites..."
python3 -m pytest tests/ -v

echo -e "\n[3/3] Executing Report Consistency Validator (validate_report.py)..."
python3 validate_report.py EQUYLAPTA7POINT8_2_REPORT.md results.json adapter_update_audit.json

echo -e "\n================================================================================"
echo "EQUYLAPTA 7.8.2 REPRODUCTION COMPLETED SUCCESSFULLY WITH 100% EVIDENCE INTEGRITY"
echo "================================================================================"
