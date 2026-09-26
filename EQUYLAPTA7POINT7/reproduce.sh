#!/usr/bin/env bash
# reproduce.sh — One-command reproduction harness for EQUYLAPTA 7.7
# Usage:
#   ./reproduce.sh --mode smoke   (Fast deterministic validation < 5s)
#   ./reproduce.sh --mode full    (Full experimental execution ~2 mins)

set -e

MODE="smoke"
if [ "$1" == "--mode" ] && [ -n "$2" ]; then
    MODE="$2"
elif [ "$1" == "full" ]; then
    MODE="full"
fi

echo "=============================================================================="
echo "EQUYLAPTA 7.7 REPRODUCTION HARNESS [MODE: ${MODE^^}]"
echo "=============================================================================="

WORKSPACE="/home/user"
export PYTHONPATH="${WORKSPACE}/EQUYLAPTA7POINT7:${WORKSPACE}/ai-model-fusion-lab:${WORKSPACE}:${PYTHONPATH}"

if [ "$MODE" == "full" ]; then
    echo -e "\n[1/4] Running Master Experimental Pipeline..."
    python3 "${WORKSPACE}/EQUYLAPTA7POINT7/run_experiment.py"

    echo -e "\n[2/4] Running Unit Test Suite..."
    python3 -m unittest discover -s "${WORKSPACE}/EQUYLAPTA7POINT7/tests"

    echo -e "\n[3/4] Running Consistency Validator..."
    python3 "${WORKSPACE}/EQUYLAPTA7POINT7/validation/consistency_checker.py"

    echo -e "\n[4/4] Running Automated Quality Gate..."
    python3 "${WORKSPACE}/EQUYLAPTA7POINT7/validation/results_validator.py"

else
    echo -e "\n[1/4] Running Unit Test Suite..."
    python3 -m unittest discover -s "${WORKSPACE}/EQUYLAPTA7POINT7/tests"

    echo -e "\n[2/4] Verifying Gradient Objective Exactness..."
    python3 "${WORKSPACE}/EQUYLAPTA7POINT7/validation/gradient_check.py"

    echo -e "\n[3/4] Running Consistency Validator..."
    python3 "${WORKSPACE}/EQUYLAPTA7POINT7/validation/consistency_checker.py"

    echo -e "\n[4/4] Running Automated Quality Gate..."
    python3 "${WORKSPACE}/EQUYLAPTA7POINT7/validation/results_validator.py"
fi

echo -e "\nRegenerating Final Reports..."
python3 "${WORKSPACE}/EQUYLAPTA7POINT7/reports/generate_report.py"

echo "=============================================================================="
echo "EQUYLAPTA 7.7 REPRODUCTION COMPLETED SUCCESSFULLY!"
echo "Primary Report: ${WORKSPACE}/EQUYLAPTA_7.7_REPORT.md"
echo "Text Report:    ${WORKSPACE}/EQUYLAPTA_7.7_REPORT.txt"
echo "=============================================================================="
