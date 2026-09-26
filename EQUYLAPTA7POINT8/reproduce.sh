#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/EQUYLAPTA7POINT8" ]; then
  PKG_DIR="$SCRIPT_DIR/EQUYLAPTA7POINT8"
else
  PKG_DIR="$SCRIPT_DIR"
fi
cd "$PKG_DIR"

MODE="full"
while [[ $# -gt 0 ]]; do
  case $1 in
    --mode)
      MODE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

echo "================================================================================"
echo "EQUYLAPTA 7.8 REPRODUCTION HARNESS"
echo "Mode: $MODE"
echo "Package Directory: $PKG_DIR"
echo "================================================================================"

if [ "$MODE" = "smoke" ]; then
  echo "[1/4] Running unit test suite (smoke mode)..."
  python3 -m unittest discover -s "$PKG_DIR/tests"

  echo "[2/4] Verifying analytical gradient check..."
  python3 "$PKG_DIR/validation/gradient_check.py"

  echo "[3/4] Running automated quality gate..."
  python3 "$PKG_DIR/validation/results_validator.py"

  echo "[4/4] Verifying report consistency..."
  python3 "$PKG_DIR/validation/report_consistency.py"

  echo "================================================================================"
  echo "SMOKE REPRODUCTION COMPLETED SUCCESSFULLY"
  echo "================================================================================"
else
  echo "[1/2] Executing full EQUYLAPTA 7.8 experimental pipeline..."
  python3 "$PKG_DIR/run_experiment.py"

  echo "[2/2] Running automated quality gate..."
  python3 "$PKG_DIR/validation/results_validator.py"

  echo "================================================================================"
  echo "FULL REPRODUCTION COMPLETED SUCCESSFULLY"
  echo "================================================================================"
fi
