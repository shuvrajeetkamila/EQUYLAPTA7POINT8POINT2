#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/EQUYLAPTA7POINT8_1" ]; then
  PKG_DIR="$SCRIPT_DIR/EQUYLAPTA7POINT8_1"
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
echo "EQUYLAPTA 7.8.1 REPRODUCTION HARNESS"
echo "Mode: $MODE"
echo "Package Directory: $PKG_DIR"
echo "================================================================================"

if [ "$MODE" = "smoke" ]; then
  echo "[1/3] Running unit tests..."
  python3 -m unittest discover -s "$PKG_DIR/tests"

  echo "[2/3] Verifying training gradient against finite differences..."
  python3 "$PKG_DIR/src/gradient_validation.py"

  echo "[3/3] Validating report consistency..."
  python3 "$PKG_DIR/validate_report.py"

  echo "================================================================================"
  echo "SMOKE REPRODUCTION COMPLETED SUCCESSFULLY"
  echo "================================================================================"
else
  echo "[1/3] Running full EQUYLAPTA 7.8.1 experimental pipeline..."
  python3 "$PKG_DIR/run_7_8_1.py"

  echo "[2/3] Running unit test suite..."
  python3 -m unittest discover -s "$PKG_DIR/tests"

  echo "[3/3] Validating report consistency..."
  python3 "$PKG_DIR/validate_report.py"

  echo "================================================================================"
  echo "FULL REPRODUCTION COMPLETED SUCCESSFULLY"
  echo "================================================================================"
fi
