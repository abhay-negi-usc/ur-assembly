#!/usr/bin/env bash
# Create + populate a venv for urlab (Linux/macOS, or Git Bash on Windows). Run from the repo root:
#
#   ./setup-venv.sh            # full hardware env: core + robot + perception (default)
#   ./setup-venv.sh core       # math + fusion + --dry-run only
#   ./setup-venv.sh dev        # core + perception + pytest (to run tests/)
#   ./setup-venv.sh robot      # core + arm/gripper (RTDE + Modbus)
#   ./setup-venv.sh perception # core + camera & ArUco
#
# Python 3.12 is required: ur_rtde and pyrealsense2 ship wheels only through cp312. SAM3 is NOT
# installed -- it runs from the sam3-abhay checkout under its own venv.
set -euo pipefail

layer="${1:-all}"
req="requirements/${layer}.txt"
[ -f "$req" ] || { echo "No such layer: $req (expected core/robot/perception/all/dev)" >&2; exit 1; }

# Prefer python3.12 explicitly; fall back to whatever python3 is if it is already 3.12.
if command -v python3.12 >/dev/null 2>&1; then
    py=python3.12
else
    py=python3
    echo "WARNING: python3.12 not found; using $($py --version). The robot/perception layers need" >&2
    echo "         Python <=3.12 for ur_rtde / pyrealsense2 wheels." >&2
fi

"$py" -m venv .venv
# Linux/macOS put activate in bin/; Git Bash on Windows puts it in Scripts/.
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r "$req"

echo
echo "Done. Activate later with:  source .venv/bin/activate   (or .venv/Scripts/activate on Git Bash)"
echo "Verify with:                python tests/test_smoke.py"
