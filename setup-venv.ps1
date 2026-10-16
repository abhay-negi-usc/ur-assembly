# Create + populate a venv for urlab (Windows PowerShell). Run from the repo root:
#
#   .\setup-venv.ps1           # full hardware env: core + robot + perception (default)
#   .\setup-venv.ps1 core      # math + fusion + --dry-run only
#   .\setup-venv.ps1 dev       # core + perception + pytest (to run tests/)
#   .\setup-venv.ps1 robot     # core + arm/gripper (RTDE + Modbus)
#   .\setup-venv.ps1 perception # core + camera & ArUco
#
# Python 3.12 is required: ur_rtde and pyrealsense2 ship wheels only through cp312. If `py -3.12`
# is missing, install 3.12 from python.org first. SAM3 is NOT installed -- it runs from the
# sam3-abhay checkout under its own venv.
param([string]$Layer = "all")
$ErrorActionPreference = "Stop"

$req = "requirements\$Layer.txt"
if (-not (Test-Path $req)) {
    Write-Error "No such layer: $req  (expected one of core/robot/perception/all/dev)"
}

py -3.12 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r $req

Write-Host ""
Write-Host "Done. Activate later with:  .\.venv\Scripts\Activate.ps1"
Write-Host "Verify with:                python tests\test_smoke.py"

# WHERE BARE CONFIG NAMES RESOLVE -- printed because nothing else at install time would say it.
Write-Host ""
if ($env:URLAB_CONFIG_DIR) {
    Write-Host "Config folder: URLAB_CONFIG_DIR is set -> bare config names resolve in $env:URLAB_CONFIG_DIR"
} else {
    Write-Host "Config folder: bare config names (--config <name>) resolve in $((Get-Location).Path)\configs -- this repo's (UR10e) configs."
}
Write-Host "  Importing urlab from ANOTHER project? Point it at that project's configs folder:"
Write-Host '      $env:URLAB_CONFIG_DIR = "C:\path\to\other-project\configs"'
Write-Host "  Unset, that project would get THIS repo's configs (robot.ip = the UR10e)."
Write-Host "  Details: README.md -> 'Using urlab from another project'."
