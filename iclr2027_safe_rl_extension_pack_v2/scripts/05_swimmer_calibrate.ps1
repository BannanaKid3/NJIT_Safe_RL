param(
    [string]$Repo = ".",
    [string]$ModelPrefix = "",
    [int]$Episodes = 30
)
$ErrorActionPreference = "Stop"
$RepoPath = (Resolve-Path $Repo).Path
$Pack = Split-Path -Parent $PSScriptRoot
if ($ModelPrefix) {
  python "$Pack\calibration\calibrate_swimmer_joint_speed.py" --repo "$RepoPath" --model-prefix "$ModelPrefix" --episodes $Episodes
} else {
  python "$Pack\calibration\calibrate_swimmer_joint_speed.py" --repo "$RepoPath" --episodes $Episodes
}
python "$Pack\tools\preflight.py" --repo "$RepoPath" --require-swimmer
