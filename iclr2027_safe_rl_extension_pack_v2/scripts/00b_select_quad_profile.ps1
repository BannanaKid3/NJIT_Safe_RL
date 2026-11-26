param(
    [string]$Repo = ".",
    [Parameter(Mandatory=$true)][string]$Profile
)
$ErrorActionPreference = "Stop"
$RepoPath = (Resolve-Path $Repo).Path
$Pack = Split-Path -Parent $PSScriptRoot
python "$Pack\tools\select_quadrotor_profile.py" --repo "$RepoPath" --profile $Profile
python "$Pack\tools\preflight.py" --repo "$RepoPath"
