param(
    [string]$Repo = ".",
    [string]$QuadProfile = "white_noise_0p01"
)
$ErrorActionPreference = "Stop"
$RepoPath = (Resolve-Path $Repo).Path
$Pack = Split-Path -Parent $PSScriptRoot
python "$Pack\setup_pipeline.py" --repo "$RepoPath" --quad-profile $QuadProfile
python "$Pack\tools\preflight.py" --repo "$RepoPath"
Write-Host "Installed overlay into $RepoPath"
Write-Host "Quad P profile: $QuadProfile"
