param([string]$Repo = ".")
$ErrorActionPreference = "Stop"
$RepoPath=(Resolve-Path $Repo).Path
$Pack=Split-Path -Parent $PSScriptRoot
python "$Pack\paper_plotting\paper_style_plots.py" --threshold 0.1 --outdir "$RepoPath\paper_plots\swimmer_np" `
  --method "Multi" "$RepoPath\data_train\SwimmerMulti\run501" "$RepoPath\data_train\SwimmerMulti\run505" `
  --method "Combined" "$RepoPath\data_train\SwimmerCombined\run601" "$RepoPath\data_train\SwimmerCombined\run605"
Write-Host "For Quadrotor, pass your actual existing two Multi NP run folders plus QuadrotorCombined run201/run205 to paper_style_plots.py with threshold 0.5."
