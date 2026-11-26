param(
    [string]$Repo = ".",
    [string]$ExistingMultiRunSeed2 = "",
    [string]$ExistingMultiRunSeed5 = ""
)
$ErrorActionPreference = "Stop"
$RepoPath = (Resolve-Path $Repo).Path
$Pack = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force -Path "$RepoPath\analysis_results" | Out-Null
python "$Pack\analysis\analyze_two_seed_method.py" `
  --run "$RepoPath\data_train\QuadrotorCombined\run201" `
  --run "$RepoPath\data_train\QuadrotorCombined\run205" `
  --threshold 0.5 --label "Quadrotor Combined NP" `
  --out "$RepoPath\analysis_results\quad_combined_np.json"
if ($ExistingMultiRunSeed2 -and $ExistingMultiRunSeed5) {
  python "$Pack\analysis\compare_multi_vs_combined.py" `
    --multi-run "$ExistingMultiRunSeed2" --multi-run "$ExistingMultiRunSeed5" `
    --combined-run "$RepoPath\data_train\QuadrotorCombined\run201" `
    --combined-run "$RepoPath\data_train\QuadrotorCombined\run205" `
    --threshold 0.5 --out "$RepoPath\analysis_results\quad_multi_vs_combined_np.json"
} else {
  Write-Host "Combined analyzed. To compare against existing Multi NP, rerun this script with -ExistingMultiRunSeed2 and -ExistingMultiRunSeed5."
}
