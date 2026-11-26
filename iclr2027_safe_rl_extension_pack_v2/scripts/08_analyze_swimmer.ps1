param([string]$Repo = ".")
$ErrorActionPreference = "Stop"
$RepoPath=(Resolve-Path $Repo).Path
$Pack=Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force -Path "$RepoPath\analysis_results" | Out-Null
python "$Pack\analysis\analyze_two_seed_method.py" --run "$RepoPath\data_train\SwimmerMulti\run501" --run "$RepoPath\data_train\SwimmerMulti\run505" --threshold 0.1 --label "Swimmer Multi NP" --out "$RepoPath\analysis_results\swimmer_multi_np.json"
python "$Pack\analysis\analyze_two_seed_method.py" --run "$RepoPath\data_train\SwimmerCombined\run601" --run "$RepoPath\data_train\SwimmerCombined\run605" --threshold 0.1 --label "Swimmer Combined NP" --out "$RepoPath\analysis_results\swimmer_combined_np.json"
python "$Pack\analysis\compare_multi_vs_combined.py" --multi-run "$RepoPath\data_train\SwimmerMulti\run501" --multi-run "$RepoPath\data_train\SwimmerMulti\run505" --combined-run "$RepoPath\data_train\SwimmerCombined\run601" --combined-run "$RepoPath\data_train\SwimmerCombined\run605" --threshold 0.1 --out "$RepoPath\analysis_results\swimmer_multi_vs_combined_np.json"
