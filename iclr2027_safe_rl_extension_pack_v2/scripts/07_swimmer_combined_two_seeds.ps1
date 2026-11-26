param([string]$Repo = ".")
$ErrorActionPreference = "Stop"
$RepoPath=(Resolve-Path $Repo).Path
$Python=(Get-Command python).Source
$Pack=Split-Path -Parent $PSScriptRoot
python "$Pack\tools\preflight.py" --repo "$RepoPath" --require-swimmer
New-Item -ItemType Directory -Force -Path "$RepoPath\logs" | Out-Null
$worker = {
  param($r,$p,$run,$seed,$log)
  Set-Location $r
  & $p ".\code_ipm_rcmdp_rcrl_max_swimmer_combined.py" `
    --run $run --seed $seed --env SwimmerWithPos --persistent_eps 0.1 --beta 30000 --K_epochs 5 `
    --max_train_steps 16000 --warm_start_episode 150 --lr_cost 1e-3 --entropy_coef 0.001 `
    --weight_reg 0.0 --sigma_viscosity 0.0 2>&1 | Tee-Object -FilePath $log
}
$jobs=@()
$jobs += Start-Job -Name "swimmer_combined_seed2" -ScriptBlock $worker -ArgumentList $RepoPath,$Python,601,2,"$RepoPath\logs\swimmer_combined_seed2.log"
$jobs += Start-Job -Name "swimmer_combined_seed5" -ScriptBlock $worker -ArgumentList $RepoPath,$Python,605,5,"$RepoPath\logs\swimmer_combined_seed5.log"
$jobs | Wait-Job | Out-Null
$jobs | Receive-Job
$jobs | Remove-Job
