param([string]$Repo = ".", [double]$SigmaViscosity = 0.7)
$ErrorActionPreference = "Stop"
$RepoPath=(Resolve-Path $Repo).Path
$Python=(Get-Command python).Source
$Pack=Split-Path -Parent $PSScriptRoot
python "$Pack\tools\preflight.py" --repo "$RepoPath" --require-swimmer
New-Item -ItemType Directory -Force -Path "$RepoPath\logs" | Out-Null
$worker = {
  param($r,$p,$run,$seed,$sigma,$log)
  Set-Location $r
  & $p ".\code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint_perturbed.py" `
    --run $run --seed $seed --env SwimmerWithPos --persistent_eps 0.1 --beta 30000 --K_epochs 5 `
    --max_train_steps 16000 --warm_start_episode 150 --lr_cost 1e-3 --entropy_coef 0.001 `
    --weight_reg 0.001 --sigma_viscosity $sigma --cost_dim 2 --cost_scale 1.0 `
    2>&1 | Tee-Object -FilePath $log
}
$jobs=@()
$jobs += Start-Job -Name "swimmer_multi_pr_seed2" -ScriptBlock $worker -ArgumentList $RepoPath,$Python,701,2,$SigmaViscosity,"$RepoPath\logs\swimmer_multi_pr_seed2.log"
$jobs += Start-Job -Name "swimmer_multi_pr_seed5" -ScriptBlock $worker -ArgumentList $RepoPath,$Python,705,5,$SigmaViscosity,"$RepoPath\logs\swimmer_multi_pr_seed5.log"
$jobs | Wait-Job | Out-Null
$jobs | Receive-Job
$jobs | Remove-Job
