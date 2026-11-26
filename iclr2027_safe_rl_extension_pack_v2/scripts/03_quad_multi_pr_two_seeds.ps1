param([string]$Repo = ".")
$ErrorActionPreference = "Stop"
$RepoPath = (Resolve-Path $Repo).Path
$Python = (Get-Command python).Source
$Pack = Split-Path -Parent $PSScriptRoot
python "$Pack\tools\preflight.py" --repo "$RepoPath"
New-Item -ItemType Directory -Force -Path "$RepoPath\logs" | Out-Null
$worker = {
  param($r,$p,$run,$seed,$log)
  Set-Location $r
  & $p ".\code_ipm_rcmdp_rcrl_max_quadrotor_multi_constraint_perturbed.py" `
    --run $run --seed $seed --persistent_eps 0.5 --beta 30000 --K_epochs 5 `
    --max_train_steps 40000 --warm_start_episode 1000 --lr_cost 5e-4 --lr_c 5e-4 --lr_a 5e-4 `
    --entropy_coef 0.007 --weight_reg 0.001 --batch_size 1024 --mini_batch_size 256 `
    --cost_scale 100.0 --cost_dim 4 2>&1 | Tee-Object -FilePath $log
}
$jobs=@()
$jobs += Start-Job -Name "quad_multi_pr_seed2" -ScriptBlock $worker -ArgumentList $RepoPath,$Python,301,2,"$RepoPath\logs\quad_multi_pr_seed2.log"
$jobs += Start-Job -Name "quad_multi_pr_seed5" -ScriptBlock $worker -ArgumentList $RepoPath,$Python,305,5,"$RepoPath\logs\quad_multi_pr_seed5.log"
Write-Host "Running Quadrotor Multi P+R. The trainer will abort if env.disturbances is empty."
$jobs | Wait-Job | Out-Null
$jobs | Receive-Job
$jobs | Remove-Job
