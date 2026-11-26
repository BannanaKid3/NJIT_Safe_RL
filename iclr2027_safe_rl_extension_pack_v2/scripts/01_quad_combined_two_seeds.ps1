param([string]$Repo = ".")
$ErrorActionPreference = "Stop"
$RepoPath = (Resolve-Path $Repo).Path
$Python = (Get-Command python).Source
New-Item -ItemType Directory -Force -Path "$RepoPath\logs" | Out-Null
$worker = {
  param($r,$p,$run,$seed,$log)
  Set-Location $r
  & $p ".\code_ipm_rcmdp_rcrl_max_quadrotor_combined.py" `
    --run $run --seed $seed --persistent_eps 0.5 --beta 30000 --K_epochs 5 `
    --max_train_steps 40000 --warm_start_episode 1000 --lr_cost 5e-4 --lr_c 5e-4 --lr_a 5e-4 `
    --entropy_coef 0.007 --weight_reg 0.0 --batch_size 1024 --mini_batch_size 256 --cost_scale 100.0 `
    2>&1 | Tee-Object -FilePath $log
}
$jobs=@()
$jobs += Start-Job -Name "quad_combined_seed2" -ScriptBlock $worker -ArgumentList $RepoPath,$Python,201,2,"$RepoPath\logs\quad_combined_seed2.log"
$jobs += Start-Job -Name "quad_combined_seed5" -ScriptBlock $worker -ArgumentList $RepoPath,$Python,205,5,"$RepoPath\logs\quad_combined_seed5.log"
Write-Host "Running Quadrotor Combined seeds 2 and 5 in parallel..."
$jobs | Wait-Job | Out-Null
$jobs | Receive-Job
$jobs | Remove-Job
Write-Host "DONE: data_train/QuadrotorCombined/run201 and run205"
