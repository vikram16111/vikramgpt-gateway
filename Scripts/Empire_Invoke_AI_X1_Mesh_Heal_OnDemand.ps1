<#
  RUN ON AI_X1 (worker). Symmetric to Empire_Invoke_HP_Mesh_Heal_OnDemand.ps1.

  Runs Empire_Node_Mesh_Heal.ps1 -NodeRole AI_X1 once (Y: remap + SMB resolve to HP).

  Call before jobs that need \\HP-SPECTRE\Empire or Y:\ when the path is missing.

  Credentials: machine env EMPIRE_LAN_HP_USER / EMPIRE_LAN_HP_PASS (see Set_Empire_LAN_Link_To_HP.ps1).

  Deploy: ensure this file exists under C:\Empire_AI_X1\Scripts (Push_Empire_Mesh_Scripts_To_AI_X1.ps1 from HP).
#>
param(
  [string]$ScriptsRoot = $PSScriptRoot,
  [int]$PauseSecondsAfter = 3
)

$heal = Join-Path $ScriptsRoot "Empire_Node_Mesh_Heal.ps1"
if (-not (Test-Path -LiteralPath $heal)) {
  $heal = "C:\Empire\Scripts\Empire_Node_Mesh_Heal.ps1"
}
if (-not (Test-Path -LiteralPath $heal)) {
  Write-Host "[Empire] WARN: Missing Empire_Node_Mesh_Heal.ps1" -ForegroundColor DarkYellow
  return
}
$psExe = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
Write-Host "[Empire] On-demand LAN self-heal (AI_X1 -> HP) before mesh job..." -ForegroundColor Cyan
& $psExe -NoProfile -ExecutionPolicy Bypass -File $heal -NodeRole AI_X1
if ($PauseSecondsAfter -gt 0) {
  Start-Sleep -Seconds $PauseSecondsAfter
}
