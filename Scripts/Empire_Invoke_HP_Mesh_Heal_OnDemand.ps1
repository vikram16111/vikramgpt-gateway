<#
  Runs Empire_Node_Mesh_Heal.ps1 -NodeRole HP once (Z: remap + SMB resolve to AI_X1).

  Call this when a job needs the worker share / node_bus and the path is not reachable yet.
  Uses a child PowerShell process so heal errors do not abort the caller.

  Credentials: machine env EMPIRE_LAN_USER / EMPIRE_LAN_PASS (see Set_Empire_LAN_Link.ps1).
#>
param(
  [string]$ScriptsRoot = $PSScriptRoot,
  [int]$PauseSecondsAfter = 3
)

$heal = Join-Path $ScriptsRoot "Empire_Node_Mesh_Heal.ps1"
if (-not (Test-Path -LiteralPath $heal)) {
  Write-Host "[Empire] WARN: Missing $heal" -ForegroundColor DarkYellow
  return
}
$psExe = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
Write-Host "[Empire] On-demand LAN self-heal (HP -> AI_X1) before mesh job..." -ForegroundColor Cyan
& $psExe -NoProfile -ExecutionPolicy Bypass -File $heal -NodeRole HP
if ($PauseSecondsAfter -gt 0) {
  Start-Sleep -Seconds $PauseSecondsAfter
}
