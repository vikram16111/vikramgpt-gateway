<#
  RUN ON HP AS ADMIN (once per machine, or after mapping changes).

  Creates directory junction C:\Empire_AI_X1 -> Z:\ when Z: is the Empire_AI_X1 share root.

  If Z: is missing: automatically runs HP mesh heal (same as node_bus jobs), waits, retries - hands-free.

  C:\Empire_AI_X1 must NOT exist as a normal folder with data (mklink will fail).

  Idempotent: if C:\Empire_AI_X1 already exists as junction to Z:\, exits OK.

  -SkipLanHeal: do not auto-heal (debug only).
#>
param(
  [string]$TargetDrive = "Z:",
  [int]$HealAttempts = 2,
  [switch]$SkipLanHeal
)

$ErrorActionPreference = "Stop"
$junctionPath = "C:\Empire_AI_X1"
$targetRoot = if ($TargetDrive.EndsWith("\")) { $TargetDrive.TrimEnd("\") } else { $TargetDrive }

$invokeHeal = Join-Path $PSScriptRoot "Empire_Invoke_HP_Mesh_Heal_OnDemand.ps1"

function Test-ZOrTarget {
  param([string]$p)
  if ([string]::IsNullOrWhiteSpace($p)) { return $false }
  try { return Test-Path -LiteralPath $p } catch { return $false }
}

if (-not (Test-ZOrTarget $targetRoot)) {
  if (-not $SkipLanHeal -and (Test-Path -LiteralPath $invokeHeal)) {
    for ($i = 0; $i -lt $HealAttempts; $i++) {
      Write-Host "[Empire] $targetRoot missing - running HP mesh heal (attempt $($i + 1)/$HealAttempts)..." -ForegroundColor Cyan
      & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $invokeHeal -ScriptsRoot $PSScriptRoot
      Start-Sleep -Seconds 4
      if (Test-ZOrTarget $targetRoot) { break }
    }
  }
}

if (-not (Test-ZOrTarget $targetRoot)) {
  Write-Host "Target $targetRoot still not reachable after heal." -ForegroundColor Yellow
  Write-Host "Check: AI_X1 online, LAN, firewall 445, machine env EMPIRE_LAN_USER/EMPIRE_LAN_PASS, or run:" -ForegroundColor Yellow
  Write-Host '  powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Empire\Scripts\Diagnose_Empire_HP_To_AI_X1_Lan.ps1"' -ForegroundColor DarkYellow
  exit 1
}

if (Test-Path -LiteralPath $junctionPath) {
  $item = Get-Item -LiteralPath $junctionPath -Force -ErrorAction SilentlyContinue
  if ($item -and $item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
    Write-Host "OK: $junctionPath already exists as reparse point (junction/symlink)." -ForegroundColor Green
    exit 0
  }
  Write-Host "FAIL: $junctionPath exists and is not a junction. Remove or rename it manually, then re-run." -ForegroundColor Red
  exit 1
}

Write-Host "Creating junction: $junctionPath -> $targetRoot (requires Admin)" -ForegroundColor Cyan
cmd.exe /c "mklink /J `"$junctionPath`" `"$targetRoot`""
if ($LASTEXITCODE -ne 0) {
  Write-Host "mklink failed (exit $LASTEXITCODE). Run PowerShell as Administrator." -ForegroundColor Red
  exit $LASTEXITCODE
}
Write-Host "OK: junction created." -ForegroundColor Green
exit 0
