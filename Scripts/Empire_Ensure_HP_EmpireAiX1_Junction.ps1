<#
  RUN ON HP AS ADMIN (once per machine, or after mapping changes).

  Creates a **directory symlink** C:\Empire_AI_X1 -> UNC worker root (NOT a junction to Z:).
  Windows junctions (mklink /J) cannot target a network drive; "Local volumes are required" is expected for /J -> Z:.

  Worker root is resolved from the SMB path of Z: (Get-SmbMapping / net use), then tries in order:
    \\<server>\Empire_AI_X1
    \\<server>\<share>\Empire_AI_X1   (when Z: is e.g. \\ip\Empire)

  If Z: is missing: runs HP mesh heal, waits, retries.

  -SkipLanHeal: do not auto-heal (debug only).
#>
param(
  [string]$TargetDrive = "Z:",
  [int]$HealAttempts = 2,
  [switch]$SkipLanHeal
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Empire_Worker_Smb_Resolve.ps1")
$junctionPath = "C:\Empire_AI_X1"
$targetDriveLetter = ($TargetDrive.Trim() -replace ':$', '')
if ($targetDriveLetter.Length -ne 1) {
  Write-Host "FAIL: TargetDrive must be a single letter drive, e.g. Z:" -ForegroundColor Red
  exit 1
}
$localPath = ($targetDriveLetter.ToUpper() + ":")

$invokeHeal = Join-Path $PSScriptRoot "Empire_Invoke_HP_Mesh_Heal_OnDemand.ps1"

function Test-ZOrTarget {
  param([string]$p)
  if ([string]::IsNullOrWhiteSpace($p)) { return $false }
  try { return Test-Path -LiteralPath $p } catch { return $false }
}

function Get-SmbRemotePathForDriveLetter {
  param([string]$LetterColon)
  $lc = $LetterColon.Trim()
  if ($lc -notmatch '^[A-Za-z]:$') { return $null }
  try {
    $m = Get-SmbMapping -LocalPath $lc -ErrorAction Stop
    if ($m -and $m.RemotePath) { return ([string]$m.RemotePath).Trim() }
  } catch {}
  $net = cmd.exe /c "net use $lc" 2>&1 | Out-String
  if ($net -match '(?im)Remote name\s+(\\\S+)') { return $Matches[1].Trim() }
  return $null
}

function Resolve-EmpireAiX1RootUncFromMappedDrive {
  param([string]$MappedRemoteUnc)
  if ([string]::IsNullOrWhiteSpace($MappedRemoteUnc)) { return $null }
  if ($MappedRemoteUnc -notmatch '^\\\\([^\\]+)\\([^\\]+)$') { return $null }
  $server = $Matches[1]
  $share = $Matches[2]
  $ordered = @(
    "\\$server\Empire_AI_X1",
    "\\$server\$share\Empire_AI_X1"
  )
  $seen = @{}
  foreach ($u in $ordered) {
    $k = $u.ToLowerInvariant()
    if ($seen.ContainsKey($k)) { continue }
    $seen[$k] = $true
    try {
      if (Test-Path -LiteralPath $u) { return $u }
    } catch {}
  }
  return $null
}

if (-not (Test-ZOrTarget $localPath)) {
  if (-not $SkipLanHeal -and (Test-Path -LiteralPath $invokeHeal)) {
    for ($i = 0; $i -lt $HealAttempts; $i++) {
      Write-Host "[Empire] $localPath missing - running HP mesh heal (attempt $($i + 1)/$HealAttempts)..." -ForegroundColor Cyan
      & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $invokeHeal -ScriptsRoot $PSScriptRoot
      Start-Sleep -Seconds 4
      if (Test-ZOrTarget $localPath) { break }
    }
  }
}

if (-not (Test-ZOrTarget $localPath)) {
  Write-Host "Target $localPath still not reachable after heal." -ForegroundColor Yellow
  Write-Host "Check: AI_X1 online, LAN, firewall 445, machine env EMPIRE_LAN_USER/EMPIRE_LAN_PASS, or run:" -ForegroundColor Yellow
  Write-Host '  powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Empire\Scripts\Diagnose_Empire_HP_To_AI_X1_Lan.ps1"' -ForegroundColor DarkYellow
  exit 1
}

$remoteZ = Get-SmbRemotePathForDriveLetter -LetterColon $localPath
if (-not $remoteZ) {
  Write-Host "FAIL: Could not read SMB remote path for $localPath (net use / Get-SmbMapping)." -ForegroundColor Red
  exit 1
}

if ($remoteZ -match '^\\\\(\d{1,3}(?:\.\d{1,3}){3})\\') {
  $zSrv = $Matches[1]
  if (Test-EmpireIpIsThisMachine -Ip $zSrv) {
    if (Remove-EmpireMappedDriveIfRemoteIsThisMachine -DriveLetter $targetDriveLetter) {
      Write-Host "Auto-heal: disconnected ${localPath} (was loopback to this PC). Re-run Set_Empire_LAN_Link.ps1 (or mesh heal), then this script." -ForegroundColor Yellow
    } else {
      Write-Host "FAIL: ${localPath} points at THIS PC ($zSrv), not AI_X1. Run Set_Empire_LAN_Link.ps1 after fixing SMB on AI_X1." -ForegroundColor Yellow
    }
    exit 1
  }
}

$workerRootUnc = Resolve-EmpireAiX1RootUncFromMappedDrive -MappedRemoteUnc $remoteZ
if (-not $workerRootUnc) {
  Write-Host "FAIL: No worker root UNC found from mapped drive $localPath -> $remoteZ" -ForegroundColor Yellow
  Write-Host "Expected an existing path: \\<server>\Empire_AI_X1 or $remoteZ\Empire_AI_X1" -ForegroundColor Yellow
  Write-Host "Fix: on AI_X1 share Empire_AI_X1 (or nest Empire_AI_X1 under the share you mapped), then re-run Set_Empire_LAN_Link / mesh heal." -ForegroundColor Yellow
  exit 1
}

Write-Host "[Empire] Z: maps to: $remoteZ" -ForegroundColor DarkGray
Write-Host "[Empire] Symlink target (worker root): $workerRootUnc" -ForegroundColor DarkGray

if (Test-Path -LiteralPath $junctionPath) {
  $item = Get-Item -LiteralPath $junctionPath -Force -ErrorAction SilentlyContinue
  if ($item -and $item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
    Write-Host "OK: $junctionPath already exists as reparse point (junction/symlink)." -ForegroundColor Green
    exit 0
  }
  Write-Host "FAIL: $junctionPath exists and is not a symlink/junction. Remove or rename it manually, then re-run." -ForegroundColor Red
  exit 1
}

Write-Host "Creating directory symlink: $junctionPath -> $workerRootUnc (Admin + symlink privilege; NOT mklink /J to Z:)" -ForegroundColor Cyan
cmd.exe /c "mklink /D `"$junctionPath`" `"$workerRootUnc`""
if ($LASTEXITCODE -ne 0) {
  Write-Host "mklink /D failed (exit $LASTEXITCODE). Run PowerShell as Administrator; enable symlink privilege / Developer Mode if required." -ForegroundColor Red
  exit $LASTEXITCODE
}
Write-Host "OK: directory symlink created." -ForegroundColor Green
exit 0
