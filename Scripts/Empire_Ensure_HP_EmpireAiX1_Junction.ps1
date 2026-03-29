<#
  RUN ON HP AS ADMIN (once per machine, or after Z: mapping changes).

  Creates directory junction C:\Empire_AI_X1 -> Z:\ when Z: is the Empire_AI_X1 share root.
  So agents and tools can use C:\Empire_AI_X1 on HP the same path shape as on the worker.

  Preconditions:
    - Z:\ must already map to \\<worker>\Empire_AI_X1 (run Set_Empire_LAN_Link.ps1 or mesh heal first).
    - C:\Empire_AI_X1 must NOT exist as a normal folder with data (mklink will fail).

  Idempotent: if C:\Empire_AI_X1 already exists and is a junction to Z:\, exits OK.
#>
param(
  [string]$TargetDrive = "Z:"
)

$ErrorActionPreference = "Stop"
$junctionPath = "C:\Empire_AI_X1"
$targetRoot = if ($TargetDrive.EndsWith("\")) { $TargetDrive.TrimEnd("\") } else { $TargetDrive }

if (-not (Test-Path -LiteralPath $targetRoot)) {
  Write-Host "Target $targetRoot not found. Run mesh heal / Set_Empire_LAN_Link.ps1 first." -ForegroundColor Yellow
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
