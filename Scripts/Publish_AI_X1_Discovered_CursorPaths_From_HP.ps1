<#
  RUN ON HP: publish DISCOVERED_CURSOR_PATHS_LOCAL.txt onto the AI_X1 share (Z: or UNC)
  using only the EMPIRE_ROOT tree (no D: / vikramgpt scan), then refresh HP canonical merge.

  Optional: EMPIRE_AI_X1_UNC_ROOT = \\host\Empire_AI_X1 when Z: is not mapped.
#>
$ErrorActionPreference = "Stop"

$meshHeal = Join-Path $PSScriptRoot "Empire_Invoke_HP_Mesh_Heal_OnDemand.ps1"

function Get-EmpireAiX1ShareRoot {
  if ($env:EMPIRE_AI_X1_UNC_ROOT) {
    $u = $env:EMPIRE_AI_X1_UNC_ROOT.Trim().TrimEnd('\')
    if (Test-Path -LiteralPath $u) { return $u }
  }
  if (Test-Path -LiteralPath "Z:\") {
    $z = (Resolve-Path -LiteralPath "Z:\").Path.TrimEnd('\')
    return $z
  }
  $routesPath = Join-Path $PSScriptRoot "empire_node_routes.json"
  if (Test-Path -LiteralPath $routesPath) {
    try {
      $j = Get-Content -LiteralPath $routesPath -Raw -Encoding UTF8 | ConvertFrom-Json
      $ai = @($j.nodes | Where-Object { $_.name -eq "AI_X1" } | Select-Object -First 1)
      if ($ai -and $ai.lan_shares) {
        foreach ($s in $ai.lan_shares) {
          $p = [string]$s.Trim().TrimEnd('\')
          if ($p -match '^\\\\' -and (Test-Path -LiteralPath $p)) { return $p }
        }
      }
    } catch {
    }
  }
  return $null
}

$shareRoot = Get-EmpireAiX1ShareRoot
if (-not $shareRoot -and (Test-Path -LiteralPath $meshHeal)) {
  Write-Host "AI_X1 share not visible; on-demand HP mesh heal, then retry..." -ForegroundColor DarkYellow
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $meshHeal -ScriptsRoot $PSScriptRoot
  Start-Sleep -Seconds 3
  $shareRoot = Get-EmpireAiX1ShareRoot
}
if (-not $shareRoot) {
  Write-Host "No AI_X1 share visible. Map Z:, set EMPIRE_AI_X1_UNC_ROOT, or fix LAN." -ForegroundColor Yellow
  exit 1
}

$py = Join-Path $PSScriptRoot "Sync_Discovered_Cursor_Paths.py"
if (-not (Test-Path -LiteralPath $py)) { throw "Missing $py" }

$outRel = "generated\DISCOVERED_CURSOR_PATHS_LOCAL.txt"
$outFile = Join-Path $shareRoot $outRel
$outDir = Split-Path -Parent $outFile
if (-not (Test-Path -LiteralPath $outDir)) {
  New-Item -Path $outDir -ItemType Directory -Force | Out-Null
}

$env:EMPIRE_ROOT = $shareRoot
$env:DISCOVERED_CURSOR_OUTPUT = $outFile
$env:DISCOVERED_CURSOR_NO_MERGE = "1"
$env:DISCOVERED_CURSOR_LOCAL_TREE_ONLY = "1"
try {
  & python $py
  if ($LASTEXITCODE -ne 0) { throw "Python exit $LASTEXITCODE" }
} finally {
  Remove-Item Env:EMPIRE_ROOT -ErrorAction SilentlyContinue
  Remove-Item Env:DISCOVERED_CURSOR_OUTPUT -ErrorAction SilentlyContinue
  Remove-Item Env:DISCOVERED_CURSOR_NO_MERGE -ErrorAction SilentlyContinue
  Remove-Item Env:DISCOVERED_CURSOR_LOCAL_TREE_ONLY -ErrorAction SilentlyContinue
}

Write-Host "Published worker local list: $outFile" -ForegroundColor Green

& python $py
if ($LASTEXITCODE -ne 0) { throw "Canonical merge python exit $LASTEXITCODE" }
Write-Host "Refreshed C:\Empire\generated\DISCOVERED_CURSOR_PATHS.txt (merge if share file visible)." -ForegroundColor Cyan
exit 0
