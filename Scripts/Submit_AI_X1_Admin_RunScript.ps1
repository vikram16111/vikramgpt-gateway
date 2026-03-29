<#
  Submit an admin job to AI_X1 (RUN ON HP).

  Writes a request into AI_X1 node_bus admin inbox.
  AI_X1's AdminExecutor scheduled task executes it elevated.

  Task type: run_script (allowlisted: script under C:\Empire_AI_X1\Scripts\*.ps1).

  Bus root resolution (no Join-Path on missing Z:):
    1) -BusRoot if that path exists
    2) EMPIRE_AI_X1_NODE_BUS
    3) node_bus_roots from Scripts\empire_node_routes.json (HP node)
    4) UNC from generated\health\empire_lan_worker_last_ip.txt
    5) Z:\node_bus

  If primary path missing: runs Empire_Invoke_HP_Mesh_Heal_OnDemand.ps1 once (remap Z / resolve worker), then retries.
  If still no root: Yellow message + exit 1 (recoverable mesh failure; not throw).

  -SkipLanHeal: skip on-demand heal (debug only).
#>
param(
  [Parameter(Mandatory=$true)][string]$ScriptPathOnAI,
  [string]$Args = "",
  [string]$BusRoot = "Z:\node_bus",
  [string]$FromNode = "HP",
  [string]$ToNode = "AI_X1",
  [switch]$SkipLanHeal
)

$ErrorActionPreference = "Stop"

function Convert-NodeToFolder([string]$NodeName) { return (($NodeName.ToLower() -replace "[^a-z0-9]+", "_").Trim("_")) }

function Resolve-EmpireAiX1NodeBusRoot {
  param(
    [string]$Preferred = ""
  )
  $candidates = New-Object System.Collections.Generic.List[string]
  if ($Preferred) { [void]$candidates.Add($Preferred.Trim().TrimEnd('\')) }

  $envBus = $env:EMPIRE_AI_X1_NODE_BUS
  if ($envBus) { [void]$candidates.Add($envBus.Trim().TrimEnd('\')) }

  $routesPath = Join-Path $PSScriptRoot "empire_node_routes.json"
  if (Test-Path -LiteralPath $routesPath) {
    try {
      $j = Get-Content -LiteralPath $routesPath -Raw -Encoding UTF8 | ConvertFrom-Json
      $hp = @($j.nodes | Where-Object { $_.name -eq "HP" } | Select-Object -First 1)
      if ($hp -and $hp.node_bus_roots) {
        foreach ($nb in $hp.node_bus_roots) {
          if ($nb) { [void]$candidates.Add([string]$nb.Trim().TrimEnd('\')) }
        }
      }
    } catch {
      # ignore bad JSON; fall through to IP + Z
    }
  }

  $hint = Join-Path $PSScriptRoot "..\generated\health\empire_lan_worker_last_ip.txt"
  if (-not (Test-Path -LiteralPath $hint)) {
    $hint = "C:\Empire\generated\health\empire_lan_worker_last_ip.txt"
  }
  if (Test-Path -LiteralPath $hint) {
    $wip = (Get-Content -LiteralPath $hint -First 1).Trim()
    if ($wip -match '^\d{1,3}(\.\d{1,3}){3}$') {
      [void]$candidates.Add("\\$wip\Empire_AI_X1\node_bus")
    }
  }

  [void]$candidates.Add("Z:\node_bus")

  $seen = @{}
  foreach ($root in $candidates) {
    if ([string]::IsNullOrWhiteSpace($root)) { continue }
    $k = $root.ToLowerInvariant()
    if ($seen.ContainsKey($k)) { continue }
    $seen[$k] = $true
    try {
      if (Test-Path -LiteralPath $root) {
        return [pscustomobject]@{ Ok = $true; Root = $root; Tried = "" }
      }
    } catch {
      continue
    }
  }

  $tried = ($candidates | Where-Object { $_ } | ForEach-Object { $_.ToString() } | Select-Object -Unique) -join "; "
  return [pscustomobject]@{ Ok = $false; Root = ""; Tried = $tried }
}

function Test-NodeBusLiteral([string]$p) {
  if ([string]::IsNullOrWhiteSpace($p)) { return $false }
  try { return Test-Path -LiteralPath $p.Trim().TrimEnd('\') } catch { return $false }
}

function Invoke-EmpireHpMeshHealOnDemand {
  $invoke = Join-Path $PSScriptRoot "Empire_Invoke_HP_Mesh_Heal_OnDemand.ps1"
  if (-not (Test-Path -LiteralPath $invoke)) { return }
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $invoke -ScriptsRoot $PSScriptRoot
}

$trimmed = $BusRoot.Trim().TrimEnd('\')
if (-not (Test-NodeBusLiteral $trimmed)) {
  if (-not $SkipLanHeal) {
    Invoke-EmpireHpMeshHealOnDemand
    Start-Sleep -Seconds 2
  }
}
if (-not (Test-NodeBusLiteral $trimmed)) {
  $res = Resolve-EmpireAiX1NodeBusRoot -Preferred $BusRoot
  if (-not $res.Ok) {
    if (-not $SkipLanHeal) {
      Write-Host "[Empire] Retrying LAN heal once more, then node_bus resolve..." -ForegroundColor DarkYellow
      Invoke-EmpireHpMeshHealOnDemand
      Start-Sleep -Seconds 3
      $res = Resolve-EmpireAiX1NodeBusRoot -Preferred $BusRoot
    }
  }
  if (-not $res.Ok) {
    Write-Host "No reachable node_bus root. Tried: $($res.Tried)" -ForegroundColor Yellow
    Write-Host "After heal: confirm AI_X1 online, SMB share Empire_AI_X1, machine env EMPIRE_LAN_*, or set EMPIRE_AI_X1_NODE_BUS." -ForegroundColor Yellow
    exit 1
  }
  $BusRoot = $res.Root
} else {
  $BusRoot = $trimmed
}

$toFolder = Convert-NodeToFolder -NodeName $ToNode
$inbox = Join-Path $BusRoot ("to_" + $toFolder + "\admin_inbox")
if (-not (Test-Path -LiteralPath $inbox)) { New-Item -Path $inbox -ItemType Directory -Force | Out-Null }

$jobId = "admin_{0}_{1}" -f (Get-Date -Format "yyyyMMdd_HHmmss"), ([guid]::NewGuid().ToString("N").Substring(0, 8))
$request = @{
  job_id = $jobId
  from = $FromNode
  to = $ToNode
  task = "run_script"
  payload = @{
    script = $ScriptPathOnAI
    args = $Args
  }
  created_at = (Get-Date).ToString("s")
}

$path = Join-Path $inbox "$jobId.request.json"
$request | ConvertTo-Json -Depth 8 | Set-Content -Path $path -Encoding UTF8
Write-Output "Submitted admin job: $jobId"
Write-Output "Request path: $path"
Write-Output "Bus root used: $BusRoot"
