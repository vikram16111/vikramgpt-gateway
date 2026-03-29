<#
  RUN ON HP when Z: maps to worker share, OR ON WORKER with -WorkerRoot C:\Empire_AI_X1.

  Ensures DNA-aligned skeleton: generated\*, repos\vikramgpt-gateway (git clone),
  Project_LAN_Connections_Nodes_AI_X1 from Empire_Worker_Scaffold_Assets, Z:\.cursor\rules shell.

  After this (from HP): python Scripts\Sync_Discovered_Cursor_Paths.py
                       Sync_Empire_Shared_Cursor_Rules.ps1
#>
param(
    [string]$WorkerRoot = "",
    [string]$GitRemote = "https://github.com/vikram16111/vikramgpt-gateway.git",
    [switch]$SkipGitClone,
    [switch]$SkipLanProject
)

$ErrorActionPreference = "Stop"
$scriptsRoot = $PSScriptRoot

if ([string]::IsNullOrWhiteSpace($WorkerRoot)) {
    if (Test-Path -LiteralPath "Z:\") {
        $WorkerRoot = "Z:\"
    } else {
        Write-Host "[Empire-Skeleton] FAIL: No -WorkerRoot and Z:\ not mapped. Map LAN or pass -WorkerRoot C:\Empire_AI_X1" -ForegroundColor Yellow
        exit 1
    }
}

$WorkerRoot = ($WorkerRoot.TrimEnd('\') + '\')
if (-not (Test-Path -LiteralPath $WorkerRoot)) {
    Write-Host "[Empire-Skeleton] FAIL: Worker root not reachable: $WorkerRoot" -ForegroundColor Yellow
    exit 1
}

Write-Host "[Empire-Skeleton] Worker root: $WorkerRoot"

$dirs = @(
    "generated\health",
    "generated\logs",
    "generated\cursor-reference",
    "repos"
)
foreach ($rel in $dirs) {
    $p = Join-Path $WorkerRoot $rel
    if (-not (Test-Path -LiteralPath $p)) {
        New-Item -ItemType Directory -Path $p -Force | Out-Null
        Write-Host "OK: mkdir $rel"
    }
}

$cursorRules = Join-Path $WorkerRoot ".cursor\rules"
if (-not (Test-Path -LiteralPath $cursorRules)) {
    New-Item -ItemType Directory -Path $cursorRules -Force | Out-Null
}
$empireRoot = Split-Path -Parent $scriptsRoot
$ptr = Join-Path $empireRoot ".cursor\rules\core-empire-pointer-to-canonical.mdc"
if (Test-Path -LiteralPath $ptr) {
    Copy-Item -LiteralPath $ptr -Destination (Join-Path $cursorRules "core-empire-pointer-to-canonical.mdc") -Force
    Write-Host "OK: worker .cursor/rules pointer"
} else {
    Write-Host "[Empire-Skeleton] WARN: no core-empire-pointer-to-canonical.mdc to copy" -ForegroundColor DarkYellow
}

$assetSrc = Join-Path $scriptsRoot "Empire_Worker_Scaffold_Assets\Project_LAN_Connections_Nodes_AI_X1"
$projDst = Join-Path $WorkerRoot "Project_LAN_Connections_Nodes_AI_X1"
if (-not $SkipLanProject) {
    if (Test-Path -LiteralPath $assetSrc) {
        if (Test-Path -LiteralPath $projDst) {
            Remove-Item -LiteralPath $projDst -Recurse -Force -ErrorAction SilentlyContinue
        }
        Copy-Item -LiteralPath $assetSrc -Destination $projDst -Recurse -Force
        Write-Host "OK: Project_LAN_Connections_Nodes_AI_X1 from scaffold assets"
    } else {
        Write-Host "[Empire-Skeleton] WARN: missing $assetSrc (run Push from HP or copy Scripts tree)" -ForegroundColor DarkYellow
    }
}

if (-not $SkipGitClone) {
    $repos = Join-Path $WorkerRoot "repos"
    $clone = Join-Path $repos "vikramgpt-gateway"
    if (Test-Path -LiteralPath (Join-Path $clone ".git")) {
        Write-Host "OK: git clone already exists: $clone"
    } else {
        $gitExe = (Get-Command git -ErrorAction SilentlyContinue).Source
        if (-not $gitExe) {
            Write-Host "[Empire-Skeleton] WARN: git not in PATH; skip clone" -ForegroundColor DarkYellow
        } else {
            Write-Host "[Empire-Skeleton] Cloning $GitRemote -> $clone"
            & git clone -- $GitRemote $clone
            if ($LASTEXITCODE -ne 0) {
                Write-Host "[Empire-Skeleton] WARN: git clone exit $LASTEXITCODE (private repo? run auth on worker)" -ForegroundColor DarkYellow
            } else {
                Write-Host "OK: git clone done"
            }
        }
    }
}

Write-Host "[Empire-Skeleton] Next on HP: python C:\Empire\Scripts\Sync_Discovered_Cursor_Paths.py"
Write-Host "[Empire-Skeleton] Next on HP: Sync_Empire_Shared_Cursor_Rules.ps1 (Auto or Full)"
exit 0
