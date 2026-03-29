<#
  Sync shared Cursor rules from Empire CORE to discovered workspaces.
  Edit rules only under C:\Empire\.cursor\rules then run this script.

  -Strategy Auto (default): FULL copy only for roots listed in generated\cursor_paths_full_rules_roots.txt
    (default: C:\Empire). All other discovered roots get a single STUB pointer rule (micro-satellite model).
  -Strategy Full: copy all rules everywhere (legacy; many file copies).
  -Strategy Stub: stub only for all targets (except canonical .cursor\rules self).
#>
param(
  [ValidateSet('Auto', 'Full', 'Stub')]
  [string]$Strategy = 'Auto'
)

$ErrorActionPreference = 'Stop'
$core = 'C:\Empire\.cursor\rules'
$discoverFile = 'C:\Empire\generated\DISCOVERED_CURSOR_PATHS.txt'
$fullRootsFile = 'C:\Empire\generated\cursor_paths_full_rules_roots.txt'
$files = @(
  'core-vms-autonomy-handover.mdc',
  'core-vms-handsfree-eyesfree-ux.mdc',
  'core-hp-bootstrap-one-recovery.mdc',
  'core-aws-secrets-backup-before-change.mdc',
  'core-empire-secrets-strict.mdc',
  'empire-twenty-slots-universal.mdc',
  'core-gdrive-precise-paths.mdc',
  'core-gdrive-upload-consent-and-canonical-ids.mdc',
  'core-cg-gdrive-structured-layout.mdc',
  'core-execution-verification-intent.mdc',
  'core-generated-child-canonical.mdc',
  'core-empire-reading-no-empty-staging.mdc',
  'core-hp-can-be-off-24x7-services.mdc',
  'core-node-utility-suffix-naming.mdc',
  'core-human-auditable-names-and-readmes.mdc',
  'core-no-popup-background-automation.mdc',
  'core-one-line-copypaste.mdc',
  'core-vms-no-scroll-reply.mdc',
  'core-empire-agent-decision-order.mdc',
  'core-powershell-empire-unified.mdc',
  'core-lambda-authoring-parity.mdc',
  'core-restructure-no-orphans.mdc',
  'core-incremental-reuse-no-full-rewrite.mdc',
  'core-cursor-context-handover.mdc',
  'core-empire-cross-node-filesystem-read.mdc',
  'core-empire-hp-agent-ai-x1-execution.mdc',
  'core-empire-multi-node-all-nodes-memory.mdc',
  'core-empire-mesh-lan-plans-abcd.mdc',
  'core-discovered-paths-sync.mdc',
  'core-empire-pointer-to-canonical.mdc',
  'core-empire-workspace-rename-relocate-discipline.mdc',
  'core-empire-project-workspace-naming.mdc',
  'core-empire-multi-node-mirror-register.mdc',
  'core-empire-adhoc-vs-constitution.mdc',
  'core-empire-multi-node-folder-dna.mdc'
)

$pointerOnly = @('core-empire-pointer-to-canonical.mdc')

if (-not (Test-Path $discoverFile)) {
  throw "Missing discovery list: $discoverFile (run: python C:\Empire\Scripts\Sync_Discovered_Cursor_Paths.py)"
}

function Get-FullRuleRootConfig {
  $exact = New-Object 'System.Collections.Generic.HashSet[string]'
  $prefix = New-Object 'System.Collections.Generic.HashSet[string]'
  [void]$exact.Add((Resolve-Path 'C:\Empire').Path.ToLowerInvariant())
  if (Test-Path -LiteralPath $fullRootsFile) {
    Get-Content -LiteralPath $fullRootsFile -Encoding UTF8 | ForEach-Object {
      $line = ($_.Split('#', 2)[0]).Trim()
      if (-not $line) { }
      elseif ($line -match '^(?i)PREFIX\s+(.+)$') {
        $p = $Matches[1].Trim()
        if ($p -and (Test-Path -LiteralPath $p)) {
          [void]$prefix.Add((Resolve-Path -LiteralPath $p).Path.ToLowerInvariant())
        }
      }
      elseif (Test-Path -LiteralPath $line) {
        [void]$exact.Add((Resolve-Path -LiteralPath $line).Path.ToLowerInvariant())
      }
    }
  }
  return [pscustomobject]@{ Exact = $exact; Prefix = $prefix }
}

function Test-WorkspaceGetsFullRules([string]$RootResolvedLower, $Cfg) {
  if ($Cfg.Exact.Contains($RootResolvedLower)) { return $true }
  foreach ($p in $Cfg.Prefix) {
    if ($RootResolvedLower -eq $p) { return $true }
    if ($RootResolvedLower.StartsWith($p + '\')) { return $true }
  }
  return $false
}

$fullCfg = Get-FullRuleRootConfig

$workspaceRoots = Get-Content $discoverFile -Encoding UTF8 | Where-Object {
  $t = $_.Trim()
  $t -ne '' -and -not $t.StartsWith('#') -and ($t -match '^[A-Za-z]:\\')
} | Sort-Object -Unique

$targets = @()
foreach ($root in $workspaceRoots) {
  $targets += (Join-Path $root '.cursor\rules')
}

$synced = 0
$fullTargets = 0
$stubTargets = 0
$coreResolved = (Resolve-Path $core).Path

foreach ($t in ($targets | Sort-Object -Unique)) {
  if (-not (Test-Path $t)) { New-Item -ItemType Directory -Path $t -Force | Out-Null }
  $tr = (Resolve-Path $t).Path
  if ($tr -eq $coreResolved) { continue }

  $parentRoot = Split-Path -Parent (Split-Path -Parent $tr)
  try {
    $rootResolved = (Resolve-Path -LiteralPath $parentRoot).Path.ToLowerInvariant()
  } catch {
    $rootResolved = $parentRoot.ToLowerInvariant()
  }

  $useFull = $false
  if ($Strategy -eq 'Full') { $useFull = $true }
  elseif ($Strategy -eq 'Stub') { $useFull = $false }
  else {
    $useFull = Test-WorkspaceGetsFullRules -RootResolvedLower $rootResolved -Cfg $fullCfg
  }

  if ($useFull) {
    $fullTargets++
    foreach ($f in $files) {
      $src = Join-Path $core $f
      if (-not (Test-Path $src)) { throw "Missing CORE rule: $src" }
      Copy-Item -LiteralPath $src -Destination (Join-Path $tr $f) -Force
      $synced++
    }
  } else {
    $stubTargets++
    foreach ($f in $files) {
      $dest = Join-Path $tr $f
      if (Test-Path -LiteralPath $dest) { Remove-Item -LiteralPath $dest -Force -ErrorAction SilentlyContinue }
    }
    foreach ($f in $pointerOnly) {
      $src = Join-Path $core $f
      if (-not (Test-Path $src)) { throw "Missing CORE pointer: $src" }
      Copy-Item -LiteralPath $src -Destination (Join-Path $tr $f) -Force
      $synced++
    }
  }
}

$modeDesc = if ($Strategy -eq 'Full') { 'FULL all roots' } elseif ($Strategy -eq 'Stub') { 'STUB all roots' } else { 'AUTO' }
Write-Host ("OK: strategy={0} ({1}). Workspaces scanned: {2} (excludes canonical CORE self). FULL rule roots: {3}; STUB-only roots: {4}; total file copy ops: {5}." -f $Strategy, $modeDesc, $targets.Count, $fullTargets, $stubTargets, $synced)
Write-Host "  STUB = one pointer .mdc per workspace; FULL = all shared rules (see generated\cursor_paths_full_rules_roots.txt)." -ForegroundColor DarkGray
