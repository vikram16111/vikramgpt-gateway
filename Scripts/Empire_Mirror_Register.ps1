<#
  Dot-source or call functions. Append-only audit of mesh/git mirror events (UTC).
  Canonical log: <EmpireRoot>\generated\logs\EMPIRE_MIRROR_REGISTER.md
  Override root: env EMPIRE_ROOT
#>

function Get-EmpireMirrorRegisterRoot {
    $r = [Environment]::GetEnvironmentVariable('EMPIRE_ROOT', 'Machine')
    if ([string]::IsNullOrWhiteSpace($r)) { $r = [Environment]::GetEnvironmentVariable('EMPIRE_ROOT', 'User') }
    if (-not [string]::IsNullOrWhiteSpace($r)) { return $r.Trim() }
    $scripts = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
    return (Split-Path -Parent $scripts)
}

function Add-EmpireMirrorRegisterEntry {
    param(
        [Parameter(Mandatory = $true)][string]$SourceNode,
        [Parameter(Mandatory = $true)][string]$Action,
        [Parameter(Mandatory = $true)][string]$Target,
        [string]$Detail = ""
    )
    $empireRoot = Get-EmpireMirrorRegisterRoot
    $logDir = Join-Path $empireRoot "generated\logs"
    $logFile = Join-Path $logDir "EMPIRE_MIRROR_REGISTER.md"
    if (-not (Test-Path -LiteralPath $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }
    $ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss\Z")
    $safeDetail = ($Detail -replace '\|', '/' -replace "`r?`n", ' ')
    $line = "| $ts | $SourceNode | $Action | $Target | $safeDetail |"
    if (-not (Test-Path -LiteralPath $logFile)) {
        $header = @(
            '# EMPIRE_MIRROR_REGISTER',
            '',
            'Append-only. **Mirroring does not change ownership or authorship**; edit/delete rules follow the original owner and project doctrine.',
            '',
            '| UTC | NODE | ACTION | TARGET | DETAIL |',
            '|-----|------|--------|--------|--------|'
        )
        Set-Content -LiteralPath $logFile -Value $header -Encoding UTF8
    }
    Add-Content -LiteralPath $logFile -Value $line -Encoding UTF8
}
