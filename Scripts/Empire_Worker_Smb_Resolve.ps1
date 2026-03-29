<#

  Dot-source from Push / Diagnose / Set_Empire_LAN_Link / Pull.

  Central SMB candidate layouts + session probe (uses EMPIRE_LAN_USER / EMPIRE_LAN_PASS Machine env).

  ASCII only.

#>



function Get-CmdNulQuietResolve {

    return '>' + 'nul 2' + '>' + '&1'

}



function Convert-RepoRelPathToCandidate {

    param([string]$RelPath)

    $rel = ($RelPath -replace '/', '\').Trim('\')

    if ([string]::IsNullOrWhiteSpace($rel)) { return $null }

    $idx = $rel.IndexOf('\')

    if ($idx -lt 0) {

        return [pscustomobject]@{ share = $rel; scriptsSubPath = 'Scripts'; label = ('repo_rel:' + $rel) }

    }

    $s = $rel.Substring(0, $idx)

    $rest = $rel.Substring($idx + 1).Trim('\')

    if ([string]::IsNullOrWhiteSpace($rest)) { $rest = 'Scripts' }

    return [pscustomobject]@{ share = $s; scriptsSubPath = $rest; label = ('repo_rel:' + $rel) }

}



function Get-EmpireWorkerSmbBuiltin {

    return @(

        [pscustomobject]@{ share = 'Empire_AI_X1'; scriptsSubPath = 'Scripts'; label = 'share_is_repo_root' },

        [pscustomobject]@{ share = 'C_AI_X1'; scriptsSubPath = 'Empire_AI_X1\Scripts'; label = 'nested_under_C_AI_X1' },

        [pscustomobject]@{ share = 'Empire'; scriptsSubPath = 'Empire_AI_X1\Scripts'; label = 'share_Empire_nested' }

    )

}



function Merge-CandidateRowsUnique {

    param([object[]]$Rows)

    $seen = @{}

    $out = @()

    foreach ($r in $Rows) {

        if (-not $r -or [string]::IsNullOrWhiteSpace($r.share)) { continue }

        $k = ($r.share.Trim().ToLowerInvariant()) + '|' + ($r.scriptsSubPath -replace '/', '\').Trim('\').ToLowerInvariant()

        if ($seen.ContainsKey($k)) { continue }

        $seen[$k] = $true

        $out += $r

    }

    return $out

}



function Get-EmpireWorkerSmbProbeServers {

    param(

        [string]$WorkerIp,

        [string]$WorkerHost,

        [string]$HealthDir

    )

    $ordered = @()

    $seen = @{}



    foreach ($x in @($WorkerIp, $WorkerHost)) {

        if ([string]::IsNullOrWhiteSpace($x)) { continue }

        $k = $x.Trim()

        if ($seen.ContainsKey($k)) { continue }

        $seen[$k] = $true

        $ordered += $k

    }



    $envHosts = [Environment]::GetEnvironmentVariable('EMPIRE_WORKER_SMB_HOSTS', 'Machine')

    if (-not $envHosts) { $envHosts = [Environment]::GetEnvironmentVariable('EMPIRE_WORKER_SMB_HOSTS', 'User') }

    if (-not [string]::IsNullOrWhiteSpace($envHosts)) {

        foreach ($h in $envHosts.Split(',')) {

            $k = $h.Trim()

            if ([string]::IsNullOrWhiteSpace($k)) { continue }

            if ($seen.ContainsKey($k)) { continue }

            $seen[$k] = $true

            $ordered += $k

        }

    }



    if (-not [string]::IsNullOrWhiteSpace($HealthDir)) {

        $jsonPath = Join-Path $HealthDir 'EMPIRE_WORKER_SMB_CANDIDATES.json'

        if (Test-Path -LiteralPath $jsonPath) {

            try {

                $j = (Get-Content -LiteralPath $jsonPath -Raw -Encoding UTF8) | ConvertFrom-Json

                if ($j.worker_hosts) {

                    foreach ($h in @($j.worker_hosts)) {

                        $k = ([string]$h).Trim()

                        if ([string]::IsNullOrWhiteSpace($k)) { continue }

                        if ($seen.ContainsKey($k)) { continue }

                        $seen[$k] = $true

                        $ordered += $k

                    }

                }

            } catch {}

        }

    }



    foreach ($b in @('AI-X1', 'ai-x1', 'Win11-ai_x1')) {

        if ($seen.ContainsKey($b)) { continue }

        $seen[$b] = $true

        $ordered += $b

    }



    return $ordered

}



function Get-EmpireWorkerSmbCandidates {

    param([string]$HealthDir)

    $jsonPath = Join-Path $HealthDir 'EMPIRE_WORKER_SMB_CANDIDATES.json'

    $builtIn = Get-EmpireWorkerSmbBuiltin

    if (-not (Test-Path -LiteralPath $jsonPath)) { return $builtIn }

    try {

        $j = (Get-Content -LiteralPath $jsonPath -Raw -Encoding UTF8) | ConvertFrom-Json

        $fromRepo = @()

        if ($j.repo_rel_paths) {

            foreach ($rp in @($j.repo_rel_paths)) {

                $c = Convert-RepoRelPathToCandidate -RelPath ([string]$rp)

                if ($c) { $fromRepo += $c }

            }

        }

        $fromExplicit = @()

        if ($j.candidates -and @($j.candidates).Count -gt 0) {

            $fromExplicit = @($j.candidates | ForEach-Object {

                [pscustomobject]@{

                    share          = [string]$_.share

                    scriptsSubPath = [string]$_.scriptsSubPath

                    label          = if ($_.label) { [string]$_.label } else { [string]$_.share }

                }

            })

        }

        if ($j.replace_repo_rel_paths -eq $true) {

            if ($fromRepo.Count -gt 0) { return (Merge-CandidateRowsUnique -Rows $fromRepo) }

            if ($fromExplicit.Count -gt 0) { return (Merge-CandidateRowsUnique -Rows $fromExplicit) }

            return $builtIn

        }

        $merged = @()

        $merged += $builtIn

        $merged += $fromExplicit

        $merged += $fromRepo

        return (Merge-CandidateRowsUnique -Rows $merged)

    } catch {

        return $builtIn

    }

}



function Join-WorkerServerShareUnc {

    param([string]$Server, [string]$Share, [string]$SubPath)

    $bs = [char]92

    $u = $bs + $bs + $Server + $bs + $Share

    if (-not [string]::IsNullOrWhiteSpace($SubPath)) {

        $rest = ($SubPath -replace '/', '\').Trim('\')

        if ($rest.Length -gt 0) { $u = $u + $bs + $rest }

    }

    return $u

}



function Get-MeshResolutionCachePath {

    param([string]$HealthDir)

    return (Join-Path $HealthDir 'empire_worker_mesh_resolution.json')

}



function Read-MeshResolutionCache {

    param([string]$HealthDir)

    $p = Get-MeshResolutionCachePath -HealthDir $HealthDir

    if (-not (Test-Path -LiteralPath $p)) { return $null }

    try {

        return ((Get-Content -LiteralPath $p -Raw -Encoding UTF8) | ConvertFrom-Json)

    } catch { return $null }

}



function Save-MeshResolutionCache {

    param(

        [string]$HealthDir,

        [string]$WorkerIp,

        [string]$WorkerShare,

        [string]$ScriptsSubPath,

        [string]$WorkerHost

    )

    $p = Get-MeshResolutionCachePath -HealthDir $HealthDir

    $o = [ordered]@{

        workerIp       = $WorkerIp

        workerShare    = $WorkerShare

        scriptsSubPath = $ScriptsSubPath

        workerHost     = $WorkerHost

        updatedUtc     = (Get-Date).ToUniversalTime().ToString('s') + 'Z'

    }

    ($o | ConvertTo-Json -Compress) | Set-Content -LiteralPath $p -Encoding UTF8 -Force

}



function Get-EmpireLanCredUser {

    $u = [Environment]::GetEnvironmentVariable('EMPIRE_LAN_USER', 'Machine')

    if (-not $u) { $u = [Environment]::GetEnvironmentVariable('EMPIRE_LAN_USER', 'User') }

    return $u

}



function Get-EmpireLanCredPass {

    $p = [Environment]::GetEnvironmentVariable('EMPIRE_LAN_PASS', 'Machine')

    if (-not $p) { $p = [Environment]::GetEnvironmentVariable('EMPIRE_LAN_PASS', 'User') }

    return $p

}



function Test-WorkerSmbSession {

    param(

        [string]$Server,

        [string]$Share,

        [string]$User,

        [string]$Pass

    )

    if ([string]::IsNullOrWhiteSpace($User)) { $User = Get-EmpireLanCredUser }

    if ($null -eq $Pass -or $Pass -eq '') { $Pass = Get-EmpireLanCredPass }

    $nr = Get-CmdNulQuietResolve

    $bs = [char]92

    $unc = $bs + $bs + $Server + $bs + $Share

    foreach ($L in @('Q', 'P', 'O', 'N', 'M')) {

        cmd.exe /c ('net use ' + $L + ': /delete /y ' + $nr)

        if (-not [string]::IsNullOrWhiteSpace($User)) {

            $uEsc = $User -replace '"', '""'

            $pEsc = if ($null -ne $Pass -and $Pass -ne '') { ($Pass.ToString()) -replace '"', '""' } else { '' }

            $mapE = 'net use ' + $L + ': ' + $unc + ' /user:"' + $uEsc + '" "' + $pEsc + '" ' + $nr

            cmd.exe /c $mapE

        } else {

            cmd.exe /c ('net use ' + $L + ': ' + $unc + ' ' + $nr)

        }

        $ok = ($LASTEXITCODE -eq 0) -and (Test-Path -LiteralPath ($L + ':\'))

        cmd.exe /c ('net use ' + $L + ': /delete /y ' + $nr)

        if ($ok) { return $true }

    }

    return $false

}



function Test-WorkerScriptsUncParentReachable {

    param([string]$ScriptsUncFull)

    try {

        $par = Split-Path -Parent $ScriptsUncFull

        return [bool](Test-Path -LiteralPath $par)

    } catch { return $false }

}



function Test-WorkerMeshScriptsUncViable {

    param([string]$ScriptsUncFull)

    if ([string]::IsNullOrWhiteSpace($ScriptsUncFull)) { return $false }

    try {

        if (Test-WorkerScriptsUncParentReachable -ScriptsUncFull $ScriptsUncFull) { return $true }

        return [bool](Test-Path -LiteralPath $ScriptsUncFull.Trim())

    } catch { return $false }

}



function Test-WorkerScriptsUncResolvable {

    param([string]$ScriptsUncFull)

    if (Test-WorkerScriptsUncParentReachable -ScriptsUncFull $ScriptsUncFull) { return $true }

    try {

        $scriptsUncFull = $ScriptsUncFull

        $parent = Split-Path -Parent $scriptsUncFull

        $shareRoot = Split-Path -Parent $parent

        return [bool](Test-Path -LiteralPath $shareRoot)

    } catch { return $false }

}



function Resolve-PushMeshScriptsUnc {

    param(

        [string]$WorkerIp,

        [string]$WorkerHost,

        [string]$HealthDir

    )

    $envUnc = $env:EMPIRE_WORKER_SMB_SCRIPTS_UNC

    if (-not [string]::IsNullOrWhiteSpace($envUnc)) {

        $t = $envUnc.Trim()

        if (Test-WorkerMeshScriptsUncViable -ScriptsUncFull $t) { return $t }

    }

    $cands = Get-EmpireWorkerSmbCandidates -HealthDir $HealthDir

    $cache = Read-MeshResolutionCache -HealthDir $HealthDir

    if ($cache -and $cache.workerIp -eq $WorkerIp -and $cache.workerShare) {

        $ssp = if ($cache.scriptsSubPath) { [string]$cache.scriptsSubPath } else { 'Scripts' }

        $probeForCache = Get-EmpireWorkerSmbProbeServers -WorkerIp $WorkerIp -WorkerHost $WorkerHost -HealthDir $HealthDir

        foreach ($srvC in $probeForCache) {

            if ([string]::IsNullOrWhiteSpace($srvC)) { continue }

            $uncTry = Join-WorkerServerShareUnc -Server $srvC -Share ([string]$cache.workerShare) -SubPath $ssp

            if (Test-WorkerMeshScriptsUncViable -ScriptsUncFull $uncTry) { return $uncTry }

        }

    }

    $servers = Get-EmpireWorkerSmbProbeServers -WorkerIp $WorkerIp -WorkerHost $WorkerHost -HealthDir $HealthDir

    foreach ($srv in $servers) {

        if ([string]::IsNullOrWhiteSpace($srv)) { continue }

        foreach ($c in $cands) {

            $unc = Join-WorkerServerShareUnc -Server $srv -Share $c.share -SubPath $c.scriptsSubPath

            if (Test-WorkerMeshScriptsUncViable -ScriptsUncFull $unc) {

                if ($srv -eq $WorkerIp) {

                    Save-MeshResolutionCache -HealthDir $HealthDir -WorkerIp $WorkerIp -WorkerShare $c.share -ScriptsSubPath $c.scriptsSubPath -WorkerHost $WorkerHost

                }

                return $unc

            }

        }

    }

    return $null

}


