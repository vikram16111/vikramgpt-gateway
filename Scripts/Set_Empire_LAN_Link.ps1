param(
    [string]$WorkerIp = "",
    [string]$WorkerHost = "AI-X1",
    [string]$ShareName = "",
    [string]$DriveLetter = "Z",
    [string]$Username = "AI-X1\VMS AI X1",
    [string]$Password = "StrongPass123!",
    [ValidateRange(200,30000)][int]$ConnectTimeoutMs = 1500,
    [switch]$ScheduleWatcher,
    # Pin this machine's current 192.168.* address to a hosts alias (default HP-SPECTRE). Avoids stale fixed IPs after DHCP renew.
    [string]$SelfLanAlias = "HP-SPECTRE",
    [switch]$SkipSelfLanHostsPin,
    [string[]]$LastKnownIpFiles = @(),
    # Empty = auto: <Empire root next to Scripts>\generated\health\empire_lan_worker_last_ip.txt
    [string]$PersistLastIpTo = "",
    [string]$AlsoPersistLastIpTo = "",
    [string]$TryTheseIpsFirst = "",
    # Off by default: neighbor sweep + /254 scan can take many minutes (looks "stuck"). Set -AllowSlowLanScan or EMPIRE_LAN_ALLOW_SLOW_LAN_SCAN=1 to enable.
    [switch]$AllowSlowLanScan,
    # If -WorkerIp is set but SMB probe fails, run full auto-resolve (DNS, files, EMPIRE_LAN_TRY_IPS, optional slow scan) excluding that IP. Default ON.
    [switch]$NoFallbackAutoResolve
)

$ErrorActionPreference = "Stop"

$script:EmpireLanDoSlowScan = $AllowSlowLanScan
if (-not $script:EmpireLanDoSlowScan -and ($env:EMPIRE_LAN_ALLOW_SLOW_LAN_SCAN -eq '1')) {
    $script:EmpireLanDoSlowScan = $true
}

$EmpireRoot = Split-Path -Parent $PSScriptRoot
$EmpireHealthDir = Join-Path $EmpireRoot "generated\health"
if (-not (Test-Path -LiteralPath $EmpireHealthDir)) {
    New-Item -ItemType Directory -Path $EmpireHealthDir -Force | Out-Null
}
if ([string]::IsNullOrWhiteSpace($PersistLastIpTo)) {
    $PersistLastIpTo = Join-Path $EmpireHealthDir "empire_lan_worker_last_ip.txt"
}

$EmpireSmbResolveLib = Join-Path $PSScriptRoot "Empire_Worker_Smb_Resolve.ps1"
if (Test-Path -LiteralPath $EmpireSmbResolveLib) { . $EmpireSmbResolveLib }

$uM = [Environment]::GetEnvironmentVariable('EMPIRE_LAN_USER', 'Machine')
$uU = [Environment]::GetEnvironmentVariable('EMPIRE_LAN_USER', 'User')
if (-not [string]::IsNullOrWhiteSpace($uM)) { $Username = $uM }
if (-not [string]::IsNullOrWhiteSpace($uU)) { $Username = $uU }
$pM = [Environment]::GetEnvironmentVariable('EMPIRE_LAN_PASS', 'Machine')
$pU = [Environment]::GetEnvironmentVariable('EMPIRE_LAN_PASS', 'User')
if (-not [string]::IsNullOrWhiteSpace($pM)) { $Password = $pM }
if (-not [string]::IsNullOrWhiteSpace($pU)) { $Password = $pU }

function Write-Step {
    param([string]$Message)
    Write-Host "[Empire-LAN] $Message"
}

# Build >nul 2>&1 without 2> in one PS token; avoid [string]::Concat(char,...) (PS5 overload -> ArgumentNullException).
function Get-CmdNulQuiet {
    return '>' + 'nul 2' + '>' + '&1'
}

function Invoke-NetUseDeleteQuiet([string]$DriveLetter) {
    $nr = Get-CmdNulQuiet
    cmd.exe /c ('net use ' + $DriveLetter + ': /delete /y ' + $nr)
}

function Ensure-PrivateNetwork {
    $profiles = Get-NetConnectionProfile
    foreach ($p in $profiles) {
        if ($p.NetworkCategory -ne "Private") {
            Write-Step "Setting network profile '$($p.Name)' to Private"
            try {
                Set-NetConnectionProfile -InterfaceIndex $p.InterfaceIndex -NetworkCategory Private
            } catch {
                Write-Step "WARN: Could not set network profile to Private (non-admin or policy). Continuing."
            }
        }
    }
}

function Ensure-DiscoveryAndSharing {
    Write-Step "Ensuring discovery and SMB services are active"
    $services = @("FDResPub", "fdPHost", "LanmanServer", "LanmanWorkstation")
    foreach ($svc in $services) {
        Set-Service -Name $svc -StartupType Automatic -ErrorAction SilentlyContinue
        Start-Service -Name $svc -ErrorAction SilentlyContinue
    }

    Write-Step "Enabling firewall groups for discovery and sharing"
    cmd.exe /c "netsh advfirewall firewall set rule group=`"Network Discovery`" new enable=Yes" | Out-Null
    cmd.exe /c "netsh advfirewall firewall set rule group=`"File and Printer Sharing`" new enable=Yes" | Out-Null
}

function Ensure-HostsEntry {
    param(
        [string]$Ip,
        [string]$HostAlias
    )
    $hostsPath = "$env:SystemRoot\System32\drivers\etc\hosts"
    $entry = "$Ip $HostAlias"
    $content = Get-Content -Path $hostsPath -ErrorAction SilentlyContinue
    if (-not ($content -match "^\s*$([regex]::Escape($Ip))\s+$([regex]::Escape($HostAlias))\s*$")) {
        Write-Step "Adding hosts entry: $entry"
        Add-Content -Path $hostsPath -Value $entry
    }
}

function Read-LastKnownIpFromFiles {
    param([string[]]$Paths)
    foreach ($p in $Paths) {
        if ([string]::IsNullOrWhiteSpace($p)) { continue }
        if (-not (Test-Path -LiteralPath $p)) { continue }
        foreach ($line in Get-Content -LiteralPath $p -ErrorAction SilentlyContinue) {
            $t = $line.Trim()
            if ($t -match '^\d{1,3}(\.\d{1,3}){3}$') {
                return $t
            }
        }
    }
    return $null
}

function Get-Local192SubnetPrefix {
    try {
        $me = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
            Where-Object { $_.IPAddress -match '^192\.168\.\d+\.\d+$' -and $_.PrefixOrigin -ne 'WellKnown' } |
            Select-Object -First 1 -ExpandProperty IPAddress
        if (-not $me) { return $null }
        $parts = $me.Split('.')
        return "$($parts[0]).$($parts[1]).$($parts[2])"
    } catch {
        return $null
    }
}

function Test-WorkerEmpireShare {
    param(
        [string]$Ip,
        [string]$Share,
        [string]$User,
        [string]$Pass
    )
    $nr = Get-CmdNulQuiet
    $bs = [char]92
    $unc = $bs + $bs + $Ip + $bs + $Share
    foreach ($L in @('Y', 'T', 'U', 'V', 'W', 'X', 'R', 'S')) {
        Invoke-NetUseDeleteQuiet -DriveLetter $L
        if (-not [string]::IsNullOrWhiteSpace($User)) {
            $uEsc = $User -replace '"', '""'
            $pEsc = $Pass -replace '"', '""'
            $mapE = 'net use ' + $L + ': ' + $unc + ' /user:"' + $uEsc + '" "' + $pEsc + '" ' + $nr
            cmd.exe /c $mapE
            if ($LASTEXITCODE -eq 0 -and (Test-Path "${L}:\")) {
                Invoke-NetUseDeleteQuiet -DriveLetter $L
                return $true
            }
            Invoke-NetUseDeleteQuiet -DriveLetter $L
        }
        $mapI = 'net use ' + $L + ': ' + $unc + ' ' + $nr
        cmd.exe /c $mapI
        if ($LASTEXITCODE -eq 0 -and (Test-Path "${L}:\")) {
            Invoke-NetUseDeleteQuiet -DriveLetter $L
            return $true
        }
        Invoke-NetUseDeleteQuiet -DriveLetter $L
    }
    return $false
}

function Test-WorkerSharesForIp {
    param([string]$Ip, [string[]]$ShareNames, [string]$User, [string]$Pass)
    if (Test-EmpireIpIsThisMachine -Ip $Ip) {
        Write-Step "SKIP: $Ip is this PC's own IPv4 (not AI_X1). empire_lan_worker_last_ip.txt or discovery pointed at HP by mistake."
        return $null
    }
    Write-Step "SMB map probes for \\$Ip\<share> (temporary drive letters) - not frozen; each net use can take ~10-45s if the host is slow"
    foreach ($s in $ShareNames) {
        if ([string]::IsNullOrWhiteSpace($s)) { continue }
        Write-Step "  trying share name: $($s.Trim())"
        if (Test-WorkerEmpireShare -Ip $Ip -Share $s.Trim() -User $User -Pass $Pass) { return $s.Trim() }
    }
    return $null
}

function Resolve-WorkerIpDynamic {
    param(
        [string]$HostName,
        [int]$TimeoutMs,
        [string[]]$ExtraLastKnownFiles,
        [string]$CommaSeparatedIps,
        [string[]]$ShareNames,
        [string]$Username,
        [string]$Password,
        [string]$EmpireRootForHealth
    )
    $tryFiles = @()
    if ($env:EMPIRE_LAN_WORKER_LAST_IP_FILE) { $tryFiles += $env:EMPIRE_LAN_WORKER_LAST_IP_FILE.Trim() }
    if ($EmpireRootForHealth) {
        $tryFiles += (Join-Path $EmpireRootForHealth "generated\health\empire_lan_worker_last_ip.txt")
    }
    $gd = "G:\My Drive\Local_C\Empire\generated\health\empire_lan_worker_last_ip.txt"
    if (Test-Path -LiteralPath (Split-Path -Parent $gd)) { $tryFiles += $gd }
    foreach ($x in $ExtraLastKnownFiles) { if ($x) { $tryFiles += $x } }

    $tryFiles = $tryFiles | Select-Object -Unique

    $fromFile = Read-LastKnownIpFromFiles -Paths $tryFiles
    if ($fromFile) {
        if ($excluded.ContainsKey($fromFile)) {
            Write-Step "SKIP: last-known IP $fromFile excluded (already probed / unreachable)."
        } elseif (Test-EmpireIpIsThisMachine -Ip $fromFile) {
            Write-Step "SKIP: last-known IP file lists $fromFile = THIS PC (wrong). Delete or replace with AI_X1's IPv4 (ipconfig on worker)."
        } else {
            Write-Step "Trying last-known IP from file: $fromFile (SMB candidate shares)"
            if (-not (Test-TcpPort -TargetAddress $fromFile -Port 445 -TimeoutMs $TimeoutMs)) {
                Write-Step "SKIP: $($fromFile):445 not reachable within ${TimeoutMs}ms (offline/firewall) - not stuck; trying next method"
            } else {
                $won = Test-WorkerSharesForIp -Ip $fromFile -ShareNames $ShareNames -User $Username -Pass $Password
                if ($won) { $script:ResolvedWorkerShareName = $won; return $fromFile }
            }
        }
    }

    foreach ($raw in @($CommaSeparatedIps -split ',')) {
        $ip = $raw.Trim()
        if ($ip -match '^\d{1,3}(\.\d{1,3}){3}$') {
            if ($excluded.ContainsKey($ip)) { continue }
            if (Test-EmpireIpIsThisMachine -Ip $ip) {
                Write-Step "SKIP fallback IP $ip (this PC, not worker)"
                continue
            }
            Write-Step "Trying fallback IP: $ip"
            if (-not (Test-TcpPort -TargetAddress $ip -Port 445 -TimeoutMs $TimeoutMs)) {
                Write-Step "SKIP: $($ip):445 not reachable within ${TimeoutMs}ms"
            } else {
                $won = Test-WorkerSharesForIp -Ip $ip -ShareNames $ShareNames -User $Username -Pass $Password
                if ($won) { $script:ResolvedWorkerShareName = $won; return $ip }
            }
        }
    }

    $names = @($HostName, "AI_X1", "AI-X1", "Empire-AI-X1", "EMPIRE-AI-X1") | Select-Object -Unique
    foreach ($hn in $names) {
        try {
            $dns = Resolve-DnsName $hn -ErrorAction Stop | Where-Object { $_.IPAddress } | Select-Object -First 1
            if ($dns -and $dns.IPAddress) {
                $ip = [string]$dns.IPAddress
                if ($excluded.ContainsKey($ip)) { continue }
                if (Test-EmpireIpIsThisMachine -Ip $ip) { continue }
                if (Test-TcpPort -TargetAddress $ip -Port 445 -TimeoutMs $TimeoutMs) {
                    $won = Test-WorkerSharesForIp -Ip $ip -ShareNames $ShareNames -User $Username -Pass $Password
                    if ($won) { $script:ResolvedWorkerShareName = $won; return $ip }
                }
            }
        } catch {}
    }

    if (-not $script:EmpireLanDoSlowScan) {
        Write-Step "SKIP slow LAN discovery (ARP neighbors + /24 scan). Fast path only. To enable: -AllowSlowLanScan or set Machine env EMPIRE_LAN_ALLOW_SLOW_LAN_SCAN=1"
        return $null
    }

    $candidates = @()
    try {
        $candidates = Get-NetNeighbor -AddressFamily IPv4 -ErrorAction SilentlyContinue |
            Where-Object { $_.IPAddress -like "192.168.*.*" } |
            Select-Object -ExpandProperty IPAddress -Unique
    } catch {}

    foreach ($ip in $candidates) {
        if ($excluded.ContainsKey($ip)) { continue }
        if (Test-EmpireIpIsThisMachine -Ip $ip) { continue }
        if (-not (Test-TcpPort -TargetAddress $ip -Port 445 -TimeoutMs $TimeoutMs)) { continue }
        $won = Test-WorkerSharesForIp -Ip $ip -ShareNames $ShareNames -User $Username -Pass $Password
        if ($won) { $script:ResolvedWorkerShareName = $won; return [string]$ip }
    }

    $prefix = Get-Local192SubnetPrefix
    if ($prefix) {
        Write-Step "Scanning $prefix.0/24 for SMB (candidate shares) - may take several minutes"
        foreach ($n in 1..254) {
            $ip = "$prefix.$n"
            if ($excluded.ContainsKey($ip)) { continue }
            if (Test-EmpireIpIsThisMachine -Ip $ip) { continue }
            if (-not (Test-TcpPort -TargetAddress $ip -Port 445 -TimeoutMs ([Math]::Min(400, $TimeoutMs)))) { continue }
            $won = Test-WorkerSharesForIp -Ip $ip -ShareNames $ShareNames -User $Username -Pass $Password
            if ($won) {
                Write-Step "Found share $won at $ip"
                $script:ResolvedWorkerShareName = $won
                return $ip
            }
        }
    }

    return $null
}

function Test-TcpPort {
    param(
        [Parameter(Mandatory=$true)][string]$TargetAddress,
        [Parameter(Mandatory=$true)][int]$Port,
        [ValidateRange(200,30000)][int]$TimeoutMs = 1500
    )
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $iar = $client.BeginConnect($TargetAddress, $Port, $null, $null)
        $ok = $iar.AsyncWaitHandle.WaitOne($TimeoutMs, $false)
        if (-not $ok) {
            try { $client.Close() } catch {}
            return $false
        }
        $client.EndConnect($iar)
        $client.Close()
        return $true
    } catch {
        try { $client.Close() } catch {}
        return $false
    }
}

function Map-WorkerDrive {
    param(
        [string]$Letter,
        [string]$Ip,
        [string]$HostAlias,
        [string]$Share,
        [string]$User = "",
        [string]$Pass = ""
    )
    $rootIp = "\\$Ip\$Share"
    $rootHost = "\\$HostAlias\$Share"

    # CredTarget must match the server part of RootUnc (IP vs hostname) or Windows may ignore cmdkey / mis-associate creds.
    function Invoke-NetUseMap {
        param([string]$RootUnc, [bool]$UseExplicit, [string]$CredTarget)
        Invoke-NetUseDeleteQuiet -DriveLetter $Letter
        $nr = Get-CmdNulQuiet
        if ($UseExplicit -and -not [string]::IsNullOrWhiteSpace($User)) {
            Write-Step "Trying map $RootUnc (saved SMB user)"
            if (-not [string]::IsNullOrWhiteSpace($CredTarget)) {
                cmd.exe /c ('cmdkey /delete:' + $CredTarget) 2>&1 | Out-Null
                $safeUser = $User -replace " ", "^ "
                cmd.exe /c ('cmdkey /add:' + $CredTarget + ' /user:' + $safeUser + ' /pass:' + $Pass) 2>&1 | Out-Null
            }
            $uEsc = $User -replace '"', '""'
            $pEsc = $Pass -replace '"', '""'
            $mapE = 'net use ' + $Letter + ': ' + $RootUnc + ' /user:"' + $uEsc + '" "' + $pEsc + '" /persistent:yes ' + $nr
            cmd.exe /c $mapE
            $code = $LASTEXITCODE
            if (-not [string]::IsNullOrWhiteSpace($CredTarget)) {
                cmd.exe /c ('cmdkey /delete:' + $CredTarget) 2>&1 | Out-Null
            }
            if ($code -eq 1223) { return 1223 }
            if ($code -eq 0 -and (Test-Path "$Letter`:\")) { return 0 }
        }
        Write-Step "Trying map $RootUnc (Windows sign-in)"
        $mapI = 'net use ' + $Letter + ': ' + $RootUnc + ' /persistent:yes ' + $nr
        cmd.exe /c $mapI
        if ($LASTEXITCODE -eq 1223) { return 1223 }
        if ($LASTEXITCODE -eq 0 -and (Test-Path "$Letter`:\")) { return 0 }
        return $LASTEXITCODE
    }

    Write-Step "Clearing old mapping for $Letter`:"
    Invoke-NetUseDeleteQuiet -DriveLetter $Letter

    $r = Invoke-NetUseMap -RootUnc $rootIp -UseExplicit $true -CredTarget $Ip
    if ($r -eq 0) {
        Write-Step "Mapped successfully via IP"
        return $true
    }
    if ($r -eq 1223) {
        Write-Host "[Empire-LAN] FAIL: SMB mapping canceled (1223). Allow saved creds or fix account." -ForegroundColor Yellow
        return $false
    }

    $r2 = Invoke-NetUseMap -RootUnc $rootHost -UseExplicit $true -CredTarget $HostAlias
    if ($r2 -eq 0) {
        Write-Step "Mapped successfully via hostname"
        return $true
    }
    if ($r2 -eq 1223) {
        Write-Host "[Empire-LAN] FAIL: SMB mapping canceled (1223). Allow saved creds or fix account." -ForegroundColor Yellow
        return $false
    }

    Write-Host "[Empire-LAN] FAIL: Could not map $Letter`: to $rootIp or $rootHost (exit $r2)." -ForegroundColor Yellow
    return $false
}

function Test-WorkerDriveRW {
    param([string]$Letter)
    $probe = "$Letter`:\_empire_lan_probe.txt"
    $stamp = (Get-Date).ToString("s")
    Set-Content -Path $probe -Value "LAN probe $stamp"
    if (-not (Test-Path $probe)) {
        throw "Write probe failed: $probe"
    }
    Remove-Item -Path $probe -Force
    if (Test-Path $probe) {
        throw "Delete probe failed: $probe"
    }
    Write-Step "Read/write/delete probe passed on $Letter`:"
}

function Register-WifiReconnectWatcher {
    param(
        [string]$ScriptPath
    )
    $taskNameLogon = "Empire_LAN_Reconnect_Watcher_Logon"
    $taskNameStartup = "Empire_LAN_Reconnect_Watcher_Startup"
    $triggerLogon = New-ScheduledTaskTrigger -AtLogOn
    $triggerStartup = New-ScheduledTaskTrigger -AtStartup
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$ScriptPath`""
    # SYSTEM + ServiceAccount = non-interactive (no popup windows).
    $principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
    try { Unregister-ScheduledTask -TaskName $taskNameLogon -Confirm:$false -ErrorAction SilentlyContinue | Out-Null } catch {}
    try { Unregister-ScheduledTask -TaskName $taskNameStartup -Confirm:$false -ErrorAction SilentlyContinue | Out-Null } catch {}
    Register-ScheduledTask -TaskName $taskNameLogon -Trigger $triggerLogon -Action $action -Principal $principal -Force | Out-Null
    Register-ScheduledTask -TaskName $taskNameStartup -Trigger $triggerStartup -Action $action -Principal $principal -Force | Out-Null
    Write-Step "Registered scheduled tasks: $taskNameLogon, $taskNameStartup"
}

$script:ResolvedWorkerShareName = $null
if (-not [string]::IsNullOrWhiteSpace($ShareName)) {
    $shareNamesForProbe = @($ShareName.Trim())
} else {
    $shareNamesForProbe = @((Get-EmpireWorkerSmbCandidates -HealthDir $EmpireHealthDir | ForEach-Object { $_.share }) | Select-Object -Unique)
    if ($shareNamesForProbe.Count -eq 0) { $shareNamesForProbe = @('Empire_AI_X1', 'C_AI_X1', 'Empire') }
}

Write-Step "Starting Empire LAN link setup"
Ensure-PrivateNetwork
Ensure-DiscoveryAndSharing
if (Remove-EmpireMappedDriveIfRemoteIsThisMachine -DriveLetter $DriveLetter) {
    Write-Step "Auto-heal: removed ${DriveLetter}: (SMB target was this PC, not AI_X1)."
}
if (-not [string]::IsNullOrWhiteSpace($WorkerIp)) {
    $wip = $WorkerIp.Trim()
    if (Test-EmpireIpIsThisMachine -Ip $wip) {
        Write-Host "[Empire-LAN] FAIL: -WorkerIp $wip is THIS PC's address, not AI_X1. Run ipconfig on AI_X1 and pass that IPv4." -ForegroundColor Yellow
        exit 1
    }
    Write-Step "Verifying $wip against SMB share candidates ($($shareNamesForProbe -join ', '))"
    $won = Test-WorkerSharesForIp -Ip $wip -ShareNames $shareNamesForProbe -User $Username -Pass $Password
    if (-not $won -and -not $NoFallbackAutoResolve) {
        Write-Step "Explicit worker $wip not reachable; auto-resolving (DNS, last-known files, EMPIRE_LAN_TRY_IPS; excluding $wip)..."
        $mergedFirst = $TryTheseIpsFirst
        if ($env:EMPIRE_LAN_TRY_IPS) {
            $mergedFirst = if ($mergedFirst) { "$env:EMPIRE_LAN_TRY_IPS,$mergedFirst" } else { $env:EMPIRE_LAN_TRY_IPS }
        }
        $WorkerIp = Resolve-WorkerIpDynamic -HostName $WorkerHost -TimeoutMs $ConnectTimeoutMs -ExtraLastKnownFiles $LastKnownIpFiles -CommaSeparatedIps $mergedFirst -ShareNames $shareNamesForProbe -Username $Username -Password $Password -EmpireRootForHealth $EmpireRoot -ExcludeIps @($wip)
        if ([string]::IsNullOrWhiteSpace($WorkerIp)) {
            Write-Host "[Empire-LAN] FAIL: Explicit worker $wip unreachable and auto-resolve found no other host with candidate shares. On AI_X1: enable File Sharing, share Empire_AI_X1 (or Empire\Empire_AI_X1), allow TCP 445 on Private network; set Machine env EMPIRE_LAN_USER / EMPIRE_LAN_PASS if needed. Or: -AllowSlowLanScan for subnet sweep." -ForegroundColor Yellow
            exit 1
        }
    } elseif (-not $won) {
        Write-Host "[Empire-LAN] FAIL: No candidate share reachable from this PC. Edit generated\health\EMPIRE_WORKER_SMB_CANDIDATES.json or pass -ShareName." -ForegroundColor Yellow
        exit 1
    } else {
        $script:ResolvedWorkerShareName = $won
        $WorkerIp = $wip
    }
} else {
    $mergedFirst = $TryTheseIpsFirst
    if ($env:EMPIRE_LAN_TRY_IPS) {
        $mergedFirst = if ($mergedFirst) { "$env:EMPIRE_LAN_TRY_IPS,$mergedFirst" } else { $env:EMPIRE_LAN_TRY_IPS }
    }
    $WorkerIp = Resolve-WorkerIpDynamic -HostName $WorkerHost -TimeoutMs $ConnectTimeoutMs -ExtraLastKnownFiles $LastKnownIpFiles -CommaSeparatedIps $mergedFirst -ShareNames $shareNamesForProbe -Username $Username -Password $Password -EmpireRootForHealth $EmpireRoot
}
if ([string]::IsNullOrWhiteSpace($WorkerIp)) {
    Write-Host "[Empire-LAN] FAIL: No worker found for candidate shares. Push/Diagnose, firewall 445, VPN, EMPIRE_WORKER_SMB_CANDIDATES.json." -ForegroundColor Yellow
    exit 1
}
$mapShare = $script:ResolvedWorkerShareName
if ([string]::IsNullOrWhiteSpace($mapShare)) {
    Write-Host "[Empire-LAN] FAIL: Resolved IP but not share name (internal). Re-run with -ShareName." -ForegroundColor Yellow
    exit 1
}
Write-Step "Worker IP: $WorkerIp (share: $mapShare)"
Ensure-HostsEntry -Ip $WorkerIp -HostAlias $WorkerHost
if (-not $SkipSelfLanHostsPin) {
    try {
        $selfIp = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
            Where-Object { $_.IPAddress -match '^192\.168\.' } |
            Select-Object -First 1 -ExpandProperty IPAddress
        if ($selfIp -and $SelfLanAlias) {
            Ensure-HostsEntry -Ip $selfIp -HostAlias $SelfLanAlias
            Write-Step "Pinned $SelfLanAlias -> $selfIp (this PC LAN)"
        }
    } catch {
        Write-Step "WARN: Could not auto-pin self LAN to hosts."
    }
}
ipconfig /flushdns | Out-Null
nbtstat -R | Out-Null
if (-not (Map-WorkerDrive -Letter $DriveLetter -Ip $WorkerIp -HostAlias $WorkerHost -Share $mapShare -User $Username -Pass $Password)) {
    exit 1
}
try {
    $row = (Get-EmpireWorkerSmbCandidates -HealthDir $EmpireHealthDir) | Where-Object { $_.share -eq $mapShare } | Select-Object -First 1
    $ssp = if ($row -and $row.scriptsSubPath) { [string]$row.scriptsSubPath } else { "Scripts" }
    Save-MeshResolutionCache -HealthDir $EmpireHealthDir -WorkerIp $WorkerIp -WorkerShare $mapShare -ScriptsSubPath $ssp -WorkerHost $WorkerHost
} catch {}
try {
    Test-WorkerDriveRW -Letter $DriveLetter
} catch {
    Write-Step "WARN: Root write probe failed (expected on admin shares). Continuing."
}

if ($ScheduleWatcher) {
    Register-WifiReconnectWatcher -ScriptPath $PSCommandPath
}

try {
    if (Test-EmpireIpIsThisMachine -Ip $WorkerIp) {
        Write-Step "WARN: Not saving empire_lan_worker_last_ip.txt (resolved IP is this PC - misconfiguration)."
    } elseif (-not [string]::IsNullOrWhiteSpace($PersistLastIpTo)) {
        $dir = Split-Path -Parent $PersistLastIpTo
        if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
        Set-Content -LiteralPath $PersistLastIpTo -Value $WorkerIp -Encoding ascii
        Write-Step "Saved last worker IP for next boot: $PersistLastIpTo"
    }
    if (-not (Test-EmpireIpIsThisMachine -Ip $WorkerIp) -and -not [string]::IsNullOrWhiteSpace($AlsoPersistLastIpTo)) {
        $dir2 = Split-Path -Parent $AlsoPersistLastIpTo
        if (-not (Test-Path -LiteralPath $dir2)) { New-Item -ItemType Directory -Path $dir2 -Force | Out-Null }
        Set-Content -LiteralPath $AlsoPersistLastIpTo -Value $WorkerIp -Encoding ascii
        Write-Step "Mirrored last worker IP to: $AlsoPersistLastIpTo"
    }
} catch {
    Write-Step "WARN: Could not persist last worker IP ($($_.Exception.Message))"
}

Write-Step "Empire LAN link is healthy and stable."
