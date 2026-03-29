<#
  RUN ON HP when Z: is broken - pushes mesh .ps1 to AI_X1 over UNC (admin share).
  Default UNC uses worker IP from C:\Empire\generated\health\empire_lan_worker_last_ip.txt or -WorkerIp.
  ASCII-only strings (avoid parser breakage on some editors/encodings).
#>
param(
    [string]$WorkerIp = "",
    [string]$WorkerHost = "AI-X1",
    [string]$UncScripts = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Empire_Worker_Smb_Resolve.ps1")
$EmpireRootPush = Split-Path -Parent $PSScriptRoot
$HealthDirPush = Join-Path $EmpireRootPush "generated\health"
$src = "C:\Empire\Scripts"
$files = @(
    "Set_Empire_LAN_Link.ps1",
    "Set_Empire_LAN_Link_To_HP.ps1",
    "Empire_Publish_Worker_LanIp_ForHp.ps1",
    "Empire_Publish_Hp_LanIp_ForAiX1.ps1",
    "Register_Empire_AI_X1_PublishIp_Logon_Task.ps1",
    "Register_Empire_AI_X1_MapY_Logon_Task.ps1",
    "Register_Empire_HP_PublishHpIp_Logon_Task.ps1",
    "Empire_Node_Mesh_Heal.ps1",
    "Empire_Vpn_May_Block_Lan.ps1",
    "Publish_Empire_Lan_Heartbeat_To_NodeBus.ps1",
    "Push_Empire_Mesh_Scripts_To_AI_X1.ps1",
    "Pull_Empire_Mesh_Scripts_From_AI_X1.ps1",
    "Sync_Empire_Mesh_Scripts_To_HP_From_AI_X1.ps1",
    "Diagnose_Empire_HP_To_AI_X1_Lan.ps1",
    "Empire_Worker_Smb_Resolve.ps1",
    "Empire_LAN_Share_Resolve.ps1",
    "Sync_Discovered_Cursor_Paths.py",
    "Publish_Local_Discovered_Cursor_Paths.ps1",
    "Rename_AI_X1_Projects_Schedular_To_Suffixed.ps1",
    "Sync_Empire_Shared_Cursor_Rules.ps1",
    "Queue_AI_X1_Rename_Projects_Schedular_Admin.ps1",
    "Empire_Invoke_HP_Mesh_Heal_OnDemand.ps1",
    "Empire_Invoke_AI_X1_Mesh_Heal_OnDemand.ps1",
    "Empire_Ensure_HP_EmpireAiX1_Junction.ps1",
    "Submit_AI_X1_Admin_RunScript.ps1",
    "Publish_AI_X1_Discovered_CursorPaths_From_HP.ps1",
    "Empire_Mirror_Register.ps1",
    "Ensure_Empire_Worker_Node_Skeleton.ps1"
)

$hint = "C:\Empire\generated\health\empire_lan_worker_last_ip.txt"
if (-not $UncScripts) {
    if (-not $WorkerIp) {
        if (Test-Path -LiteralPath $hint) {
            $WorkerIp = (Get-Content -LiteralPath $hint -First 1).Trim()
        }
    }
    if ($WorkerIp -and (Test-EmpireIpIsThisMachine -Ip $WorkerIp)) {
        Write-Host "[Empire-Push] FAIL: $hint lists THIS PC ($WorkerIp), not AI_X1. On AI_X1 run ipconfig; set line 1 to that IPv4, or: -WorkerIp <AI_X1_IP>" -ForegroundColor Yellow
        exit 1
    }
    if (-not $WorkerIp) {
        Write-Host "[Empire-Push] FAIL: Set -WorkerIp or create $hint (one IPv4 line) or -UncScripts." -ForegroundColor Yellow
        exit 1
    }
    if (-not (Test-Path -LiteralPath $HealthDirPush)) {
        New-Item -ItemType Directory -Path $HealthDirPush -Force | Out-Null
    }
    $resolved = Resolve-PushMeshScriptsUnc -WorkerIp $WorkerIp -WorkerHost $WorkerHost -HealthDir $HealthDirPush
    if (-not $resolved) {
        Write-Host "[Empire-Push] FAIL: No candidate Scripts UNC reachable. Diagnose + edit EMPIRE_WORKER_SMB_CANDIDATES.json or set EMPIRE_WORKER_SMB_SCRIPTS_UNC." -ForegroundColor Yellow
        Write-Host ("[Empire-Push] Run: powershell -NoProfile -ExecutionPolicy Bypass -File `"C:\Empire\Scripts\Diagnose_Empire_HP_To_AI_X1_Lan.ps1`" -WorkerIp " + $WorkerIp) -ForegroundColor Yellow
        exit 1
    }
    $UncScripts = $resolved
}

$parent = Split-Path -Parent $UncScripts
if (-not (Test-Path -LiteralPath $parent)) {
    Write-Host "[Empire-Push] FAIL: Cannot reach $parent (worker off, VPN, firewall 445, or share layout)." -ForegroundColor Yellow
    Write-Host ("[Empire-Push] Run: powershell -NoProfile -ExecutionPolicy Bypass -File `"C:\Empire\Scripts\Diagnose_Empire_HP_To_AI_X1_Lan.ps1`" -WorkerIp " + $WorkerIp) -ForegroundColor Yellow
    exit 1
}
if (-not (Test-Path -LiteralPath $UncScripts)) {
    New-Item -ItemType Directory -Path $UncScripts -Force | Out-Null
}

foreach ($f in $files) {
    $fp = Join-Path $src $f
    if (-not (Test-Path -LiteralPath $fp)) {
        Write-Host "SKIP missing: $fp" -ForegroundColor Yellow
        continue
    }
    Copy-Item -LiteralPath $fp -Destination (Join-Path $UncScripts $f) -Force
    Write-Host "OK: $f"
}
$bundleSrc = Join-Path $src "Empire_Worker_Scaffold_Assets"
if (Test-Path -LiteralPath $bundleSrc) {
    $bundleDst = Join-Path $UncScripts "Empire_Worker_Scaffold_Assets"
    Copy-Item -LiteralPath $bundleSrc -Destination $bundleDst -Recurse -Force
    Write-Host "OK: Empire_Worker_Scaffold_Assets (recursive)"
}
$regLib = Join-Path $PSScriptRoot "Empire_Mirror_Register.ps1"
if (Test-Path -LiteralPath $regLib) {
    try {
        . $regLib
        Add-EmpireMirrorRegisterEntry -SourceNode "HP" -Action "MESH_PUSH_SCRIPTS" -Target $UncScripts -Detail ("Push_Empire_Mesh_Scripts_To_AI_X1 count=" + $files.Count)
    } catch {}
}
Write-Host "Done -> $UncScripts"
$ipToSave = $WorkerIp
if (-not $ipToSave -and $UncScripts -match '^\\\\(\d{1,3}(?:\.\d{1,3}){3})\\') {
    $ipToSave = $Matches[1]
}
if ($ipToSave) {
    $hintDir = Split-Path -Parent $hint
    if (-not (Test-Path -LiteralPath $hintDir)) {
        New-Item -ItemType Directory -Path $hintDir -Force | Out-Null
    }
    Set-Content -LiteralPath $hint -Value $ipToSave.Trim() -Encoding ascii
    Write-Host "OK: worker IP hint -> $hint"
}
