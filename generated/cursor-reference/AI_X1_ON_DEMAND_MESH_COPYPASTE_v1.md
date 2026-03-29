# AI_X1 — copy/paste: on-demand mesh + scheduled Y: (worker)

Run in **Administrator PowerShell on AI_X1**.

## 1) One-time (if not already registered)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Empire_AI_X1\Scripts\Register_Empire_LAN_Dynamic_Map_Scheduled_From_AI_X1.ps1"
```

Machine env on AI_X1: **`EMPIRE_LAN_HP_USER`**, **`EMPIRE_LAN_HP_PASS`** (HP share credentials).

## 2) On-demand before any job that needs HP / Y:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Empire_AI_X1\Scripts\Empire_Invoke_AI_X1_Mesh_Heal_OnDemand.ps1"
```

If that script is missing, pull mesh from HP first (from HP, Admin):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Empire\Scripts\Push_Empire_Mesh_Scripts_To_AI_X1.ps1"
```

## 3) Full heal entry point (same as scheduled inner)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Empire_AI_X1\Scripts\Empire_Node_Mesh_Heal.ps1" -NodeRole AI_X1
```
