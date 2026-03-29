# Project_LAN_Connections_Nodes_AI_X1

**Node:** AI_X1 — LAN / SMB mesh runbook (mirror of HP `Project_LAN_Connections_Nodes_HP`).

**Paths:** Prefer local `C:\Empire_AI_X1\...`. When HP maps your share as **`Z:`**, HP may edit `Z:\Project_LAN_Connections_Nodes_AI_X1\` — same files.

## Quick commands (run on AI_X1)

- Mesh to HP: `Empire_Node_Mesh_Heal.ps1 -NodeRole AI_X1`
- Publish LAN IP for HP: `Empire_Publish_Worker_LanIp_ForHp.ps1`
- Git (if clone exists): `cd repos\vikramgpt-gateway` then `git pull`

## Full runbook

Copy the section list from HP `Project_LAN_Connections_Nodes_HP\README.md` — steps are the same; swap roles (you are worker, HP is control).

## generated

Use **`generated_Project_LAN_Connections_Nodes_AI_X1\`** for local notes only.

**Rev:** 2026-03-29 — scaffold from `Ensure_Empire_Worker_Node_Skeleton.ps1`.
