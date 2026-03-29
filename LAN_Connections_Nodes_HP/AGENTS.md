# LAN_Connections_Nodes_HP — Agent brief

**Root:** `C:\Empire\LAN_Connections_Nodes_HP`  
**Node:** HP (control plane). **Sibling naming on workers:** `LAN_Connections_Nodes_AI_X1` (or `*_AI_X1`) to avoid confusion.

## Authority

- **Automation source of truth:** `C:\Empire\Scripts\` (LAN/mesh scripts live there; do not duplicate long-term).
- **Empire-wide doctrine:** `C:\Empire\.cursor\rules\`, `C:\Empire\generated\cursor-reference\`, `C:\Empire\generated\OPERATING_CONTRACT.md`.
- **This folder:** Runbooks, notes, and **project-local** outputs under `generated_LAN_Connections_Nodes_HP\`.

## Must-read for LAN work

- `README.md` (this project) — full mesh runbook.
- `C:\Empire\generated\cursor-reference\EMPIRE_MESH_LAN_PLANS_ABCD_v1.md`
- `C:\Empire\generated\cursor-reference\EMPIRE_NODE_LAYOUT_AND_PATHS_v1.md`
- Rules: `core-empire-multi-node-mirror-register.mdc`, `core-empire-adhoc-vs-constitution.mdc`, `core-empire-multi-node-all-nodes-memory.mdc`

## generated (this project)

See `generated_LAN_Connections_Nodes_HP\README.md`. Do not put this project’s local logs in `C:\Empire\generated\`.

## Discovery

After scaffolding, run on HP: `python C:\Empire\Scripts\Sync_Discovered_Cursor_Paths.py` so this root appears in `C:\Empire\generated\DISCOVERED_CURSOR_PATHS.txt` (requires substantive `.cursor` files).
