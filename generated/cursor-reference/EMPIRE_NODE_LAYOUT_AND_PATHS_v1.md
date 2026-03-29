# Empire node layout & paths — agreed placement v1

**Audience:** VMS + all agents. **Memory aid:** VMS uses **HP** as the **only** human audit surface for “where are all Cursor workspaces?” — **`C:\Empire\generated\DISCOVERED_CURSOR_PATHS.txt`**.

---

## Governance (Alzheimer’s-safe)

- **VMS does not hand-create** new Empire-scoped projects on disk. **Ask an agent** to scaffold under the correct **`C:\Empire_<NODE_ID>\`** (or `C:\Empire` on HP) so naming, `node_bus`, and discovery stay consistent.
- **Each node** may keep a **local** `generated\DISCOVERED_CURSOR_PATHS_LOCAL.txt`; **HP merges** into the **canonical** file via **`generated\cursor_paths_merge_sources.txt`** + `Sync_Discovered_Cursor_Paths.py`.

---

## Browser_Operator

- **Canonical 24/7 runtime:** **AI_X1** — `C:\Empire_AI_X1\Browser_Operator` (see **`HP_SafeDelete_Checklist_AI_X1_Cutover.ps1`**).
- **HP:** optional **mirror/archive** only if you want a cold copy; **not required** for automation. Cutover checklist assumed migration **to AI_X1** as primary.

---

## Project_Openclaw

- **Not** an active HP Cursor project root for 24/7 work — **canonical:** **`C:\Empire_AI_X1\Project_Openclaw`**.
- Remove HP duplicate from routine workflows once AI_X1 copy is verified (safe-delete checklist).

---

## Projects_schedular

| Node | Path | Note |
|------|------|------|
| **HP** | `C:\Empire\Projects_schedular` | No suffix (control authoring). |
| **AI_X1** | `C:\Empire_AI_X1\Projects_schedular_AI_X1` | **Suffixed** for human audit (`_<NODE_ID>`). |

- **Mirror:** `Scripts\Mirror_Projects_schedular_To_AI_X1.ps1` → destination folder **`Projects_schedular_AI_X1`** by default.
- **Rename on AI_X1:** from HP run `Scripts\Queue_AI_X1_Rename_Projects_Schedular_Admin.ps1` (queues AdminExecutor). On AI_X1 locally use **`C:\Empire_AI_X1\Scripts\Rename_AI_X1_Projects_Schedular_To_Suffixed.ps1`** only — not `C:\Empire\Scripts\...` on that PC.
- **Future node N:** `Projects_schedular_<N>` on that node.

---

## Project_LAN_Connections_Nodes (mesh runbook workspaces)

| Node | Path | Note |
|------|------|------|
| **HP** | `C:\Empire\Project_LAN_Connections_Nodes_HP` | Dedicated Cursor root for LAN/SMB mesh docs + `generated_Project_LAN_Connections_Nodes_HP\`. |
| **AI_X1** | `C:\Empire_AI_X1\Project_LAN_Connections_Nodes_AI_X1` (recommended) | Same purpose; **`Project_` prefix** + **`_AI_X1`** suffix so the name is not confused with HP’s folder. |

- **Scripts (canonical):** always `C:\Empire\Scripts\` on HP; workers receive updates via Push / git per mesh playbook.
- **Naming convention (new multi-node workspaces):** use a clear **project prefix** (e.g. `Project_`) and **node suffix** (`_HP`, `_AI_X1`, …) when the same logical project exists on more than one machine.
- **Renames / moves:** follow `C:\Empire\.cursor\rules\core-empire-workspace-rename-relocate-discipline.mdc` and re-run `Sync_Discovered_Cursor_Paths.py`.

---

## Cursor rules — hybrid (“Starlink” model)

- **CORE (full rules):** `C:\Empire\.cursor\rules\` on HP.
- **Satellite workspaces** (other discovered roots): default **`Sync_Empire_Shared_Cursor_Rules.ps1 -Strategy Auto`** copies **one** stub **`core-empire-pointer-to-canonical.mdc`**; full 30+ files only for roots listed in **`generated\cursor_paths_full_rules_roots.txt`** (default: `C:\Empire` only).
- **Legacy full blast:** `-Strategy Full` (thousands of copies — use sparingly).
- **Worker node** (e.g. AI_X1): after `git pull` or Push, keep **full** set under **`C:\Empire_AI_X1\.cursor\rules`** if that root is opened as primary — add **`Z:\`** or **`C:\Empire_AI_X1`** to **`cursor_paths_full_rules_roots.txt`** on HP if you want Full sync pushed to the mapped share from HP.

---

## Discovery pipeline (auto)

1. **HP heartbeat** runs **`Sync_Discovered_Cursor_Paths.py`** ≤ every 24h (`EMPIRE_CURSOR_PATHS_SYNC_HOURS`).
2. **Workers** run **`Publish_Local_Discovered_Cursor_Paths.ps1`** after new projects (or nightly task).
3. **HP** merges **`DISCOVERED_CURSOR_PATHS_LOCAL.txt`** from reachable paths in **`cursor_paths_merge_sources.txt`**.
4. **Rules sync** on HP: **`Sync_Empire_Shared_Cursor_Rules.ps1`** (default **Auto**).

---

## Larger picture (autonomy)

Strategic direction letter (Round 1): **`MyGPT_modification suggestions\Later_Empire\Full Automation Architecture_p1_28Mar26.txt`** — local-first, nomadic, self-heal, cost discipline; mesh + Git are **infrastructure**, not the full autonomy stack.

---

**Rev:** v1.2 — 2026-03-29 — `Project_LAN_Connections_Nodes_HP` rename + rename/relocate discipline pointer.
