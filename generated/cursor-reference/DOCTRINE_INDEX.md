# Doctrine Index — Single Source of Truth

**Purpose:** List doctrine chapters and their scope. Chapters do **not** supersede each other; they are **simultaneously valid**. Use **CHAPTER** naming; use **RevN** only when a chapter is edited.  
**Precedence when chapters conflict:** (1) VMS instruction, (2) VikramGPT Orchestrator policy, (3) explicit precedence in this Index.

---

| Chapter file | Purpose | Scope | Owner layer | Precedence notes |
|--------------|---------|-------|-------------|------------------|
| `CH01A_Cursor_Execution_Doctrine__Strict.md` | Stricter execution doctrine (architect-aware; 65/35, Discovery, hardening). | Lambda refactors, registry, deploy flows when **stricter enforcement** is required. | T4/T5 | Use when task explicitly needs stricter enforcement. |
| `CH01B_Cursor_Execution_Doctrine__AI_Centric.md` | AI-centric execution doctrine (no architect-level behaviors; suggest don’t impose). | Default Cursor behavior: execution, refactor, code integrity. | T4/T5 | **Default active.** Use unless task requires CH01A. |
| `CH03_Data_Governance_Continuity__Execution_Rules.md` | Data, memory, logs, continuity — execution-layer rules from Section 6. | Data edits, migrations, namespaces, backup assumptions, memory behavior. | T4/T5 | Applies whenever touching data/memory/logs/continuity. |
| `CH04_Multi_MyGPT_Orchestration__Factory_Model.md` | Multi-MyGPT Factory model — one Empire spine, many domain brains; MyGPT-aware LambdaFactory; namespace isolation; parallel evolution. | LambdaFactory, registry, routing, namespaces, multi-MyGPT / multi-domain work. | T4/T5 | Parallel evolution only; extend not rewrite; adaptive compatibility. |
| `CH05_Local_Workspaces_and_CORE_Doctrine.md` | Relationship between `C:\Empire` CORE and other local workspaces; thin pointers and performance guidance; **VMS file path authority** (§7). | Any non-Empire project on C:/D:, workspace linkage, project/sub-project structure, **any user-supplied file path**. | T4/T5 | Use whenever working in local folders outside `C:\Empire` (e.g. `D:\Heyaru Belgium`). **When VMS gives a file path, apply CH01B §16 and CH05 §7 — mandatory.** |

---

## VMS file path authority (all agents, all projects)

When VMS provides an **explicit absolute file path**, every agent must **attempt a direct Read on that path** and must **not** contradict VMS based on directory/glob listing alone. Doctrine: **CH01B §16**, **CH05 §7**. No exception by project or workspace.

---

## How to use (Cursor)

1. **Refer to this Index first** when answering Empire questions.
2. **Select the relevant chapter by scope** (execution default = CH01B; data/memory/logs = also CH03; stricter = CH01A; LambdaFactory / multi-MyGPT = also CH04).
3. **Do not treat chapters as versions** — no “v2 supersedes v1”. Multiple chapters apply together by scope.

---

**Cross-ref:** `.cursor/rules/cursor-memory-first.mdc`, OPERATING_CONTRACT.md, STRATEGIC_MEMORY_UPGRADE_v1.md.

---

## Consolidation & recovery (Empire-wide)

| Doc | Purpose |
|-----|---------|
| `generated/CONSOLIDATION_EXECUTION_ROLLOUT_v1.md` | Phased rollout: discovery → analysis → design → waves → **CGAction v41** validation. |
| `generated/EMPIRE_PARENT_CHILD_PROJECT_SYNC_v1.md` | Parent `C:\Empire` ↔ child projects + `D:\` workspaces; learnings flow **up** to core. |
| `generated/EMPIRE_RECOVERY_CHATGPT_MAR3_18_AUDIT_v1.md` | Synthesis from 25 ChatGPT share extracts + letter; **not** sole authority. |
| `generated/DISCOVERED_CURSOR_PATHS.txt` | Auto-generated path index — run `Scripts/Sync_Discovered_Cursor_Paths.py` after new roots/projects. |
| `generated/CG_GDRIVE_UNIVERSAL_LAYOUT_AND_CONTEXT_v1.md` | Universal Drive layout + date/context naming + `cursor-replies-to-chatgpt` sidecars; Kundli = sample. |
| `generated/EMPIRE_RECOVERY_COMPLETENESS_CHECKLIST_v1.md` | Informal recovery % framing + checklist for “executable” vs bit-identical recovery. |
| `generated/EMPIRE_OCR_TIER_LADDER_v1.md` | L1–L5 (VMS) ↔ Empire engines / Textract / vision; 5 DTC backlog; full-stack “100% Empire” includes OCR coverage. |
| `generated/OPENCLAW_PROVIDER_PROFILES_AND_CHAT_COVERAGE_MAP_v1.md` | OpenClaw `*_profile` dirs, HP vs mini PC, Secrets Manager ref, 25 share runs vs MyGPT files vs `cursor-replies-to-chatgpt`. |
| `generated/MYGPT_SHARE_EXTRACT_CROSSREF_v1.md` | **Verified** MyGPT paste ↔ share URL (e.g. Gmail p4 ↔ `q_0006`); TG/WA profile migration; SSH/cloud pointers. |
| `generated/cursor-reference/EMPIRE_HP_AGENT_AI_X1_CAPABILITIES_v1.md` | **HP → AI_X1:** deploy Scripts via SMB, UNC read, `Invoke_Empire_Remote_Node_Command`; rule `core-empire-hp-agent-ai-x1-execution.mdc`. |
| `generated/cursor-reference/VMS_EMAIL_INTELLIGENCE_SCORING_SOURCE_OF_TRUTH_v1.md` | VMS email scoring + Outlook cleanup lists + pointer to `Full Automation Architecture_p2_28Mar26.txt`; agents own `generated/projects/email_ai_Empire_HP/`. |
| `generated/LAN_MESH_HP_AI_X1_NOMADIC_PLAYBOOK_v1.md` | HP ↔ AI_X1 LAN + **dual-side** recovery (`Set_Empire_LAN_Link_To_HP.ps1`, `Publish_Empire_Lan_Heartbeat_To_NodeBus.ps1`). |
| `generated/cursor-multi-node/EMPIRE_NODE_MESH_DRIVE_LETTERS_v1.md` | **`Z:` HP→worker**, **`Y:` worker→HP**; `Empire_Node_Mesh_Heal.ps1`; `Push_Empire_Mesh_Scripts_To_AI_X1.ps1`; PowerShell `$Host` fix (`-TargetAddress`). |
| `generated/cursor-reference/EMPIRE_MESH_LAN_PLANS_ABCD_v1.md` | **Mesh Plans A–D:** SMB auto-resolve, last-known IP + schedules, UNC override, LAN-down bridge; **bidirectional** heal; new node / router / reinstall; agents find IP. Rule: `core-empire-mesh-lan-plans-abcd.mdc`. |
| `generated/cursor-reference/EMPIRE_CLOUD_GIT_MULTI_NODE_v1.md` | **Private cloud Git** for consolidated Empire source across nodes; what to commit; pull workflow; relation to SMB mesh + `Sync_Empire_Shared_Cursor_Rules.ps1`. |
| `generated/cursor-reference/CURSOR_RULES_SYNC_v1.md` | **Workspace list:** `generated\DISCOVERED_CURSOR_PATHS.txt`; **hybrid Full vs Stub** rules sync (`Sync_Empire_Shared_Cursor_Rules.ps1 -Strategy Auto`). |
| `generated/cursor-reference/EMPIRE_NODE_LAYOUT_AND_PATHS_v1.md` | **Agreed placement:** Browser_Operator, Project_Openclaw, `Projects_schedular` vs `Projects_schedular_AI_X1`, `Project_LAN_Connections_Nodes_*`; VMS asks agents to create projects; HP-only audit TXT. |
| `.cursor/rules/core-empire-workspace-rename-relocate-discipline.mdc` | **Rename / relocate / migrate:** grep old paths, align `generated_*` child folder, update layout doc, **`Sync_Discovered_Cursor_Paths.py`**, git, tasks, notify nodes. |
| `.cursor/rules/core-empire-project-workspace-naming.mdc` | **New roots:** `Project_<ProjectName>_<NODE>`; agent scaffolds + discovery; **no postman busywork** when HP↔node is up; relay only when link or human gate blocks automation. |
| `generated/mygpt_share_crossref_report.md` | **Auto** matrix: `Scripts/verify_mygpt_share_crossref.py` — all MyGPT `.txt` vs 25 share corpora. |
| `generated/CLOUD_SSH_FALLBACK_EMPIRE_v1.md` | Cloud SSH fallback; includes **live EC2 inventory** when run from Cursor + AWS CLI. |
