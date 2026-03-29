Canonical copy: .cursor/AGENTS.md

**Author:** Cursor

# Empire — AI Agent Guidelines

**For:** Cursor, ChatGPT, Claude, MyGPT, and any AI working on this codebase.

## Before You Start

- **Cross-node reads (HP ↔ AI_X1):** **`.cursor/rules/core-empire-cross-node-filesystem-read.mdc`** — resolve `C:\Empire_AI_X1\...` via **`Z:\`** or **UNC** from **`Scripts\empire_node_routes.json`** before claiming a path is missing. **Mesh recovery Plans A–D (any node, router/new node/reinstall):** **`generated/cursor-reference/EMPIRE_MESH_LAN_PLANS_ABCD_v1.md`** + **`.cursor/rules/core-empire-mesh-lan-plans-abcd.mdc`**. **Canonical whoami + empirical SMB:** **`C:\Empire\whoami_HP_&_AI_X1.txt`**. **Local-only LAN notes (gitignored):** **`C:\Empire\generated\secrets_local\`**. **Discovered workspaces (HP canonical audit):** **`generated\DISCOVERED_CURSOR_PATHS.txt`** — `Scripts\Sync_Discovered_Cursor_Paths.py` + worker merge. **Layout / project creation:** **`generated/cursor-reference/EMPIRE_NODE_LAYOUT_AND_PATHS_v1.md`**. **Rules hybrid + Git pointers:** **`generated/cursor-reference/CURSOR_RULES_SYNC_v1.md`**, **`generated/cursor-reference/EMPIRE_CLOUD_GIT_MULTI_NODE_v1.md`**. **Run scripts with full path:** **`powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Empire\Scripts\....ps1"`** (see **`.cursor/rules/core-one-line-copypaste.mdc`**).
- **HP agents acting on AI_X1:** **`.cursor/rules/core-empire-hp-agent-ai-x1-execution.mdc`** — **Deploy** (`Deploy_Empire_Scripts_To_AI_X1_Share.ps1`), **UNC**, **`Invoke_Empire_Remote_Node_Command.ps1`** **before** saying work can only happen manually on AI_X1. Detail: **`generated/cursor-reference/EMPIRE_HP_AGENT_AI_X1_CAPABILITIES_v1.md`**.
- **This PC (LIGHT_NODE_HP / CONTROL_NODE):** **`.cursor/rules/core-hp-bootstrap-one-recovery.mdc`** — **RAM guard** (`Scripts\Empire_Resource_Guard.ps1`), **RECOVERY_LOG** (`generated\logs\RECOVERY_LOG.md`), **one-click** **`bootstrap\ONE_CLICK_EMPIRE_BOOTSTRAP.ps1`** (alias: `one_click_recovery.ps1`), **`node_config.json`**, **`generated\bootstrap\BOOTSTRAP_AND_ONE_CLICK_EMPIRE_v1.md`**, **heartbeat** (`Scripts\empire_heartbeat.py`, task **`Empire_Heartbeat_HP`**, **`generated\health\HEARTBEAT_ARCHITECTURE_v1.md`**). **Copy** that rule to other projects on this machine.
- **VMS autonomy (core):** **`.cursor/rules/core-vms-autonomy-handover.mdc`** — AI does what AI can; **handover = copy-paste block in chat first**; minimize folder juggling for VMS. Vision: **`generated/vision/TRUE_AUTONOMY_AND_20_SLOTS_ROADMAP_v1.md`**.
- **VMS UX (all agents):** **`.cursor/rules/core-vms-handsfree-eyesfree-ux.mdc`** — hands-free & eyes-free (short replies), **ask consent** before costly/risky/irreversible actions. Prefer **agents executing** over long paste lists. **Do not hand-copy rules:** run **`Scripts\Sync_Empire_Shared_Cursor_Rules.ps1`** after editing CORE rules — see **`generated/cursor-reference/CURSOR_RULES_SYNC_v1.md`**.
- **CG letters → Google Drive:** **`generated/CG_LETTER_GDRIVE_UPLOAD_FOR_AGENTS.md`** — use **`Tools\empire_upload_cg_to_gdrive.py`** + `secrets\google_drive\` (OAuth); do not ask VMS to manual-upload unless blocked.
- **CG + Drive structure (all projects):** **`generated/CG_GDRIVE_UNIVERSAL_LAYOUT_AND_CONTEXT_v1.md`** — entity folders, optional date-sorted names, `cursor-replies-to-chatgpt` sidecars (`*_CONTEXT.md`); recovery % / verification: **`generated/EMPIRE_RECOVERY_COMPLETENESS_CHECKLIST_v1.md`**.
- **OCR tiers (L1–L5) & full “Empire” scope:** **`generated/EMPIRE_OCR_TIER_LADDER_v1.md`** — local/Textract/heavy OCR/vision; 5 DTC backlog; not only CGAction v41.
- **OpenClaw profiles + chat coverage:** **`generated/OPENCLAW_PROVIDER_PROFILES_AND_CHAT_COVERAGE_MAP_v1.md`** — 22 provider profiles, HP vs mini PC, `Empire_RAG_Secrets_v1`, 25 `chatgpt_share_q_*` runs vs MyGPT paste folders.
- **MyGPT file ↔ share extract verification:** **`generated/MYGPT_SHARE_EXTRACT_CROSSREF_v1.md`** — example Gmail `p4` ↔ `69bec366…` ↔ `chatgpt_share_q_0006`; WhatsApp/Telegram profile copy to mini PC. **Re-run full matrix:** `python Scripts\verify_mygpt_share_crossref.py` → `generated/mygpt_share_crossref_report.md`.
- **AWS Secrets:** Before changing **`Empire_RAG_Secrets_v1`**, run **`Scripts\Backup_Empire_RAG_Secrets_Local.ps1`** — see **`.cursor/rules/core-aws-secrets-backup-before-change.mdc`**, **`.cursor/rules/core-empire-secrets-strict.mdc`** (format + no ad-hoc writes), **`generated/AWS_CLI_AND_SECRETS_BACKUP_SETUP_FOR_VMS_v1.md`**, **`generated/AWS_DUAL_PROFILES_AND_COST_POLICY_v1.md`**, **`generated/AWS_CREDENTIALS_PERSISTENCE_AND_RESTORE_v1.md`**, **`generated/AWS_PROFILE_EMPIRE_AND_ENV_v1.md`**, **`generated/secrets_backups/README.md`**. Optional copy blocks: **`generated/COPY_BLOCKS_FOR_VMS.md`**.
- **Agent handover (universal — every synced workspace root):** **`generated/AGENT_HANDOVER_CONTEXT_POLICY_v1.md`**. **Fresh context / new chat:** read **`generated/HANDOVER_NOTES/LATEST_HANDOVER_POINTER.md`** first (no VMS paste required); doctrine **`generated/cursor-reference/CURSOR_CONTEXT_HANDOVER_UNIVERSAL_v1.md`**; scaffold **`Scripts\Ensure_Handover_Scaffold_All_Workspaces.ps1`**. Rules: **`.cursor/rules/core-cursor-context-handover.mdc`**, **`.cursor/rules/core-vms-autonomy-handover.mdc`**. **Session installs:** **`generated/logs/SESSION_RECOVERY_LOG.md`**.
- **LLM path or cost:** Read **`generated/EMPIRE_TEMPORARY_STOPS_AND_FIXES.md`** first — SmartRunner and the EventBridge rule to it are temporarily off; fixes must be done before re-enable.
- Then read and follow:

1. **`generated/OPERATING_CONTRACT.md`** — Universal behaviors:
   - Syntax verification before shipping edits
   - Batch checks when given a list of files
   - Utils 3-tier, secrets, response codes
   - Save recurring topics so all AI can follow

2. **`generated/LAMBDA_IMPROVEMENTS_AND_AUDIT.md`** — Utils alignment, response codes, improvement list.

3. **`generated/MASTER_OF_ONE_AND_UNIVERSAL_TEMPLATE_REMARKS.md`** — Master of One, section banners, Universal Template rules.

5. **`.cursor/rules/empire-lambda-domain-strategy.mdc`** — Lambda domain strategy: do **not** keep adding new Lambdas; prefer **extending existing domain runtimes** (add models, capabilities, modules). Before asking VMS for a new Lambda, check for a corresponding domain runtime. Keeps system light, fast, cost-efficient. See also `Consolidate_Empire/generated/CONSOLIDATION_ADVICE.md`.

4. **`generated/UTILS_AND_LAMBDA_ALIGNMENT.md`** — Utils load sequence, key functions.

## VMS contact emails (all projects)

- **Two emails:** info@heyaru.com (stable), vikram16111@gmail.com (nomadic / AI services — OpenAI, Cursor, AWS, Grok, N8N, etc.). When a script or form allows two, add both. See **`.cursor/rules/core-vms-contact-emails.mdc`**.

## File organization (all projects)

- **Smart organizer:** All agent output only under **`generated/`** with subfolders. Do **not** create one file per tiny thought — use rolling files, merge ideas, archive stale logs. When organizing or cleaning, read **`generated/cursor-reference/FILE_ORGANIZATION_AND_CONSOLIDATION_v1.md`** and **`.cursor/rules/core-smart-file-organizer.mdc`**.
- **Email AI:** **RAW .eml on HP (`D:\Outlook\...`); parsed JSONL/attachments on AI_X1 under `HP_D_MIRROR` (use `Z:\HP_D_MIRROR\...` from HP when LAN mapped).** Registry: **`generated/automation/EMAIL_AI_ARCHIVES_REGISTRY_v1.json`**. Rule: **`empire-generated-emailai-vms-junk-nodebus.mdc`**. Do not ask VMS for internal JSONL paths.
- **Always prepare setup guide:** Keep `generated/NEW_PROJECT_SETUP_GUIDE.md` as a living, recovery-safe guide for new project setup/install/sync steps. Update it whenever a new setup lesson is learned.

## Joint TEST log & DISCOVERED_CURSOR_PATHS (CORE rule)

- **Joint TEST log:** Every test logs Designer / Executor / Worker. Log: `generated/logs/JOINT_TEST_LOG.md`; policy: `generated/cursor-reference/JOINT_TEST_POLICY_v1.md`. When screenshot requested, use a worker that can capture (e.g. OC). Rule: `.cursor/rules/core-ensure-wiring-and-test.mdc`.
- **DISCOVERED_CURSOR_PATHS.txt:** Kept up to date **only** by `Scripts\Sync_Discovered_Cursor_Paths.py`. Do not edit the file by hand. Run the script before any task that uses the list. See `generated/JOINT_TEST_LOG_AND_PATHS_AWARENESS.md` and `.cursor/rules/core-discovered-paths-sync.mdc`.
- **Test dedupe contract (mandatory):** Before running a test, check latest matching entry in `generated/logs/JOINT_TEST_LOG.md`. If latest is PASS and no relevant code/config changed, do not rerun. If latest is FAIL, run fix→retest loop until PASS or blocker is logged.
- **Resource safety:** On this machine profile (16GB RAM), run tests in small batches, prefer targeted tests over full suites, and avoid parallel heavy workloads that risk OOM.

## Key Paths

- Routing: `Lambda_Empire_ConfigManager\ConfigManager_Routing_s3_vikramgpt-memory-vault_config_routing_json\`
- Utils: `Lambda_Empire_Utils\empire_api_utils_v4.5-U.1.4.76.py`
- Lambdas: See `LAMBDA_ALIGNMENT_LOG.md` for file paths.

## Batch Syntax Check

```bash
python Scripts\batch_syntax_check.py
```

Run before deployments. All 50 Lambdas must pass. Logs to `generated/AGENT_ACTIONS_LOG.md`.

## Auditability

- All agent-run tasks must be traceable. See `OPERATING_CONTRACT.md` §3.
- Log location: `generated/AGENT_ACTIONS_LOG.md`.
