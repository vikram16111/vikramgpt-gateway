# generated_Project_LAN_Connections_Nodes_HP

**Canonical child-project output root** for `Project_LAN_Connections_Nodes_HP` (see `core-generated-child-canonical.mdc`).

**Rename discipline:** If this project folder is renamed, this directory **must** be renamed to match: `generated_<ExactProjectFolderName>\`.

## Use this folder for

- Local **run logs** you choose to keep (optional copies of script output, dated notes).
- **Session checklists** (e.g. “verified Z: and Y: on date X”).
- **Exports** of `net use`, `Get-SmbMapping`, or Diagnose output (redact secrets).

## Do not use for

- **Secrets** (passwords, PATs) — never commit or store raw credentials here.
- **Replacing** Empire core health files under `C:\Empire\generated\health\` — those remain the operational source for scripts (`empire_lan_worker_last_ip.txt`, `EMPIRE_WORKER_SMB_CANDIDATES.json`, mesh cache).

## Subfolders (suggested)

- `logs/` — optional append-only text or markdown.
- `exports/` — sanitized SMB/diagnostics snippets.

Empire core **mirror register** (mesh/git audit) stays at  
`C:\Empire\generated\logs\EMPIRE_MIRROR_REGISTER.md` (see `Scripts\Empire_Mirror_Register.ps1`).
