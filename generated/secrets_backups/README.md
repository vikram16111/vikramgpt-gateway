# Secrets backups (Empire)

## Folders

| Folder | Retention |
|--------|-----------|
| `auto/` | **Rolling ~30 days** — safe to prune old files after verifying restores |
| `VMS_KEPT/` | **Human-owned** — never auto-delete; VMS deletes when done |

## Before any change to AWS Secrets Manager

1. Run `Scripts\Backup_Empire_RAG_Secrets_Local.ps1 -VmsKept` (or `-Auto` for routine).
2. Verify backup file exists and size > 0.
3. Run `python Scripts\Validate_Empire_RAG_Secrets_JSON.py` on the JSON file you will upload.
4. After any successful `PutSecretValue`, append one row to **`generated\EMPIRE_SECRET_PUT_AUDIT_LOG.md`** (tracked; no secrets in that file).
5. Only then edit/rotate secrets in AWS or app config (Console uses flat JSON object with string values).

## AWS CLI

Requires **AWS CLI v2** installed and credentials (`aws configure` or SSO) with `secretsmanager:GetSecretValue` on `Empire_RAG_Secrets_v1`.

## Today (2026-03-21)

**Status:** Backup from this Cursor session **not completed** — `aws` was **not** on PATH.  
**Action:** On a machine with AWS CLI, run:

```powershell
cd C:\Empire
.\Scripts\Backup_Empire_RAG_Secrets_Local.ps1 -VmsKept
```

Output path will be printed (under `generated\secrets_backups\VMS_KEPT\`).
