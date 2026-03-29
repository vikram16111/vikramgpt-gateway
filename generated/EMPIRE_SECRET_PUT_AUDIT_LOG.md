# Empire_RAG_Secrets_v1 — put audit (append-only, tracked in Git)

**Purpose:** Every AWS `PutSecretValue` (or Console save) must leave a **human trace** here after **`Backup_Empire_RAG_Secrets_Local.ps1`** and after **`Validate_Empire_RAG_Secrets_JSON.py`** passes on the payload file.

**Do not delete rows.** Newest at bottom. No secret values in this file — paths and notes only.

| UTC timestamp | Operator / agent | Backup file path | Validate OK | Notes (VersionId optional) |
|---------------|------------------|------------------|-------------|----------------------------|
| *(template)* | | `generated\secrets_backups\...` | yes | |
