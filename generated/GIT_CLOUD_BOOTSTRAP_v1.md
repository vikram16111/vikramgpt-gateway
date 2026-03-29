# Git cloud — bootstrap done on HP (v1)

## Done in repo (2026-03-29)

- **`git init`** under **`C:\Empire`**; **`origin`** = **`https://github.com/vikram16111/vikramgpt-gateway.git`**.
- **Local-only identity** (this clone): `git config user.name` / `user.email` under **`C:\Empire`** (not `--global`).
- **Commits on `main`:** small scaffold (`chore: initial commit`) + **`feat: on-demand HP mesh heal before node_bus submit and AI_X1 publish`**.
- **Push:** run on HP in an **interactive** terminal (Git Credential Manager / PAT). Headless agents may get **`User cancelled dialog`** / **`could not read Username`** — that is expected.
- **2026-03-29:** Merged unrelated **`origin/main`** (existing gateway workflow + Python) with local commits; resolved **`README.md`**; **`git push origin main`** succeeded (`7d6e5f5` on `main`).
- **ChatGPT pg41 letter:** not adopted as a new folder layout — **`generated/cursor-reference/CHATGPT_pg41_GITHUB_LETTER_VS_REPO_v1.md`** is a **pointer only**.

## GitHub vs “local” (no confusion)

- **GitHub** (`vikramgpt-gateway`, branch **`main`**) = **shared history** and backup other machines can clone.
- **Each PC** still has a **folder with a `.git`** (e.g. **`C:\Empire`** on HP) = your **working copy**. You edit files there; **`git push`** sends commits to GitHub; **`git pull`** brings GitHub’s commits down. Nothing replaces that — Git always needs a local tree to work on.

## Keep nodes aligned (hands-free pattern)

**HP** (after you commit/push script or rule changes):

```powershell
Set-Location C:\Empire
git pull origin main
git push origin main
```

**AI_X1** (separate clone recommended, does not replace the live `C:\Empire_AI_X1` tree):

```powershell
New-Item -ItemType Directory -Force -Path "C:\Empire_AI_X1\repos" | Out-Null
Set-Location "C:\Empire_AI_X1\repos"
if (-not (Test-Path ".\vikramgpt-gateway\.git")) { git clone https://github.com/vikram16111/vikramgpt-gateway.git }
Set-Location ".\vikramgpt-gateway"
git pull origin main
```

## When HP ↔ AI_X1 LAN is up — push **rules + scripts** to the worker

SMB mesh is separate from Git. After **`Z:`** works:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Empire\Scripts\Push_Empire_Mesh_Scripts_To_AI_X1.ps1"
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Empire\Scripts\Sync_Empire_Shared_Cursor_Rules.ps1"
python C:\Empire\Scripts\Sync_Discovered_Cursor_Paths.py
```

That updates **`Scripts\`** on the share and **`.cursor\rules`** stubs/full copies on discovered roots (including under **`Z:\`** when listed).

## What “git pull on HP” means (one line)

**`cd C:\Empire` then `git pull origin main`** = **download** the latest commits **from GitHub** into **this folder** on HP. Use it after you (or another machine) **pushed** changes, so HP matches GitHub. It does **not** map drives or fix SMB — it only updates Git-tracked files.

## What VMS supplies (one reply to agent)

1. **Remote URL** — private GitHub/GitLab repo, e.g. `https://github.com/<org>/empire-core.git` or SSH `git@github.com:<org>/empire-core.git`.
2. **Branch name** — usually `main`.
3. **Git identity** on this PC (once): `git config --global user.name "..."` and `user.email "..."` if not already set.

## Agent / VMS — first push (after remote exists)

Run from **Admin not required** (normal PowerShell), **`C:\Empire`**:

```powershell
Set-Location C:\Empire
git remote add origin <PASTE_REMOTE_URL>
git add -A
git status
git commit -m "chore: initial Empire import"
git branch -M main
git push -u origin main
```

**Before `git add -A`:** skim `git status` for anything that should stay untracked; `.gitignore` already excludes secrets and many artifacts.

## Pointers / submodules (later)

See **`generated/cursor-reference/EMPIRE_CLOUD_GIT_MULTI_NODE_v1.md`** for submodule and sparse-checkout patterns.

## GitHub App vs `git remote` (vikramgpt-autopilot)

- **Normal Git push/pull** uses a **repo URL** + **PAT** or **SSH key** — not the GitHub App private key. App keys are for **GitHub API** (installations, checks).
- **Empty remote first:** create **`https://github.com/vikram16111/vikramgpt-gateway`** (or chosen name), then `git remote add origin <url>` on **`C:\Empire`** when ready for first commit.
- **Secrets disk:** `Vikram GPT/Vikram GPT_Github/` — **`.pem` and `Vikram GPT Github.txt` are gitignored**; if those files ever leaked, **rotate PATs** in GitHub settings. Do not paste tokens into chat or commits.

## EC2 / gateway (context)

**Empire_Gateway_Primary** (Ubuntu) is an **SSH host** for gateway/OpenClaw-style workloads — **not** a substitute for `git remote` unless you intentionally install a Git server there. Prefer **GitHub/GitLab** as origin for Empire source.

## Plan (small phases) — no ChatGPT required for basics

| Phase | What | Cost (typical) |
|-------|------|----------------|
| **0** | Empty **private** GitHub repo (e.g. `vikramgpt-gateway` or `empire-core`) | **$0** (GitHub free private) |
| **1** | On HP: `git config` identity once; `git remote add origin https://github.com/.../....git`; first **small** commit (or `.gitignore` + README only), `git push` | **$0**; bandwidth negligible |
| **2** | Store **HTTPS PAT** (or SSH key path) in **`Empire_RAG_Secrets_v1`** as flat string keys, e.g. **`GITHUB_PAT`** / **`GITHUB_TOKEN`** / existing names in **`EMPIRE_SECRETS_MERGE_AND_ROLES_v1.md`** — **only** after `Validate_Empire_RAG_Secrets_JSON.py` passes on the merged JSON file | AWS Secrets Manager **per-secret** monthly charge (already paying for this secret) |
| **3** | Automation: CI (GitHub Actions) **optional** — free tier has minutes cap; paid only if you exceed. **Do not** enable Actions until you want spend. | Often **$0** at low volume |
| **4** | **GitHub App** (`vikramgpt-autopilot`) = API automation (checks, app auth), **not** `git push`; keep App **separate** from line 1–2 PAT flow unless you wire API clients. | **$0** unless Marketplace paid features |

**Verify keys in AWS:** `aws secretsmanager get-secret-value --secret-id Empire_RAG_Secrets_v1 --region us-east-2 --query SecretString --output text` (operator only; **never** paste output in chat). Look for **`GITHUB_*`** / **`BITCHE_PAT`** per role map. After you add a new key, **validate JSON** before upload.

**Starting point without rotating yet:** use the existing PAT from your local gitignored notes for **`git push`** only on a trusted machine; plan rotation once remote and first push work.
