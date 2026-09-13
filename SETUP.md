# SETUP — Windows 11 + VS Code + Google Colab

Step by step, from an empty machine to a submitted repository.

**Division of labour.** Tasks 1 and 3 run fine locally on Windows. **Task 2B (fine-tuning)
needs an NVIDIA GPU** — run it in Colab. `bitsandbytes` 4-bit quantisation has limited
Windows support and none without CUDA, so don't fight it locally.

---

## 1 · Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.10 – 3.12 | **Not 3.13** — some ML wheels lag. At install time, tick **"Add python.exe to PATH"**. |
| Git | latest | https://git-scm.com/download/win |
| VS Code | latest | https://code.visualstudio.com |
| Google account | — | For Colab and Drive |

VS Code extensions — install these three (Ctrl+Shift+X):

```
ms-python.python
ms-toolsai.jupyter
charliermarsh.ruff
```

Verify in PowerShell:

```powershell
python --version     # 3.10.x – 3.12.x
git --version
```

If `python` opens the Microsoft Store, Windows' app-execution alias is intercepting it:
Settings → Apps → Advanced app settings → App execution aliases → turn **off** both
`python.exe` and `python3.exe` entries.

---

## 2 · Create the GitHub repository

The brief requires a **public** repo named `CDAZZDEV-MLE-[YourName]`.

1. github.com → **New repository**
2. Name: `CDAZZDEV-MLE-Udith` (substitute your own name)
3. **Public** — a private repo at review time is treated as a missing submission
4. Do **not** initialise with a README; you already have one
5. Create

---

## 3 · Set up locally

```powershell
cd $HOME\Documents
git clone https://github.com/YOUR_USERNAME/CDAZZDEV-MLE-Udith.git
cd CDAZZDEV-MLE-Udith
```

If you were given this as a folder rather than a clone, copy its contents in, then:

```powershell
git init
git remote add origin https://github.com/YOUR_USERNAME/CDAZZDEV-MLE-Udith.git
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> **If PowerShell refuses with an execution-policy error**, run once:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
> then re-activate. Your prompt should now start with `(.venv)`.

Install. Locally you can skip the GPU-only packages:

```powershell
python -m pip install --upgrade pip
pip install pandas numpy requests pydantic openai python-dotenv yfinance matplotlib jinja2 scikit-learn langgraph langchain-core langchain-openai ddgs streamlit
```

Or install everything (`pip install -r requirements.txt`) and expect `bitsandbytes` to
either fail or install a CPU-only stub. That is fine — it is only needed in Colab.

---

## 4 · Open in VS Code and select the interpreter

```powershell
code .
```

**This is the step people miss.** Ctrl+Shift+P → `Python: Select Interpreter` → choose the
one under `.\.venv\Scripts\python.exe`. If you skip it, VS Code uses global Python, imports
fail, and the errors look like missing packages rather than a wrong interpreter.

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": ".venv\\Scripts\\python.exe",
  "python.analysis.extraPaths": [
    "./common",
    "./task1_financial/src",
    "./task2_genai/src",
    "./task3_agentic/src"
  ],
  "python.terminal.activateEnvironment": true,
  "files.eol": "\n",
  "[python]": { "editor.defaultFormatter": "charliermarsh.ruff" }
}
```

`extraPaths` stops Pylance underlining `from indicators import ...` in red. It affects the
editor only — the modules add their own `sys.path` entries at runtime.

`"files.eol": "\n"` matters: Windows CRLF line endings in a `.py` file that Colab then runs
can produce confusing diffs. Also run once:

```powershell
git config --global core.autocrlf true
```

---

## 5 · Add your API keys

```powershell
copy .env.example .env
code .env
```

Fill in at least `GROQ_API_KEY`. `.env` is gitignored — confirm with `git status` that it
does **not** appear.

| Key | Where | Cost |
|---|---|---|
| `GROQ_API_KEY` | https://console.groq.com/keys | Free, no card |
<!-- Fill in at least one. Three is strongly advised: see the quota note below. -->
| `OPENROUTER_API_KEY` | https://openrouter.ai/keys | Free tier |
| `GEMINI_API_KEY` | https://aistudio.google.com/apikey | Free, no card (per-model daily caps) |
| `HF_TOKEN` | https://huggingface.co/settings/tokens — **WRITE** scope | Free |

To load `.env` in a local terminal session:

```powershell
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('GROQ' , bool(os.getenv('GROQ_API_KEY')))"
```

---

## 6 · Run the tests (no keys, no network, no GPU)

```powershell
python task1_financial\tests\test_indicators.py
python task1_financial\tests\test_schemas.py
python task1_financial\tests\test_end_to_end_offline.py
python task2_genai\tests\test_diversity.py
python task2_genai\tests\test_evaluate.py
python task3_agentic\tests\test_task3_offline.py
```

All six should end with `ALL CHECKS PASSED` or `N/N tests passed`. If they do, your
environment is correct before you spend any API quota.

---

## 7 · Push the initial commit

```powershell
git add .
git commit -m "Initial submission: three tasks with verification suites"
git branch -M main
git push -u origin main
```

**Then verify it publicly.** Open the repo URL in a private/incognito window while logged
out. If you cannot see it, neither can the reviewer, and the brief treats that as a missing
submission.

---

## 8 · Run the notebooks in Colab

### Add your keys to Colab Secrets

Open any notebook in Colab → **key icon** in the left sidebar → **+ Add new secret**:

| Name | Value | Notebook access |
|---|---|---|
| `GROQ_API_KEY` | your key | **ON** |
| `OPENROUTER_API_KEY` | your key | **ON** |
| `HF_TOKEN` | your write token | **ON** (Task 2 only) |

The toggle is easy to miss and the failure looks like a missing key.

### Point the clone cell at your repo

In each notebook's setup cell, replace `YOUR_USERNAME`:

```python
!git clone -q https://github.com/YOUR_USERNAME/CDAZZDEV-MLE-Udith.git
```

In Task 2, also set `HF_REPO = "YOUR_HF_USERNAME/mistral-7b-credit-clause-extraction"`.

### Add the Colab badge

Put this at the top of each task's README (already present — just fix the username):

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/CDAZZDEV-MLE-Udith/blob/main/task1_financial/notebooks/task1_equity_research.ipynb)
```

### Run order

| Notebook | Runtime | Time | Notes |
|---|---|---|---|
| Task 1 | CPU | ~10 min | Run first — it warms up nothing but validates your keys cheaply |
| Task 3 | CPU | ~15 min | Agent runs; expect occasional retries on free-tier rate limits |
| Task 2 | **T4 GPU** | ~90 min | Runtime → Change runtime type → T4 GPU **before** running anything |

> **Task 2 memory tip.** Run 2A, 2B and 2C in one session. If you hit an OOM in 2C after
> training, run the cleanup cell (`del`, `gc.collect()`, `torch.cuda.empty_cache()`) rather
> than restarting — restarting loses the merged model unless you saved it to Drive first.

> **Colab disconnects.** Free Colab reclaims idle sessions. Before starting Task 2B, mount
> Drive and set `OUT` to a Drive path, so a disconnect during training does not lose the
> adapter:
> ```python
> from google.colab import drive; drive.mount('/content/drive')
> OUT = Path('/content/drive/MyDrive/cdazzdev_out'); OUT.mkdir(parents=True, exist_ok=True)
> ```

---

## 9 · Commit the executed notebooks — **outputs intact**

This is a disqualification criterion: *"Notebook outputs cleared before submission — reviewers must see executed cell results."*

In Colab, after a successful run: **File → Download → Download .ipynb**. Then locally:

```powershell
copy $HOME\Downloads\task1_equity_research.ipynb task1_financial\notebooks\
copy $HOME\Downloads\task2_finetuning_pipeline.ipynb task2_genai\notebooks\
copy $HOME\Downloads\task3_agentic_system.ipynb task3_agentic\notebooks\

git add .
git commit -m "Add executed notebooks with outputs, agent trace and evaluation results"
git push
```

**Verify outputs actually survived** before you email anyone:

```powershell
python -c "import json; nb=json.load(open('task1_financial/notebooks/task1_equity_research.ipynb')); print('cells with outputs:', sum(1 for c in nb['cells'] if c.get('outputs')))"
```

If that prints `0`, you committed a cleared notebook. Re-download from Colab.

> ⚠️ **Do not** enable `nbstripout`, and do not let any "clean notebooks on commit" hook run
> on this repo. It will silently strip the outputs the reviewer needs.

Also confirm the Task 3 artefact is present, since it is separately required:

```powershell
python -c "print(sum(1 for _ in open('task3_agentic/logs/agent_trace.jsonl')), 'trace records')"
```

---

## 10 · Final pre-submission checklist

Run through this against the brief's own list:

```powershell
# Repo is public — check in an incognito window while logged out
start https://github.com/YOUR_USERNAME/CDAZZDEV-MLE-Udith

# No credentials anywhere in the tree
git grep -nE "(gsk_|sk-or-|hf_[A-Za-z0-9]{20,})" -- . ; if ($LASTEXITCODE -eq 1) { "No key patterns found" }

# .env not tracked
git ls-files | Select-String "\.env$"     # should return nothing

# Required files present
foreach ($f in "CITATIONS.md","REFLECTION.md","README.md","task3_agentic/logs/agent_trace.jsonl") {
  "{0,-45} {1}" -f $f, (Test-Path $f)
}
```

- [ ] Repository public and reachable while logged out
- [ ] All notebook cell outputs visible (not cleared)
- [ ] No API keys, tokens or credentials anywhere in the repository
- [ ] Hugging Face model link opens without a login (or Drive link set to "Anyone with the link")
- [ ] `CITATIONS.md` complete
- [ ] `REFLECTION.md` present and under 600 words
- [ ] `agent_trace.jsonl` committed under `task3_agentic/logs/`
- [ ] Colab badge in each task README points at your username

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` for a package you installed | VS Code is on the global interpreter | Ctrl+Shift+P → Python: Select Interpreter → `.venv` |
| `ModuleNotFoundError: indicators` | Running from the wrong directory | Run from the repository root, not from inside `src/` |
| `.ps1 cannot be loaded` on activate | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `AllProvidersFailedError` | No key resolved | Check `.env` locally, or Secrets + notebook-access toggle in Colab |
| Groq `429` on **tokens per day** | 200,000/day exhausted - roughly three full Task 3 runs | Rolling window; small amounts free every few minutes. Add `GEMINI_API_KEY` so the run continues instead of waiting |
| OpenRouter `free-models-per-day` | 50 free requests/day exhausted | Resets at 00:00 UTC. Check `X-RateLimit-Reset` in the error for the exact epoch |
| Groq `429` | Free-tier rate limit | Expected. The client backs off and fails over — add `OPENROUTER_API_KEY` so it has somewhere to go |
| `FlashAttention only supports Ampere` | T4 is sm_75 | Already handled: the notebook uses `sdpa`. Don't change it |
| `CUDA out of memory` in Task 2B | Colab gave you a smaller GPU, or state is left over | Runtime → Restart, then run 2B from the top. Do not raise batch size |
| Colab disconnects mid-training | Idle reclaim on free tier | Save `OUT` to mounted Drive before starting (see §8) |
| `bitsandbytes` fails on Windows | No CUDA / limited Windows support | Expected. Task 2B is Colab-only |
| yfinance returns empty | Rate limit or bad symbol | Wait a minute and retry; verify the symbol on finance.yahoo.com |
| DuckDuckGo search fails repeatedly | Aggressive rate limiting | Expected and handled — the tool returns a suggestion and the agent falls back to `get_news` |
