#!/usr/bin/env python3
# AirysDark-AI_builder.py — smart build fixer with self-teaching memory
#
# Features:
# - Tries previously learned patches for similar errors before calling the LLM
# - Diff-only contract, patch safety (reject/3way), rollback-on-worse
# - OpenAI → llama.cpp fallback
# - Log/artifacts + allow/deny globs + dangerous file guard

import os, sys, re, json, pathlib, subprocess, tempfile, shlex, datetime, hashlib
from typing import Optional, Tuple, List, Dict, Any

ROOT = pathlib.Path(os.getenv("PROJECT_ROOT", ".")).resolve()
TOOLS = ROOT / "tools"
KB_DIR = TOOLS / "ai_kb"
KB_FILE = KB_DIR / "knowledge.jsonl"

ATTEMPTS = int(os.getenv("AI_BUILDER_ATTEMPTS", "3"))
PROVIDER = os.getenv("PROVIDER", "openai")            # openai | llama
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
FALLBACK_PROVIDER = os.getenv("FALLBACK_PROVIDER", "llama")
LLAMA_CPP_BIN = os.getenv("LLAMA_CPP_BIN", "llama-cli")
LLAMA_MODEL_PATH = pathlib.Path(os.getenv("MODEL_PATH", "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"))
LLAMA_CTX = int(os.getenv("LLAMA_CTX", "4096"))

# Prompt/context sizing
MAX_PROMPT_TOKENS = int(os.getenv("MAX_PROMPT_TOKENS", "2500"))   # ~4 chars/token heuristic
REPO_TREE_LIMIT   = int(os.getenv("MAX_FILES_IN_TREE", "80"))
DIFF_CHAR_LIMIT   = int(os.getenv("RECENT_DIFF_MAX_CHARS", "2200"))
LOG_TAIL_LINES    = int(os.getenv("AI_LOG_TAIL", "160"))

# Build
BUILD_CMD = os.getenv("BUILD_CMD", "./gradlew assembleDebug --stacktrace")

# Edit constraints
ALLOWLIST_GLOBS = [g for g in os.getenv("ALLOWLIST_GLOBS", "").split(",") if g.strip()]
DENYLIST_GLOBS  = [g for g in os.getenv("DENYLIST_GLOBS", "").split(",") if g.strip()]

# Files / artifacts
BUILD_LOG = ROOT / "build.log"
PATCH_SNAPSHOT = ROOT / ".pre_ai_fix.patch"
AI_SUMMARY = ROOT / "ai_summary.txt"
AI_ATTEMPTS_LOG = ROOT / ".ai_attempt.jsonl"

DANGEROUS_PATH_HINTS = [
    ".github/workflows/",
    ".git/",
    "secrets.", "keystore", "gradle.properties", "local.properties",
]

PROMPT = """You are an automated build fixer working inside a Git repository.

Goal:
- Fix the current build failure with the smallest safe changes.

Resources:
- Truncated repo file list:
{repo_tree}

- Truncated recent VCS diff:
{recent_diff}

- Build command:
{build_cmd}

- Failing build log tail (last {log_tail} lines):
{build_tail}

Constraints:
- Output ONLY a unified diff (GNU patch format) with file hunks starting with '---' and '+++' lines and '@@' sections.
- Do NOT include explanations, code fences, or prose.
- Keep edits minimal and localized to fix the error.
- Prefer updating config (Gradle/CMake/etc.) or version bumps when appropriate.
- Do NOT modify unrelated files.
- If no change is needed, output an empty diff (just nothing).

If editing build systems, keep them consistent (e.g., Gradle plugin & Kotlin versions).
"""

# ---------------- helpers ----------------

def run(cmd, cwd=ROOT, capture=False, check=False, env=None):
    if capture:
        p = subprocess.run(cmd, cwd=cwd, shell=True, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        if check and p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode, cmd, p.stdout)
        return p
    else:
        return subprocess.run(cmd, cwd=cwd, shell=True, env=env, check=check)

def git(*args, capture=False):
    return run("git " + " ".join(shlex.quote(a) for a in args), capture=capture)

def ensure_git_repo():
    if not (ROOT / ".git").exists():
        run('git init')
        run('git config user.name "airysdark-ai"')
        run('git config user.email "airysdark-ai@local"')
        git("add", "-A")
        run('git commit -m "AI: initial snapshot" || true')

def repo_tree(limit=REPO_TREE_LIMIT):
    out = run("git ls-files || true", capture=True).stdout.splitlines()
    return "\n".join(out[:limit])

def recent_diff(limit_chars=DIFF_CHAR_LIMIT):
    out = run("git log --oneline -n 1 || true", capture=True).stdout.strip()
    if not out:
        return "(no recent commits)"
    diff = run("git diff --unified=2 -M -C HEAD~5..HEAD || true", capture=True).stdout
    return diff[-limit_chars:]

def log_tail(lines=LOG_TAIL_LINES):
    if not BUILD_LOG.exists():
        return "(no build log)"
    return "\n".join(BUILD_LOG.read_text(errors="ignore").splitlines()[-lines:])

def redact(text: str) -> str:
    text = re.sub(r'(?:AKIA|ASIA|SK|GH|GHO|ghp_)\\w+', '***REDACTED***', text)
    text = re.sub(r'(?i)(api[-_ ]?key|token|secret)\\s*[:=]\\s*\\S+', '\\1: ***REDACTED***', text)
    return text

def truncate_for_tokens(s: str, max_tokens=MAX_PROMPT_TOKENS):
    char_limit = max_tokens * 4
    if len(s) <= char_limit:
        return s
    head = s[: int(char_limit * 0.60)]
    tail = s[- int(char_limit * 0.35):]
    return head + "\n\n[...truncated...]\n\n" + tail

def build_once():
    with open(BUILD_LOG, "wb") as f:
        p = subprocess.Popen(BUILD_CMD, cwd=ROOT, shell=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout:
            sys.stdout.buffer.write(line)
            f.write(line)
    return p.wait()

def extract_unified_diff(text: str) -> Optional[str]:
    m = re.search(r'(?ms)^---\\s', text or "")
    return text[m.start():].strip() if m else None

def diff_touches_dangerous_paths(diff_text: str) -> bool:
    lower = diff_text.lower()
    for hint in DANGEROUS_PATH_HINTS:
        if hint in lower:
            if not ALLOWLIST_GLOBS:
                return True
    return False

def path_allowed_by_globs(filepath: str) -> bool:
    if ALLOWLIST_GLOBS:
        import fnmatch
        if not any(fnmatch.fnmatch(filepath, g.strip()) for g in ALLOWLIST_GLOBS):
            return False
    if DENYLIST_GLOBS:
        import fnmatch
        if any(fnmatch.fnmatch(filepath, g.strip()) for g in DENYLIST_GLOBS):
            return False
    return True

def filter_diff_by_globs(diff_text: str) -> str:
    # split on file boundaries and keep allowed files
    chunks = re.split(r'(?m)(?=^---\\s)', diff_text)
    kept = []
    for ch in chunks:
        if not ch.strip():
            continue
        m = re.search(r'^\\+\\+\\+\\s+(?:b/)?(.+)$', ch, flags=re.M)
        path = m.group(1).strip() if m else ""
        if path and path_allowed_by_globs(path):
            kept.append(ch)
    return "".join(kept) if kept else ""

def apply_patch(diff_text: str) -> Tuple[bool, str]:
    tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".patch")
    tmp.write(diff_text)
    tmp.close()
    try:
        git("add", "-A")
        run(f"git diff --staged > {shlex.quote(str(PATCH_SNAPSHOT))} || true")
        r = run(f"git apply --reject --whitespace=fix {shlex.quote(tmp.name)}", capture=True)
        if r.returncode != 0:
            r2 = run(f"git apply --3way --reject --whitespace=fix {shlex.quote(tmp.name)}", capture=True)
            if r2.returncode != 0:
                return False, r2.stdout
        git("add", "-A")
        run('git commit -m "AI: apply automatic fix" || true')
        return True, "applied"
    finally:
        os.unlink(tmp.name)

def compare_fail_signal(before_log: str, after_log: str) -> int:
    """Lower is better (fewer obvious failures)."""
    def score(t: str) -> int:
        t = t.lower()
        s = 0
        for kw in ["failed", "error", "exception", "could not", "not found", "undefined", "unresolved"]:
            s += t.count(kw)
        return s
    return score(after_log) - score(before_log)

# ---------------- self-teaching KB ----------------

def norm_line(s: str) -> str:
    s = s.strip()
    # remove timestamps, file paths line numbers, etc
    s = re.sub(r"/[^\\s:]+(\\.\\w+)+", "<PATH>", s)
    s = re.sub(r"\\b\\d{1,4}[:;,.]\\d{1,4}\\b", "<NUM>", s)
    s = re.sub(r"\\b\\d+\\b", "<NUM>", s)
    return s

def build_error_signature(text: str, max_lines: int = 30) -> Dict[str, Any]:
    lines = [norm_line(x) for x in (text or "").splitlines() if x.strip()]
    # take final N lines where errors usually summarize
    tail = lines[-max_lines:]
    sig_text = "\n".join(tail)
    h = hashlib.sha256(sig_text.encode("utf-8")).hexdigest()[:16]
    return {"hash": h, "preview": "\n".join(tail[:12])}

def kb_load() -> List[Dict[str, Any]]:
    KB_DIR.mkdir(parents=True, exist_ok=True)
    if not KB_FILE.exists():
        return []
    out = []
    with KB_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

def kb_save(entries: List[Dict[str, Any]]) -> None:
    KB_DIR.mkdir(parents=True, exist_ok=True)
    with KB_FILE.open("w", encoding="utf-8") as f:
        for e in entries[-500:]:  # keep last 500
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

def kb_find_candidate(sig: Dict[str, Any], entries: List[Dict, ]) -> Optional[Dict[str, Any]]:
    # simple match by hash; if not found, try fuzzy contains in preview
    for e in reversed(entries):
        if e.get("sig", {}).get("hash") == sig.get("hash"):
            return e
    # fuzzy: overlap some tokens from preview
    want = set((sig.get("preview") or "").lower().split())
    if not want:
        return None
    best = None
    best_score = 0
    for e in reversed(entries):
        got = set((e.get("sig", {}).get("preview") or "").lower().split())
        score = len(want & got)
        if score > best_score and score >= 6:
            best = e; best_score = score
    return best

def diff_is_small_and_safe(diff_text: str, max_bytes=120_000, max_files=12) -> bool:
    if len(diff_text.encode("utf-8")) > max_bytes:
        return False
    if diff_touches_dangerous_paths(diff_text):
        return False
    files = re.findall(r'^\\+\\+\\+\\s+(?:b/)?(.+)$', diff_text, re.M)
    if len(files) > max_files:
        return False
    return True

def kb_try_apply(sig: Dict[str, Any]) -> bool:
    entries = kb_load()
    cand = kb_find_candidate(sig, entries)
    if not cand:
        return False
    diff = cand.get("diff", "")
    if not diff or not diff_is_small_and_safe(diff):
        return False
    print("🧠 KB: trying previously learned patch (id:", cand.get("id", "?"), ")")
    ok, _ = apply_patch(diff)
    return ok

def kb_learn(sig: Dict[str, Any], diff: str, project_type: Optional[str] = None) -> None:
    if not diff or not diff_is_small_and_safe(diff):
        return
    entries = kb_load()
    eid = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "-" + sig["hash"]
    meta = {
        "when": datetime.datetime.utcnow().isoformat() + "Z",
        "project_type": project_type or "",
        "files": re.findall(r'^\\+\\+\\+\\s+(?:b/)?(.+)$', diff, re.M),
        "size_bytes": len(diff.encode("utf-8")),
    }
    entries.append({"id": eid, "sig": sig, "diff": diff, "meta": meta})
    kb_save(entries)
    # leave a short breadcrumb for humans
    note = TOOLS / "ai_kb" / f"learned_{eid}.txt"
    note.write_text(
        f"KB entry {eid}\n\nSignature preview:\n{sig['preview']}\n\nFiles:\n" +
        "\n".join(meta["files"]), encoding="utf-8"
    )

# ---------------- LLM providers ----------------

def _call_openai(prompt):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("openai_error: missing OPENAI_API_KEY")
    import requests
    url = "https://api.openai.com/v1/chat/completions"
    payload = {"model": OPENAI_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=180)
    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = {"raw": r.text}
        raise RuntimeError("openai_error:" + json.dumps(err))
    data = r.json()
    return data["choices"][0]["message"]["content"]

def _call_llama(prompt):
    if not LLAMA_MODEL_PATH.exists():
        raise RuntimeError(f"llama_error: model missing at {LLAMA_MODEL_PATH}")
    safe = truncate_for_tokens(prompt, MAX_PROMPT_TOKENS)
    cmd = [LLAMA_CPP_BIN, "-m", str(LLAMA_MODEL_PATH), "-p", safe, "-n", "2048", "--temp", "0.2", "-c", str(LLAMA_CTX)]
    out = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if out.returncode != 0:
        raise RuntimeError("llama_error:" + out.stdout[:500])
    return out.stdout

def call_llm(prompt):
    if PROVIDER == "openai":
        try:
            return _call_openai(prompt)
        except RuntimeError as e:
            if "openai_error" in str(e) and FALLBACK_PROVIDER == "llama":
                print("⚠️ OpenAI failed. Falling back to llama.cpp…")
                return _call_llama(prompt)
            raise
    elif PROVIDER == "llama":
        return _call_llama(prompt)
    else:
        raise RuntimeError(f"Unknown PROVIDER={PROVIDER}")

# ---------------- main ----------------

def main():
    TOOLS.mkdir(parents=True, exist_ok=True)
    KB_DIR.mkdir(parents=True, exist_ok=True)
    ensure_git_repo()

    print(f"== AirysDark-AI builder ==\nProject: {ROOT}\nProvider: {PROVIDER} (fallback: {FALLBACK_PROVIDER})")
    print(f"Build command: {BUILD_CMD}")

    # Baseline build
    base_code = build_once()
    if base_code == 0:
        print("✅ Build already succeeds. Nothing to do.")
        return 0

    before_tail = log_tail()
    base_sig = build_error_signature(before_tail)

    # 0) Try learned patch first
    if kb_try_apply(base_sig):
        print("🧠 KB patch applied. Rebuilding…")
        code = build_once()
        if code == 0:
            print("✅ Fixed by learned patch.")
            AI_SUMMARY.write_text("Fixed by KB patch\n", encoding="utf-8")
            return 0
        else:
            print("KB patch did not fully fix the build; continuing with AI.")
            # keep the change (might be partial), proceed

    # 1..N attempts with LLM
    for attempt in range(1, ATTEMPTS + 1):
        print(f"\n== Attempt {attempt}/{ATTEMPTS} ==")
        ctx = PROMPT.format(
            repo_tree=redact(truncate_for_tokens(repo_tree())),
            recent_diff=redact(truncate_for_tokens(recent_diff())),
            build_cmd=BUILD_CMD,
            log_tail=LOG_TAIL_LINES,
            build_tail=redact(truncate_for_tokens(log_tail())),
        )
        ctx = truncate_for_tokens(ctx)

        try:
            raw = call_llm(ctx)
        except Exception as e:
            print("LLM call failed:", e)
            return 1

        diff = extract_unified_diff(raw or "")
        if not diff:
            print("LLM did not return a unified diff. Stopping.")
            return 1

        if diff_touches_dangerous_paths(diff):
            print("⚠️ Proposed diff touches restricted paths; rejecting.")
            return 1

        if ALLOWLIST_GLOBS or DENYLIST_GLOBS:
            filtered = filter_diff_by_globs(diff)
            if not filtered.strip():
                print("⚠️ Diff removed by globs; nothing to apply.")
                return 1
            diff = filtered

        ok, why = apply_patch(diff)
        with open(AI_ATTEMPTS_LOG, "a", encoding="utf-8") as jf:
            jf.write(json.dumps({
                "ts": datetime.datetime.utcnow().isoformat() + "Z",
                "attempt": attempt,
                "apply_ok": ok,
                "apply_note": why,
                "diff_preview": diff[:1200],
            }) + "\n")

        if not ok:
            print("Patch apply failed.")
            return 1

        code = build_once()
        after_tail = log_tail()
        delta = compare_fail_signal(before_tail, after_tail)

        if code == 0:
            print("✅ Build fixed!")
            AI_SUMMARY.write_text(f"Build fixed on attempt {attempt}\n", encoding="utf-8")
            # learn the fix
            kb_learn(base_sig, diff, project_type=os.getenv("TARGET", ""))
            return 0
        else:
            if delta > 0:
                print("⚠️ Build seems worse; rolling back last commit.")
                git("reset", "--hard", "HEAD~1")
            else:
                pass
            before_tail = after_tail

    print("❌ Still failing after attempts.")
    return 1

if __name__ == "__main__":
    sys.exit(main())