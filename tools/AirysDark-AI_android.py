#!/usr/bin/env python3
# AirysDark-AI_android.py — Android build+fix orchestrator (self-contained)
#
# What it does:
#  1) Derives the most likely build command (AI → llama.cpp → heuristic)
#  2) Runs the build and captures build.log
#  3) If it fails, asks AI for a unified diff, applies it, and retries (loop)
#  4) Appends probe/build history to tools/android_probe.log (never deleted)
#  5) Writes/updates .github/workflows/AirysDark-AI_android.yml
#  6) Prints BUILD_CMD=... so workflows can capture it
#
# Dependencies:
#  - tools/AirysDark-AI_Request.py (centralized AI client)
#    (detector/prob bootstrap should have fetched it already)
#
# Env (optional):
#  OPENAI_API_KEY, OPENAI_MODEL=gpt-4o-mini (for OpenAI)
#  PROVIDER=openai|llama (default openai), FALLBACK_PROVIDER=llama|none
#  MODEL_PATH=...gguf, LLAMA_CPP_BIN=llama-cli, LLAMA_CTX=4096
#  AI_BUILDER_ATTEMPTS=3  (how many fix attempts)
#  AI_LOG_TAIL=160        (tail lines for build.log in prompts)
#
# Usage in workflow:
#   - name: Android runner
#     run: python3 tools/AirysDark-AI_android.py --mode run

from __future__ import annotations
import os, sys, json, re, shlex, textwrap, datetime, tempfile, subprocess, pathlib
from typing import Optional, List

ROOT  = pathlib.Path(".").resolve()
WF    = ROOT / ".github" / "workflows"
TOOLS = ROOT / "tools"
WF.mkdir(parents=True, exist_ok=True)
TOOLS.mkdir(parents=True, exist_ok=True)

# ---- Tunables ----
MAX_FIX_ATTEMPTS = int(os.getenv("AI_BUILDER_ATTEMPTS", "3"))
AI_LOG_TAIL      = int(os.getenv("AI_LOG_TAIL", "160"))

# ---- Files ----
BUILD_LOG       = ROOT / "build.log"
ANDROID_LOG     = TOOLS / "android_probe.log"           # append-only
ANDROID_JSON    = TOOLS / "android_probe.json"
AI_OUT_TXT      = TOOLS / "android_ai_response.txt"     # last AI answer (debug)
PATCH_SNAPSHOT  = ROOT  / ".pre_ai_fix.patch"

# ---- Import centralized requester (OpenAI → llama fallback) ----
def _load_requester():
    import importlib.util
    req = TOOLS / "AirysDark-AI_Request.py"
    if not req.exists():
        raise RuntimeError(f"Missing tools/AirysDark-AI_Request.py (fetch during bootstrap).")
    spec = importlib.util.spec_from_file_location("airysdark_ai_request", str(req))
    mod  = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod

# ---- small shell helper ----
def sh(cmd: str, cwd: Optional[pathlib.Path]=None, check: bool=False, capture: bool=True) -> str:
    p = subprocess.run(cmd, cwd=str(cwd or ROOT), shell=True, text=True,
                       stdout=subprocess.PIPE if capture else None,
                       stderr=subprocess.STDOUT if capture else None)
    if check and p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd, output=(p.stdout or ""))
    return p.stdout or ""

# ---- repo helpers ----
def ensure_git_repo():
    if not (ROOT / ".git").exists():
        sh("git init", check=False, capture=False)
        sh('git config user.name "airysdark-ai"', check=False)
        sh('git config user.email "airysdark-ai@local"', check=False)
        sh("git add -A", check=False)
        sh('git commit -m "bootstrap repo for android runner" || true', check=False)

def repo_tree(limit=200) -> str:
    out = sh("git ls-files || true")
    files = [ln for ln in out.splitlines() if ln.strip()]
    return "\n".join(files[:limit]) if files else "(no tracked files)";

def recent_diff(max_chars=3000) -> str:
    diff = sh("git diff --unified=2 -M -C HEAD~5..HEAD || true")
    return diff[-max_chars:] if diff else "(no recent git diff)"

def build_log_tail(lines=AI_LOG_TAIL) -> str:
    if not BUILD_LOG.exists():
        return "(no build log)"
    data = BUILD_LOG.read_text(errors="ignore").splitlines()
    start = max(0, len(data) - int(lines))
    return "\n".join(data[start:])

def append_android_log(lines: List[str]):
    stamp = datetime.datetime.utcnow().isoformat() + "Z"
    with ANDROID_LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n===== ANDROID RUN @ {stamp} =====\n")
        for ln in lines:
            f.write(ln.rstrip() + "\n")

# ---- build command discovery (AI → llama → heuristic) ----
def _find_gradlew() -> Optional[pathlib.Path]:
    direct = ROOT / "gradlew"
    if direct.exists(): return direct
    for p in ROOT.glob("**/gradlew"):
        return p
    return None

def _heuristic_build_cmd() -> str:
    gw = _find_gradlew()
    if gw is None:
        return "./gradlew assembleDebug --stacktrace"
    # prefer module 'app' if present
    app = list(ROOT.glob("**/app/build.gradle*"))
    if app:
        return f'cd {shlex.quote(str(app[0].parent))} && ./gradlew :app:assembleDebug --stacktrace'
    return f'cd {shlex.quote(str(gw.parent))} && ./gradlew assembleDebug --stacktrace'

def _android_prompt_for_cmd() -> str:
    tree = repo_tree()
    hints = {
        "has_gradlew": _find_gradlew() is not None,
        "has_settings_gradle": any(ROOT.glob("**/settings.gradle*")),
        "has_build_gradle": any(ROOT.glob("**/build.gradle*")),
        "modules_guess": [p.parent.name for p in ROOT.glob("**/build.gradle*")][:20],
    }
    tail = build_log_tail()
    return f"""You are an Android CI assistant. Output ONLY the single best shell command to build an installable artifact.
Rules:
- If needed, prefix with: cd <dir> && ./gradlew <task> --stacktrace
- Prefer Debug when multiple flavors exist.
- Valid tasks: assembleDebug, :app:assembleDebug, bundleDebug, assembleRelease, etc.
- No prose, just the command.

Repo tree (truncated):
{tree}

Hints:
{json.dumps(hints, indent=2)}

Recent build.log tail (optional):
{tail}
""".strip()

def derive_build_cmd(req_mod) -> str:
    logs = []
    # 1) OpenAI
    try:
        out = req_mod.request_ai(
            "Return ONLY the best Gradle command to build this Android project.",
            context_parts=[_android_prompt_for_cmd()],
            want_diff=False,
        )
        if isinstance(out, tuple): out = out[0]
        m = re.search(r'(^|\n)\s*(cd\s+[^\n]+?\s*&&\s*)?(\.\/gradlew|\bgradlew\b)\s+[^\n]+', out)
        if m:
            cmd = re.sub(r"\s+", " ", m.group(0).strip())
            logs.append("[cmd] OpenAI proposed: " + cmd)
            append_android_log(logs)
            return cmd if "--stacktrace" in cmd else (cmd + " --stacktrace")
    except Exception as e:
        logs.append(f"[cmd] OpenAI error: {e}")

    # 2) Fallback llama
    try:
        out = req_mod.request_ai(
            "Return ONLY the best Gradle command to build this Android project.",
            context_parts=[_android_prompt_for_cmd()],
            want_diff=False,
            provider="llama",
            fallback_provider="none",
        )
        if isinstance(out, tuple): out = out[0]
        m = re.search(r'(^|\n)\s*(cd\s+[^\n]+?\s*&&\s*)?(\.\/gradlew|\bgradlew\b)\s+[^\n]+', out)
        if m:
            cmd = re.sub(r"\s+", " ", m.group(0).strip())
            logs.append("[cmd] llama proposed: " + cmd)
            append_android_log(logs)
            return cmd if "--stacktrace" in cmd else (cmd + " --stacktrace")
    except Exception as e:
        logs.append(f"[cmd] llama error: {e}")

    # 3) Heuristic
    cmd = _heuristic_build_cmd()
    logs.append("[cmd] heuristic fallback: " + cmd)
    append_android_log(logs)
    return cmd

# ---- run build and capture build.log ----
def run_build(cmd: str) -> int:
    with open(BUILD_LOG, "wb") as f:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), shell=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.stdout
        for line in proc.stdout:
            sys.stdout.buffer.write(line)
            f.write(line)
        return proc.wait()

# ---- ask AI for a unified diff fix ----
def ask_ai_for_fix(req_mod, build_cmd: str) -> Optional[str]:
    task = ("You are an automated build fixer working in a Git repository.\n"
            "Return ONLY a valid unified diff (---/+++ with @@ hunks) that minimally fixes the build.\n"
            "Keep edits small and safe; adjust Gradle/Kotlin/Android config only if necessary.")
    context = [
        "## File list (truncated)\n" + repo_tree(),
        "## Recent git diff (truncated)\n" + recent_diff(),
        f"## Build command\n{build_cmd}",
        f"## Build log tail (last {AI_LOG_TAIL} lines)\n{build_log_tail(AI_LOG_TAIL)}",
    ]
    out_text, diff = req_mod.request_ai(
        task, context_parts=context, want_diff=True,
        system="You are a precise CI fixer. Output only a unified diff when asked for code changes."
    )
    # save raw answer for debugging
    try: AI_OUT_TXT.write_text(out_text or "", encoding="utf-8")
    except Exception: pass
    if diff: return diff
    # last chance: try extracting by regex if provider didn’t
    from AirysDark-AI_Request import extract_unified_diff as _extract  # safe import name
    return _extract(out_text or "")

def apply_patch(diff_text: str) -> bool:
    PATCH_SNAPSHOT.write_text(diff_text, encoding="utf-8")
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".patch") as tmp:
        tmp.write(diff_text)
        tmp_path = tmp.name
    try:
        sh("git add -A || true")
        sh(f"git apply --reject --whitespace=fix {tmp_path} || true")
        chg = sh("git status --porcelain")
        return bool(chg.strip())
    finally:
        try: os.unlink(tmp_path)
        except Exception: pass

# ---- generate/update final Android workflow (no builder step here) ----
def write_android_workflow(build_cmd: str):
    yml = f"""
name: AirysDark-AI - Android (generated)

on:
  workflow_dispatch: {{}}

permissions:
  contents: write
  pull-requests: write

concurrency:
  group: ${{{{ github.workflow }}}}-${{{{ github.ref }}}}
  cancel-in-progress: true

jobs:
  android-runner:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout (no credentials)
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          persist-credentials: false

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install requests

      # Android SDK
      - uses: actions/setup-java@v4
        with:
          distribution: temurin
          java-version: "17"
      - uses: android-actions/setup-android@v3
      - run: yes | sdkmanager --licenses
      - run: sdkmanager "platform-tools" "platforms;android-34" "build-tools;34.0.0"

      - name: Ensure AirysDark-AI tools (request + android only)
        run: |
          set -euo pipefail
          mkdir -p tools
          BASE_URL="https://raw.githubusercontent.com/AirysDark-AI/AirysDark-AI_builder/main/tools"
          [ -f tools/AirysDark-AI_Request.py ] || curl -fL "$BASE_URL/AirysDark-AI_Request.py" -o tools/AirysDark-AI_Request.py
          [ -f tools/AirysDark-AI_android.py ] || curl -fL "$BASE_URL/AirysDark-AI_android.py" -o tools/AirysDark-AI_android.py

      - name: Run Android AI loop (derive cmd, build, self-fix)
        id: run
        env:
          OPENAI_API_KEY: ${{{{ secrets.OPENAI_API_KEY }}}}
          OPENAI_MODEL: ${{{{ vars.OPENAI_MODEL || 'gpt-4o-mini' }}}}
          AI_BUILDER_ATTEMPTS: "3"
          AI_LOG_TAIL: "160"
        run: |
          set -euxo pipefail
          python3 tools/AirysDark-AI_android.py --mode run | tee /tmp/android.out
          if grep -q '^BUILD_CMD=' /tmp/android.out; then
            CMD=$(grep -E '^BUILD_CMD=' /tmp/android.out | sed 's/^BUILD_CMD=//')
            echo "BUILD_CMD=$CMD" >> "$GITHUB_OUTPUT"
          fi

      - name: Upload logs & artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: android-ai-loop
          retention-days: 14
          if-no-files-found: warn
          path: |
            build.log
            .pre_ai_fix.patch
            tools/android_ai_response.txt
            tools/android_probe.json
            tools/android_probe.log

      - name: Stage changes
        id: diff
        run: |
          git add -A
          if git diff --cached --quiet; then
            echo "changed=false" >> "$GITHUB_OUTPUT"
          else
            echo "changed=true" >> "$GITHUB_OUTPUT"
          fi

      - name: Pin remote with FG-PAT
        if: ${{{{ steps.diff.outputs.changed == 'true' }}}}
        env:
          BOT_TOKEN: ${{{{ secrets.BOT_TOKEN }}}}
          REPO_SLUG: ${{{{ github.repository }}}}
        run: |
          set -euxo pipefail
          git config --local --name-only --get-regexp '^http\\.https://github\\.com/\\.extraheader$' >/dev/null 2>&1 && \
            git config --local --unset-all http.https://github.com/.extraheader || true
          git config --global --add safe.directory "$GITHUB_WORKSPACE"
          git remote set-url origin "https://x-access-token:${{ '{{' }} BOT_TOKEN {{ '}}' }}@github.com/${{ '{{' }} REPO_SLUG {{ '}}' }}.git"
          git config --global url."https://x-access-token:${{ '{{' }} BOT_TOKEN {{ '}}' }}@github.com/".insteadOf "https://github.com/"
          git remote -v

      - name: Create PR with Android changes
        if: ${{{{ steps.diff.outputs.changed == 'true' }}}}
        uses: peter-evans/create-pull-request@v6
        with:
          token: ${{{{ secrets.BOT_TOKEN }}}}
          branch: ai/airysdark-ai-android-loop
          commit-message: "chore: Android AI loop changes"
          title: "AirysDark-AI: Android AI loop changes"
          body: |
            Android AI loop updated files (build fixes or config changes).
            - Build command: ${{{{ steps.run.outputs.BUILD_CMD }}}}
            - Logs: see artifact "android-ai-loop"
          labels: automation, ci
""".lstrip("\n")
    # double-brace escaping already handled
    (WF / "AirysDark-AI_android.yml").write_text(yml, encoding="utf-8")

# ---- main loop ----
def main_loop() -> int:
    ensure_git_repo()
    req = _load_requester()

    # 0) Choose/remember build command
    build_cmd = derive_build_cmd(req)
    ANDROID_JSON.write_text(json.dumps({"build_cmd": build_cmd}, indent=2), encoding="utf-8")
    append_android_log([f"[run] using build_cmd: {build_cmd}"])
    print(f"BUILD_CMD={build_cmd}")

    # 1) Write (or refresh) the workflow that runs this very file
    write_android_workflow(build_cmd)

    # 2) Try build → if fail → AI-fix loop
    for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
        append_android_log([f"[attempt] build try {attempt}/{MAX_FIX_ATTEMPTS}"])
        code = run_build(build_cmd)
        if code == 0:
            append_android_log(["[result] build OK"])
            return 0

        append_android_log([f"[result] build failed (code {code}); asking AI for diff..."])
        diff = ask_ai_for_fix(req, build_cmd)
        if not diff:
            append_android_log(["[ai] no usable diff returned"])
            continue

        changed = apply_patch(diff)
        if not changed:
            append_android_log(["[patch] applied but no changes detected; retrying"])
            continue

        append_android_log(["[patch] changes applied; will retry build"])

    append_android_log(["[final] still failing after attempts"])
    return 1

# ---- CLI ----
def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--mode" and sys.argv[2] == "run":
        return main_loop()
    print("Usage: AirysDark-AI_android.py --mode run")
    return 2

if __name__ == "__main__":
    sys.exit(main())