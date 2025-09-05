#!/usr/bin/env python3
"""
AirysDark-AI_prob.py — Step 2 (Probe)

What it does
------------
- Reads detector output (tools/airysdark_ai_scan.json)
- Recursively scans the repo (all folders) and collects a snapshot + textual hints
- Proposes a build command for env TARGET
- If TARGET=android:
    • run an extra Android probe (gradle wrapper(s), modules from settings.gradle(.kts),
      best-effort gradle task sampling) and emit an Android-only workflow that calls
      tools/AirysDark-AI_android.py (self-contained loop; no extra fetching)
- Otherwise:
    • Generates a manual-run workflow `.github/workflows/AirysDark-AI_build.yml`
      - If OPENAI_API_KEY is set, asks OpenAI to draft the workflow, else uses a robust template
- Writes:
    tools/airysdark_ai_prob_report.json
    tools/airysdark_ai_prob_report.log          (append-only; keeps history)
    tools/airysdark_ai_build_ai_response.txt    (only if AI used)
    .github/workflows/AirysDark-AI_build.yml    (non-Android targets)
    .github/workflows/AirysDark-AI_android.yml  (Android target)
    pr_body_build.md                            (for create-pull-request step)
"""

import os
import re
import sys
import json
import shlex
import pathlib
import textwrap
import subprocess

ROOT = pathlib.Path(os.getenv("PROJECT_DIR", ".")).resolve()
TOOLS = ROOT / "tools"
WF_DIR = ROOT / ".github" / "workflows"
TOOLS.mkdir(parents=True, exist_ok=True)
WF_DIR.mkdir(parents=True, exist_ok=True)

SCAN_JSON = TOOLS / "airysdark_ai_scan.json"
PROB_JSON = TOOLS / "airysdark_ai_prob_report.json"
PROB_LOG  = TOOLS / "airysdark_ai_prob_report.log"
AI_OUT    = TOOLS / "airysdark_ai_build_ai_response.txt"
PR_BODY   = ROOT / "pr_body_build.md"
BUILD_WF  = WF_DIR / "AirysDark-AI_build.yml"          # generic (non-Android)
ANDROID_WF = WF_DIR / "AirysDark-AI_android.yml"       # Android-only workflow


# ---------------- Utilities ----------------

def read_scan_json():
    if not SCAN_JSON.exists():
        return {}
    try:
        return json.loads(SCAN_JSON.read_text(errors="ignore"))
    except Exception as e:
        return {"_error": f"failed to parse {SCAN_JSON}: {e}"}

def repo_snapshot(max_files=6000, max_text_lines=80):
    """
    Walk entire repo (except .git), list files, and capture heads of many text/code files.
    """
    files = []
    doc_hints = {}
    for r, ds, fs in os.walk(ROOT):
        if ".git" in ds:
            ds.remove(".git")
        for fn in fs:
            p = pathlib.Path(r) / fn
            try:
                rel = str(p.relative_to(ROOT))
            except Exception:
                rel = str(p)
            files.append(rel)

            lower = fn.lower()
            if lower.endswith((
                ".md",".txt",".rst",".ini",".cfg",".toml",".gradle",".kts",".xml",".yml",".yaml",".json",
                ".properties",".mk",".cmake",".ninja",".conf",".bat",".ps1",".sh",".groovy",".kt",".java",
                ".cpp",".c",".h",".hpp",".swift",".go",".cs",".py",".rb",".ts",".js",".mjs",".cjs",".sql",
            )):
                try:
                    lines = (p.read_text(errors="ignore").splitlines())[:max_text_lines]
                    doc_hints[rel] = lines
                except Exception:
                    pass

            if len(files) >= max_files:
                break
        if len(files) >= max_files:
            break
    return {"files": files, "doc_hints": doc_hints}

def find_first(globs):
    for g in globs:
        found = list(ROOT.glob(g))
        if found:
            return found[0]
    return None

def _sh(cmd: str, cwd: pathlib.Path | None = None, timeout: int = 20):
    try:
        p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, shell=True, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        return p.stdout or "", p.returncode
    except Exception as e:
        return f"(exec failed: {e})", 127


# ---------------- Build command heuristics (original behavior) ----------------

def guess_android_cmd():
    wrappers = [ROOT / "gradlew", *ROOT.glob("**/gradlew")]
    wrappers = [w for w in wrappers if w.exists()]
    if not wrappers:
        return "./gradlew assembleDebug --stacktrace"
    # prefer the shortest path (closest to root) to avoid nested sample projects
    g = sorted(wrappers, key=lambda p: len(str(p)))[0]
    gradle_dir = g.parent

    # Try to parse settings for modules
    mods = []
    for sname in ("settings.gradle", "settings.gradle.kts"):
        sp = gradle_dir / sname
        if sp.exists():
            txt = sp.read_text(errors="ignore")
            incs = re.findall(r'include\s*\((.*?)\)', txt, flags=re.S | re.I)
            for raw in incs:
                parts = [p.strip(" '\"\t") for p in re.split(r'[,\s]+', raw.strip()) if p.strip()]
                for p in parts:
                    if p.startswith(":"):
                        mods.append(p[1:])
    # Prefer common app modules
    for m in mods:
        if m in ("app", "mobile", "android"):
            return f'cd "{gradle_dir}" && ./gradlew :{m}:assembleDebug --stacktrace'
    # Fallback to assembleDebug at wrapper dir
    return f'cd "{gradle_dir}" && ./gradlew assembleDebug --stacktrace'

def guess_cmake_cmd():
    root_cmake = ROOT / "CMakeLists.txt"
    if root_cmake.exists():
        return "cmake -S . -B build && cmake --build build -j"
    first = find_first(["**/CMakeLists.txt"])
    if first:
        out = f'build/{str(first.parent).replace("/", "_")}'
        return f'cmake -S "{first.parent}" -B "{out}" && cmake --build "{out}" -j'
    return "echo 'No CMakeLists.txt found' && exit 1"

def guess_linux_cmd():
    mk = find_first(["Makefile", "**/Makefile", "**/GNUmakefile"])
    if mk:
        return f'make -C "{mk.parent}" -j'
    mb = find_first(["meson.build", "**/meson.build"])
    if mb:
        d = mb.parent
        return f'(cd "{d}" && (meson setup build --wipe || true); meson setup build || true; ninja -C build)'
    nb = find_first(["build.ninja", "**/build.ninja"])
    if nb:
        return f'(cd "{nb.parent}" && ninja)'
    return "echo 'No Linux build files found' && exit 1"

def guess_node_cmd():
    pkg = find_first(["package.json", "**/package.json"])
    if pkg:
        return f'cd "{pkg.parent}" && npm ci && npm run build --if-present'
    return "echo 'No package.json found' && exit 1"

def guess_python_cmd():
    pj = find_first(["pyproject.toml", "**/pyproject.toml", "setup.py", "**/setup.py"])
    if pj:
        return f'cd "{pj.parent}" && pip install -e . && (pytest || python -m pytest || true)'
    return "echo 'No python project found' && exit 1"

def guess_rust_cmd():    return "cargo build --locked --all-targets --verbose"
def guess_dotnet_cmd():  return "dotnet restore && dotnet build -c Release"
def guess_maven_cmd():   return "mvn -B package --file pom.xml"
def guess_flutter_cmd(): return "flutter build apk --debug"
def guess_go_cmd():      return "go build ./..."
def guess_bazel_cmd():   return "bazel build //..."
def guess_scons_cmd():   return "scons -Q"
def guess_ninja_cmd():
    nb = find_first(["build.ninja","**/build.ninja"])
    if nb:
        return f'(cd "{nb.parent}" && ninja)'
    return "ninja"

def propose_build_cmd(target: str) -> str:
    t = (target or "").lower()
    if   t == "android": return guess_android_cmd()
    elif t == "cmake":   return guess_cmake_cmd()
    elif t == "linux":   return guess_linux_cmd()
    elif t == "node":    return guess_node_cmd()
    elif t == "python":  return guess_python_cmd()
    elif t == "rust":    return guess_rust_cmd()
    elif t == "dotnet":  return guess_dotnet_cmd()
    elif t == "maven":   return guess_maven_cmd()
    elif t == "flutter": return guess_flutter_cmd()
    elif t == "go":      return guess_go_cmd()
    elif t == "bazel":   return guess_bazel_cmd()
    elif t == "scons":   return guess_scons_cmd()
    elif t == "ninja":   return guess_ninja_cmd()
    else:                return "echo 'Unknown TARGET; update env.TARGET' && exit 1"


# ---------------- EXTRA: Android deep probe ----------------

def _find_gradlews_all():
    gws = []
    if (ROOT / "gradlew").exists():
        gws.append(ROOT / "gradlew")
    for p in ROOT.glob("**/gradlew"):
        if p not in gws:
            gws.append(p)
    return gws

def _parse_settings_modules(txt: str):
    mods = set()
    for m in re.findall(r'include\s*\((.*?)\)', txt, flags=re.S|re.I):
        for part in re.split(r'[,\s]+', m.strip()):
            part = part.strip().strip("'\"")
            if part.startswith(":"): part = part[1:]
            if part: mods.add(part)
    for m in re.findall(r'include\s*\(?\s*[\'"](:?[^\'"]+)[\'"]\s*\)?', txt, flags=re.I):
        v = m.strip().strip("'\"")
        if v.startswith(":"): v = v[1:]
        if v: mods.add(v)
    return sorted(mods)

def android_deep_probe():
    info = {
        "wrappers": [],
        "modules": [],
        "tasks_sample": "",
        "task_guess": [],
        "cmd_recommendation": "",
    }
    gws = _find_gradlews_all()
    info["wrappers"] = [str(x) for x in gws]

    if gws:
        gw = gws[0]
        gdir = gw.parent
        for s in ("settings.gradle", "settings.gradle.kts"):
            sp = gdir / s
            if sp.exists():
                mods = _parse_settings_modules(sp.read_text(errors="ignore"))
                info["modules"] = mods
                break
        out, _ = _sh("./gradlew -q tasks --all", cwd=gdir, timeout=20)
        if out:
            info["tasks_sample"] = out[:20000]

    guesses = []
    prefers = info["modules"] or ["app", "mobile", "android"]
    for m in prefers:
        for t in ("assembleDebug","bundleDebug","assembleRelease","bundleRelease"):
            guesses.append(f":{m}:{t}")
    for t in ("assembleDebug","bundleDebug","assembleRelease","bundleRelease","build"):
        guesses.append(t)

    if info["tasks_sample"]:
        present = [g for g in guesses if re.search(rf"(^|\s){re.escape(g)}(\s|$)", info["tasks_sample"])]
        if present:
            guesses = present
    info["task_guess"] = guesses[:8]

    if gws:
        gdir = gws[0].parent
        chosen = guesses[0] if guesses else "assembleDebug"
        info["cmd_recommendation"] = f'cd {shlex.quote(str(gdir))} && ./gradlew {chosen} --stacktrace'
    else:
        info["cmd_recommendation"] = "./gradlew assembleDebug --stacktrace"

    return info


# ---------------- Setup steps for generic build workflow ----------------

def setup_steps_yaml(ptype: str) -> str:
    if ptype == "android":
        return textwrap.dedent("""\
          - uses: actions/setup-java@v4
            with:
              distribution: temurin
              java-version: "17"
          - uses: android-actions/setup-android@v3
          - run: yes | sdkmanager --licenses
          - run: sdkmanager "platform-tools" "platforms;android-34" "build-tools;34.0.0"
        """)
    if ptype == "node":
        return textwrap.dedent("""\
          - uses: actions/setup-node@v4
            with:
              node-version: "20"
        """)
    if ptype == "rust":
        return textwrap.dedent("""\
          - uses: dtolnay/rust-toolchain@stable
          - run: rustc --version && cargo --version
        """)
    if ptype == "dotnet":
        return textwrap.dedent("""\
          - uses: actions/setup-dotnet@v4
            with:
              dotnet-version: "8.0.x"
          - run: dotnet --info
        """)
    if ptype == "maven":
        return textwrap.dedent("""\
          - uses: actions/setup-java@v4
            with:
              distribution: temurin
              java-version: "17"
          - run: mvn --version
        """)
    if ptype == "flutter":
        return textwrap.dedent("""\
          - uses: subosito/flutter-action@v2
            with:
              flutter-version: "3.22.0"
          - run: flutter --version
        """)
    if ptype == "go":
        return textwrap.dedent("""\
          - uses: actions/setup-go@v5
            with:
              go-version: "1.22"
          - run: go version
        """)
    if ptype == "linux":
        return textwrap.dedent("""\
          - name: Install Meson & Ninja (Linux builds)
            run: |
              sudo apt-get update
              sudo apt-get install -y meson ninja-build pkg-config
        """)
    # cmake/python/bazel/scons/ninja/unknown rely on setup-python only
    return ""


# ---------------- Generic build workflow generation (patched) ----------------

def render_build_workflow(target: str, build_cmd: str) -> str:
    setup = setup_steps_yaml(target)
    # Use placeholders to avoid interfering with ${{ }} in YAML
    tmpl = r"""
name: AirysDark-AI - Build (__PTYPE__)

on:
  workflow_dispatch: {}

permissions:
  contents: write
  pull-requests: write

env:
  # Optional central KB repo; leave blank to skip
  KB_COLLECTION_REPO: "AirysDark-AI/ai-kb-collection"

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout (no credentials)
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          persist-credentials: false

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

__SETUP__

      # ===== AI KB: restore cache =====
      - name: Restore AI KB cache
        uses: actions/cache@v4
        with:
          path: tools/ai_kb
          key: ai-kb-__PTYPE__-${{ github.repository }}-v1

      - name: Verify AirysDark-AI tools exist
        shell: bash
        run: |
          set -euxo pipefail
          test -f tools/AirysDark-AI_builder.py
          ls -la tools

      - name: Build (capture)
        id: build
        shell: bash
        run: |
          set -euxo pipefail
          CMD="__BUILD_CMD__"
          echo "BUILD_CMD=$CMD" >> "$GITHUB_OUTPUT"
          set +e; bash -lc "$CMD" | tee build.log; EXIT=$?; set -e
          echo "EXIT_CODE=$EXIT" >> "$GITHUB_OUTPUT"
          [ -s build.log ] || echo "(no build output captured)" > build.log
          exit 0
        continue-on-error: true

      - name: Upload build log
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: __PTYPE__-build-log
          if-no-files-found: warn
          retention-days: 7
          path: build.log

      # --- AI auto-fix (OpenAI → llama.cpp) ---
      - name: Build llama.cpp (CMake, no CURL) in temp
        if: always() && __GHA__ steps.build.outputs.EXIT_CODE __GHA_END__ != '0'
        run: |
          set -euxo pipefail
          TMP="__GHA__ runner.temp __GHA_END__"
          cd "$TMP"
          rm -rf llama.cpp
          git clone --depth=1 https://github.com/ggml-org/llama.cpp
          cd llama.cpp
          cmake -S . -B build -D CMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
          cmake --build build -j
          echo "LLAMA_CPP_BIN=$PWD/build/bin/llama-cli" >> $GITHUB_ENV

      - name: Fetch GGUF model (TinyLlama)
        if: always() && __GHA__ steps.build.outputs.EXIT_CODE __GHA_END__ != '0'
        run: |
          mkdir -p models
          curl -L -o models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
            https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

      - name: Attempt AI auto-fix (OpenAI → llama fallback)
        if: always() && __GHA__ steps.build.outputs.EXIT_CODE __GHA_END__ != '0'
        env:
          PROVIDER: openai
          FALLBACK_PROVIDER: llama
          OPENAI_API_KEY: __GHA__ secrets.OPENAI_API_KEY __GHA_END__
          OPENAI_MODEL: __GHA__ vars.OPENAI_MODEL || 'gpt-4o-mini' __GHA_END__
          MODEL_PATH: models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
          AI_BUILDER_ATTEMPTS: "3"
          BUILD_CMD: __GHA__ steps.build.outputs.BUILD_CMD __GHA_END__
        run: python3 tools/AirysDark-AI_builder.py || true

      - name: Upload AI patch (if any)
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: __PTYPE__-ai-patch
          path: .pre_ai_fix.patch
          if-no-files-found: ignore
          retention-days: 7

      # ===== AI KB: upload artifact & save cache =====
      - name: Upload AI KB (artifact)
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: ai-kb
          path: tools/ai_kb/**
          if-no-files-found: warn
          retention-days: 30

      - name: Save AI KB cache
        if: always()
        uses: actions/cache/save@v4
        with:
          path: tools/ai_kb
          key: ai-kb-__PTYPE__-${{ github.repository }}-v1

      # ===== AI KB: push snapshot to collection repo (optional) via KB_PUSH_TOKEN =====
      - name: Push KB snapshot to central collection repo
        if: always() && env.KB_COLLECTION_REPO != '' && secrets.KB_PUSH_TOKEN != ''
        shell: bash
        run: |
          set -euxo pipefail
          if [ ! -s tools/ai_kb/knowledge.jsonl ]; then
            echo "No knowledge.jsonl to push; skipping."
            exit 0
          fi
          OWNER_REPO="${GITHUB_REPOSITORY}"
          OWNER="${OWNER_REPO%%/*}"
          REPO="${OWNER_REPO#*/}"
          TS="$(date -u +'%Y-%m-%dT%H-%M-%SZ')"
          WORKDIR="$(mktemp -d)"
          git config --global user.name "airysdark-ai-bot"
          git config --global user.email "airysdark-ai-bot@users.noreply.github.com"
          git clone "https://x-access-token:${{ secrets.KB_PUSH_TOKEN }}@github.com/${{ env.KB_COLLECTION_REPO }}.git" "$WORKDIR/kb"
          cd "$WORKDIR/kb"
          mkdir -p "${OWNER}/${REPO}/snapshots"
          cp -f "$GITHUB_WORKSPACE/tools/ai_kb/knowledge.jsonl" "${OWNER}/${REPO}/knowledge.jsonl"
          cp -f "$GITHUB_WORKSPACE/tools/ai_kb/knowledge.jsonl" "${OWNER}/${REPO}/snapshots/${TS}.jsonl"
          {
            echo "repo: ${OWNER_REPO}"
            echo "run_id: ${GITHUB_RUN_ID}"
            echo "run_url: https://github.com/${OWNER_REPO}/actions/runs/${GITHUB_RUN_ID}"
            echo "ref: ${GITHUB_REF}"
            echo "timestamp: ${TS}"
          } > "${OWNER}/${REPO}/snapshots/${TS}.meta"
          git add -A
          if git diff --cached --quiet; then
            echo "No KB changes to push."
            exit 0
          fi
          git commit -m "KB snapshot: ${OWNER_REPO} @ ${TS}"
          git push origin HEAD:main

      - name: Check for changes
        id: diff
        shell: bash
        run: |
          set -euxo pipefail
          git add -A
          if git diff --cached --quiet; then
            echo "changed=false" >> "$GITHUB_OUTPUT"
          else:
            echo "changed=true" >> "$GITHUB_OUTPUT"
          fi

      - name: Create PR with AI fixes
        if: __GHA__ steps.diff.outputs.changed __GHA_END__ == 'true'
        uses: peter-evans/create-pull-request@v6
        with:
          token: __GHA__ secrets.BOT_TOKEN __GHA_END__
          branch: ai/airysdark-ai-autofix-__PTYPE__
          commit-message: "chore: AirysDark-AI auto-fix (__PTYPE__)"
          title: "AirysDark-AI: automated build fix (__PTYPE__)"
          body: |
            This PR was opened automatically by a build workflow after a failed build.
            - Build command: __GHA__ steps.build.outputs.BUILD_CMD __GHA_END__
            - Captured the failing build log
            - Proposed a minimal fix via AI
            - Committed the changes for review
          labels: "automation, ci"
""".lstrip("\n")

    setup_block = ""
    if setup.strip():
        setup_block = textwrap.indent(setup.rstrip("\n") + "\n", " " * 6)

    yml = (tmpl
           .replace("__SETUP__", setup_block.rstrip("\n"))
           .replace("__PTYPE__", target)
           .replace("__BUILD_CMD__", build_cmd.replace('"', '\\"'))
           .replace("__GHA__", "${{")
           .replace("__GHA_END__", "}}"))
    return yml


# ---------------- Android-only workflow generation (patched) ----------------

def write_workflow_android():
    """
    Android workflow calls the self-contained runner (tools/AirysDark-AI_android.py)
    which handles probe/loop/fix internally. No extra fetching here.
    Adds AI KB cache + optional KB push via KB_PUSH_TOKEN.
    Also adds a conservative fallback 'builder.py' invocation if BUILD_CMD was exposed.
    """
    yml = r"""
name: AirysDark-AI - Android (generated)

on:
  workflow_dispatch: {}

permissions:
  contents: write
  pull-requests: write

env:
  # Optional central KB repo; leave blank to skip
  KB_COLLECTION_REPO: "AirysDark-AI/ai-kb-collection"
  # on  = run AirysDark-AI_android.py (self-contained Android logic)
  # off = skip android.py and use AirysDark-AI_builder.py instead
  ANDROID_LOGIC: "on"

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
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

      - uses: actions/setup-java@v4
        with:
          distribution: temurin
          java-version: "17"
      - uses: android-actions/setup-android@v3
      - run: yes | sdkmanager --licenses
      - run: sdkmanager "platform-tools" "platforms;android-34" "build-tools;34.0.0"

      # ===== AI KB: restore cache =====
      - name: Restore AI KB cache
        uses: actions/cache@v4
        with:
          path: tools/ai_kb
          key: ai-kb-android-${{ github.repository }}-v1

      - name: Verify tools exist (no extra fetches here)
        run: |
          set -euxo pipefail
          test -f tools/AirysDark-AI_Request.py
          test -f tools/AirysDark-AI_android.py
          test -f tools/AirysDark-AI_builder.py || true
          ls -la tools

- name: Android strategy
        run: echo "ANDROID_LOGIC=${{ env.ANDROID_LOGIC }}"

      # ---------- ANDROID LOGIC: ON (uses AirysDark-AI_android.py) ----------
      - name: Run Android AI loop
        if: ${{ env.ANDROID_LOGIC == 'on' }}
        id: android_run
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          OPENAI_MODEL: ${{ vars.OPENAI_MODEL || 'gpt-4o-mini' }}
          AI_BUILDER_ATTEMPTS: "3"
          AI_LOG_TAIL: "160"
        run: |
          set -euxo pipefail
          python3 tools/AirysDark-AI_android.py --mode run | tee /tmp/android.out
          # (optional) forward any discovered command
          if grep -q '^BUILD_CMD=' /tmp/android.out; then
            CMD=$(grep -E '^BUILD_CMD=' /tmp/android.out | sed 's/^BUILD_CMD=//')
            echo "BUILD_CMD=$CMD" >> "$GITHUB_OUTPUT"
          fi

      # ---------- ANDROID LOGIC: OFF (uses AirysDark-AI_builder.py) ----------
      - name: Determine Gradle build command
        if: ${{ env.ANDROID_LOGIC == 'off' }}
        id: plan
        shell: bash
        run: |
          set -euxo pipefail
          CMD=""
          if [ -x ./gradlew ]; then
            CMD="./gradlew assembleDebug --stacktrace"
          else
            F=$(git ls-files '**/gradlew' | head -n1 || true)
            if [ -n "$F" ]; then
              cd "$(dirname "$F")"
              CMD="./gradlew assembleDebug --stacktrace"
            fi
          fi
          echo "BUILD_CMD=$CMD"
          echo "BUILD_CMD=$CMD" >> "$GITHUB_OUTPUT"

      - name: Build (capture)
        if: ${{ env.ANDROID_LOGIC == 'off' }}
        id: build
        shell: bash
        run: |
          set -euxo pipefail
          CMD="${{ steps.plan.outputs.BUILD_CMD }}"
          [ -n "$CMD" ] || { echo "No build command found"; exit 1; }
          set +e; bash -lc "$CMD" | tee build.log; EXIT=$?; set -e
          echo "EXIT_CODE=$EXIT" >> "$GITHUB_OUTPUT"
          exit 0
        continue-on-error: true

      - name: Attempt AI auto-fix with builder.py
        if: ${{ env.ANDROID_LOGIC == 'off' && steps.build.outputs.EXIT_CODE != '0' }}
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          OPENAI_MODEL: ${{ vars.OPENAI_MODEL || 'gpt-4o-mini' }}
          BUILD_CMD: ${{ steps.plan.outputs.BUILD_CMD }}
        run: python3 tools/AirysDark-AI_builder.py || true

      - name: Run Android AI loop
        id: run
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          OPENAI_MODEL: ${{ vars.OPENAI_MODEL || 'gpt-4o-mini' }}
          AI_BUILDER_ATTEMPTS: "3"
          AI_LOG_TAIL: "160"
        run: |
          set -euxo pipefail
          python3 tools/AirysDark-AI_android.py --mode run | tee /tmp/android.out
          if grep -q '^BUILD_CMD=' /tmp/android.out; then
            CMD=$(grep -E '^BUILD_CMD=' /tmp/android.out | sed 's/^BUILD_CMD=//')
            echo "BUILD_CMD=$CMD" >> "$GITHUB_OUTPUT"
          fi

      # Opportunistic fallback: if BUILD_CMD is present and build.log indicates failure, try builder.py
      - name: Attempt AI auto-fix via builder (fallback)
        if: always() && steps.run.outputs.BUILD_CMD != ''
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          OPENAI_MODEL: ${{ vars.OPENAI_MODEL || 'gpt-4o-mini' }}
          BUILD_CMD: ${{ steps.run.outputs.BUILD_CMD }}
        run: |
          set -euxo pipefail
          if [ -f build.log ]; then
            python3 tools/AirysDark-AI_builder.py || true
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
            tools/ai_kb/**

      # ===== AI KB: save cache and optional push to collection =====
      - name: Save AI KB cache
        if: always()
        uses: actions/cache/save@v4
        with:
          path: tools/ai_kb
          key: ai-kb-android-${{ github.repository }}-v1

      - name: Push KB snapshot to central collection repo
        if: always() && env.KB_COLLECTION_REPO != '' && secrets.KB_PUSH_TOKEN != ''
        shell: bash
        run: |
          set -euxo pipefail
          if [ ! -s tools/ai_kb/knowledge.jsonl ]; then
            echo "No knowledge.jsonl to push; skipping."
            exit 0
          fi
          OWNER_REPO="${GITHUB_REPOSITORY}"
          OWNER="${OWNER_REPO%%/*}"
          REPO="${OWNER_REPO#*/}"
          TS="$(date -u +'%Y-%m-%dT%H-%M-%SZ')"
          WORKDIR="$(mktemp -d)"
          git config --global user.name "airysdark-ai-bot"
          git config --global user.email "airysdark-ai-bot@users.noreply.github.com"
          git clone "https://x-access-token:${{ secrets.KB_PUSH_TOKEN }}@github.com/${{ env.KB_COLLECTION_REPO }}.git" "$WORKDIR/kb"
          cd "$WORKDIR/kb"
          mkdir -p "${OWNER}/${REPO}/snapshots"
          cp -f "$GITHUB_WORKSPACE/tools/ai_kb/knowledge.jsonl" "${OWNER}/${REPO}/knowledge.jsonl"
          cp -f "$GITHUB_WORKSPACE/tools/ai_kb/knowledge.jsonl" "${OWNER}/${REPO}/snapshots/${TS}.jsonl"
          {
            echo "repo: ${OWNER_REPO}"
            echo "run_id: ${GITHUB_RUN_ID}"
            echo "run_url: https://github.com/${OWNER_REPO}/actions/runs/${GITHUB_RUN_ID}"
            echo "ref: ${GITHUB_REF}"
            echo "timestamp: ${TS}"
          } > "${OWNER}/${REPO}/snapshots/${TS}.meta"
          git add -A
          if git diff --cached --quiet; then
            echo "No KB changes to push."
            exit 0
          fi
          git commit -m "KB snapshot: ${OWNER_REPO} @ ${TS}"
          git push origin HEAD:main

      - name: Stage changes
        id: diff
        run: |
          git add -A
          if git diff --cached --quiet; then
            echo "changed=false" >> "$GITHUB_OUTPUT"
          else:
            echo "changed=true" >> "$GITHUB_OUTPUT"
          fi

      - name: Create PR with Android changes
        if: ${{ steps.diff.outputs.changed == 'true' }}
        uses: peter-evans/create-pull-request@v6
        with:
          token: ${{ secrets.BOT_TOKEN }}
          branch: ai/airysdark-ai-android-loop
          commit-message: "chore: Android AI loop changes"
          title: "AirysDark-AI: Android AI loop changes"
          body: |
            Android AI loop updated files (build fixes or config changes).
            - Build command: ${{ steps.run.outputs.BUILD_CMD }}
            - Logs: see artifact "android-ai-loop"
          labels: automation, ci
""".lstrip("\n")
    ANDROID_WF.write_text(yml, encoding="utf-8")
    return str(ANDROID_WF)


# ---------------- OpenAI (optional) ----------------

def call_openai(prompt: str) -> str:
    import requests
    api = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api:
        raise RuntimeError("no_openai_key")
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a CI assistant. Return only a valid GitHub Actions YAML workflow."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        },
        timeout=180,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"openai_error {r.status_code}: {r.text[:400]}")
    out = r.json()["choices"][0]["message"]["content"].strip()
    out = re.sub(r"^```[a-zA-Z]*\n", "", out)
    out = re.sub(r"\n```$", "", out)
    return out

def build_ai_prompt(context: dict, target: str, build_cmd: str) -> str:
    files_list = "\n".join(context["repo"]["files"][:200])
    detector_types = ", ".join(context["detector"].get("types", []))
    return textwrap.dedent(f"""\
    Draft a GitHub Actions workflow named "AirysDark-AI - Build ({target})".
    Requirements:
    - triggers: workflow_dispatch only
    - job: ubuntu-latest
    - set up toolchain for {target} (java/android, cmake/meson, node, python, etc. as appropriate)
    - run build command: {build_cmd}
    - capture output to build.log, export EXIT_CODE via job outputs
    - if build fails, build llama.cpp (CMake, -DLLAMA_CURL=OFF), download TinyLlama GGUF, run "python3 tools/AirysDark-AI_builder.py"
    - upload build.log and .pre_ai_fix.patch as artifacts
    - if git staged changes exist, open a PR with peter-evans/create-pull-request@v6 using token ${{{{ secrets.BOT_TOKEN }}}}, title & commit message referencing target, labels "automation, ci"
    - Do not fetch tools in this workflow; they were committed by the detector step
    - Return ONLY YAML (no code fences)

    Hints:
    - Detected types: {detector_types}
    - Proposed build: {build_cmd}

    Partial file list (first 200):
    {files_list}
    """)


# ---------------- Main ----------------

def main():
    target = os.getenv("TARGET", "__SET_ME__")

    # 1) Load detector scan
    scan = read_scan_json()
    types = scan.get("types", [])
    evidence = scan.get("evidence", {})

    # 2) Deep repo snapshot
    snapshot = repo_snapshot()

    # 3) Propose a build command for TARGET
    build_cmd = propose_build_cmd(target)

    # 3b) Android deep probe (extra data & better recommendation)
    android_info = None
    if (target or "").lower() == "android":
        android_info = android_deep_probe()
        if android_info.get("cmd_recommendation"):
            build_cmd = android_info["cmd_recommendation"]

    # 4) Compose probe report (JSON) — keep AND append log history
    report = {
        "target": target,
        "proposed_build_cmd": build_cmd,
        "detector": {"types": types, "evidence": evidence},
        "repo": snapshot,
        "android_probe": android_info or {},
    }
    PROB_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with PROB_LOG.open("a", encoding="utf-8") as f:
        from datetime import datetime, timezone
        f.write(f"\n==== PROBE @ {datetime.now(timezone.utc).isoformat()} ====\n")
        f.write(json.dumps(report, indent=2) + "\n")

    # 5) Generate workflow
    generated_path = None
    if (target or "").lower() == "android":
        generated_path = write_workflow_android()
        used_ai = False
    else:
        used_ai = False
        try:
            prompt = build_ai_prompt(report, target, build_cmd)
            wf_text = call_openai(prompt)
            used_ai = True
            AI_OUT.write_text(wf_text, encoding="utf-8")
            must_have = ["workflow_dispatch", "build.log", "peter-evans/create-pull-request"]
            if not all(s in wf_text for s in must_have):
                wf_text = render_build_workflow(target, build_cmd)
                used_ai = False
        except Exception:
            wf_text = render_build_workflow(target, build_cmd)
            used_ai = False
        BUILD_WF.write_text(wf_text, encoding="utf-8")
        generated_path = str(BUILD_WF)

    # 6) PR body file
    body = []
    body.append("### AirysDark-AI: Probe results")
    body.append("")
    if types:
        body.append("**Detected build types (from detector):**")
        for t in types:
            body.append(f"- {t}")
            for ev in evidence.get(t, []):
                body.append(f"  - {ev}")
    else:
        body.append("_No build types detected by detector._")
    body.append("")
    body.append(f"**Selected target:** `{target}`")
    body.append(f"**Proposed build command:** `{build_cmd}`")
    if android_info:
        body.append("")
        body.append("**Android probe details:**")
        body.append(f"- wrappers: {android_info.get('wrappers')}")
        body.append(f"- modules: {android_info.get('modules')}")
        if android_info.get("task_guess"):
            body.append(f"- guessed tasks: {android_info['task_guess']}")
    body.append("")
    body.append(f"- Workflow written: `{generated_path}`")
    body.append(f"- AI drafted workflow: {'yes' if used_ai else 'no (template/runner path)'}")
    PR_BODY.write_text("\n".join(body) + "\n", encoding="utf-8")

    # 7) Ensure something changed for the PR step
    (TOOLS / ".ai_probe_touch").write_text(str(__import__("time").time()))

    print("✅ Probe complete")
    print(f"  Target: {target}")
    print(f"  Build cmd: {build_cmd}")
    print(f"  Report: {PROB_JSON}")
    print(f"  Log:    {PROB_LOG}")
    if ANDROID_WF.exists():
        print(f"  Android workflow: {ANDROID_WF}")
    if BUILD_WF.exists():
        print(f"  Generic workflow: {BUILD_WF}")
    print(f"  PR body: {PR_BODY}")
    return 0

if __name__ == "__main__":
    sys.exit(main())