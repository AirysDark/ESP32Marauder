#!/usr/bin/env python3
# AirysDark-AI_detector.py
#
# Step 1 of the pipeline (Detector):
#   - Scans the repo (all subfolders) for build systems
#   - Uses folder-name hints & CMake content to refine detection
#   - Writes:
#       tools/airysdark_ai_scan.json  (types + evidence + hints + sample files)
#       pr_body_detect.md             (nice PR body listing detections & next steps)
#       .github/workflows/AirysDark-AI_prob.yml  (manual-run probe workflow)
#
# The PROBE workflow (run later) will create the build workflow PR.

import os
import json
import pathlib
import textwrap
from typing import Dict, List, Tuple

ROOT = pathlib.Path(os.getenv("PROJECT_DIR", ".")).resolve()
WF_DIR = ROOT / ".github" / "workflows"
TOOLS_DIR = ROOT / "tools"
WF_DIR.mkdir(parents=True, exist_ok=True)
TOOLS_DIR.mkdir(parents=True, exist_ok=True)

SCAN_JSON = TOOLS_DIR / "airysdark_ai_scan.json"
PR_BODY_DETECT = ROOT / "pr_body_detect.md"

# ---------------- Full-repo scan ----------------

def scan_all_files(max_files=10000) -> List[Tuple[str, pathlib.Path]]:
    files = []
    for r, ds, fs in os.walk(ROOT):
        if ".git" in ds:
            ds.remove(".git")
        for fn in fs:
            p = pathlib.Path(r) / fn
            try:
                rel = p.relative_to(ROOT)
            except Exception:
                rel = p
            files.append((fn.lower(), rel))
            if len(files) >= max_files:
                return files
    return files

def read_text_safe(p: pathlib.Path) -> str:
    try:
        return (ROOT / p).read_text(errors="ignore")
    except Exception:
        return ""

def collect_dir_name_hints(files: List[Tuple[str, pathlib.Path]]) -> List[str]:
    names = set()
    for _, rel in files:
        for part in pathlib.Path(rel).parts:
            names.add(part.lower())
    # return as sorted list for JSON stability
    return sorted(names)

# ---------------- CMake classifier ----------------

ANDROID_HINTS = (
    "android", "android_abi", "android_platform", "ndk", "cmake_android", "gradle", "externalnativebuild",
    "find_library(log)", "log-lib", "loglib",
)

DESKTOP_HINTS = (
    "add_executable", "pkgconfig", "find_package(", "threads", "pthread", "x11", "wayland", "gtk", "qt",
    "set(cmake_system_name linux",
)

def cmakelists_flavor(cm_txt: str) -> str:
    t = cm_txt.lower()
    if any(h in t for h in ANDROID_HINTS):
        return "android"
    if any(h in t for h in DESKTOP_HINTS):
        return "desktop"
    return "desktop"  # default bias

# ---------------- Detection ----------------

def detect_types() -> Tuple[List[str], Dict[str, List[str]]]:
    files = scan_all_files()
    fnames = [n for n, _ in files]
    rels   = [str(p).lower() for _, p in files]
    dir_hints = collect_dir_name_hints(files)

    evidence: Dict[str, List[str]] = {}
    types: List[str] = []

    def add(t: str, why: str):
        if t not in types:
            types.append(t)
        evidence.setdefault(t, []).append(why)

    # Folder-name hints (broad)
    if "linux" in dir_hints:
        add("linux", "folder hint: 'linux' present in path segments")
    if "android" in dir_hints:
        add("android", "folder hint: 'android' present in path segments")
    if "windows" in dir_hints:
        add("windows", "folder hint: 'windows' present in path segments")

    # Android / Gradle
    for name, rel in files:
        if name == "gradlew":
            add("android", f"found wrapper: {rel}")
        if name.startswith("build.gradle"):
            add("android", f"found gradle: {rel}")
        if name.startswith("settings.gradle"):
            add("android", f"found gradle settings: {rel}")

    # CMake
    cmake_paths = [rel for (n, rel) in files if n == "cmakelists.txt"]
    if cmake_paths:
        add("cmake", f"found {len(cmake_paths)} CMakeLists.txt")
        for p in cmake_paths:
            txt = read_text_safe(p)
            flavor = cmakelists_flavor(txt)
            if flavor == "desktop":
                add("linux", f"CMakeLists suggests desktop build: {p}")
            else:
                add("android", f"CMakeLists suggests Android/NDK: {p}")

    # Linux umbrella
    for name, rel in files:
        if name in ("makefile", "gnumakefile") or name.endswith(".mk"):
            add("linux", f"found make build file: {rel}")
        if name == "meson.build":
            add("linux", f"found Meson build: {rel}")
        if name == "build.ninja":
            add("ninja", f"found Ninja build: {rel}")

    # Node
    for name, rel in files:
        if name == "package.json":
            add("node", f"found package.json: {rel}")
            break

    # Python
    for name, rel in files:
        if name == "pyproject.toml":
            add("python", f"found pyproject.toml: {rel}")
            break
    for name, rel in files:
        if name == "setup.py":
            add("python", f"found setup.py: {rel}")
            break

    # Rust
    for name, rel in files:
        if name == "cargo.toml":
            add("rust", f"found Cargo.toml: {rel}")
            break

    # .NET
    for name, rel in files:
        if name.endswith(".sln") or name.endswith(".csproj") or name.endswith(".fsproj"):
            add("dotnet", f"found .NET project: {rel}")
            break

    # Maven
    for name, rel in files:
        if name == "pom.xml":
            add("maven", f"found pom.xml: {rel}")
            break

    # Flutter
    for name, rel in files:
        if name == "pubspec.yaml":
            add("flutter", f"found pubspec.yaml: {rel}")
            break

    # Go
    for name, rel in files:
        if name == "go.mod":
            add("go", f"found go.mod: {rel}")
            break

    # Bazel
    for name, rel in files:
        base = os.path.basename(str(rel)).lower()
        if name in ("workspace", "workspace.bazel", "module.bazel") or base in ("build", "build.bazel"):
            add("bazel", f"found Bazel file: {rel}")
            break

    # SCons
    for name, rel in files:
        if name in ("sconstruct", "sconscript"):
            add("scons", f"found SCons file: {rel}")
            break

    if not types:
        add("unknown", "no known build system files detected")

    # Add top-level hints and a short file sample to the scan JSON
    sample_files = [str(p) for _, p in files[:200]]
    scan = {
        "types": types,
        "evidence": evidence,
        "dir_hints": dir_hints,
        "sample_files": sample_files,
    }
    SCAN_JSON.write_text(json.dumps(scan, indent=2), encoding="utf-8")

    return types, evidence, dir_hints, sample_files

# ---------------- PROBE workflow template (patched for KB + tokens) ----------------

def write_prob_workflow():
    """Write a single manual-run probe workflow with KB caching/collection & proper tokens."""
    yml = textwrap.dedent("""\
    name: AirysDark-AI - Probe (LLM builds workflow)

    on:
      workflow_dispatch: {}

    permissions:
      contents: write
      pull-requests: write

    # Set TARGET to one of the detected types: android, linux, cmake, node, python, rust, dotnet, maven, flutter, go
    env:
      TARGET: "__SET_ME__"
      # Optional: central KB collection repo to receive snapshots (owner/repo)
      KB_COLLECTION_REPO: "AirysDark-AI/ai-kb-collection"

    jobs:
      probe:
        runs-on: ubuntu-latest
        steps:
          - name: Guard: require manual confirmation
          if: ${{ github.event_name == 'workflow_dispatch' }}
          run: |
            if [ "${{ inputs.confirm }}" != "YES" ]; then
            echo "Refusing to run: inputs.confirm must be YES"; exit 1
            fi

         - name: Guard: default branch only
         run: |
           if [ "${{ github.ref }}" != "refs/heads/$
{{ github.event.repository.default_branch }}" ]; then
            echo "This workflow only runs on the default branch."; exit 1
           fi
           
          - name: Checkout (no credentials)
            uses: actions/checkout@v4
            with:
              fetch-depth: 0
              persist-credentials: false

          - name: Setup Python
            uses: actions/setup-python@v5
            with:
              python-version: "3.11"

          # ===== AI KB: Restore local cache =====
          - name: Restore AI KB cache
            uses: actions/cache@v4
            with:
              path: tools/ai_kb
              key: ai-kb-${{ github.repository }}-v1

          - name: Verify tools exist (added by detector PR)
            shell: bash
            run: |
              set -euxo pipefail
              test -f tools/AirysDark-AI_prob.py
              test -f tools/AirysDark-AI_builder.py
              ls -la tools
              echo "TARGET=$TARGET"

          - name: Run repo probe (AI-assisted)
            shell: bash
            env:
              TARGET: ${{ env.TARGET }}
              OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}   # optional
              OPENAI_MODEL: ${{ vars.OPENAI_MODEL || 'gpt-4o-mini' }}
            run: |
              set -euxo pipefail
              python3 tools/AirysDark-AI_prob.py

          - name: Upload probe artifacts
            if: always()
            uses: actions/upload-artifact@v4
            with:
              name: airysdark-ai-probe-artifacts
              if-no-files-found: warn
              retention-days: 7
              path: |
                tools/airysdark_ai_prob_report.json
                tools/airysdark_ai_prob_report.log
                tools/airysdark_ai_build_ai_response.txt
                pr_body_build.md
                .github/workflows/AirysDark-AI_build.yml

          # ===== AI KB: Upload as artifact =====
          - name: Upload AI KB (artifact)
            if: always()
            uses: actions/upload-artifact@v4
            with:
              name: ai-kb
              path: tools/ai_kb/**
              if-no-files-found: warn
              retention-days: 30

          # ===== AI KB: Save cache back =====
          - name: Save AI KB cache
            if: always()
            uses: actions/cache/save@v4
            with:
              path: tools/ai_kb
              key: ai-kb-${{ github.repository }}-v1

          # ===== AI KB: Push snapshot to central collection repo (optional) =====
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

          - name: Stage generated build workflow
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

          - name: Open PR with generated build workflow
            if: steps.diff.outputs.changed == 'true'
            uses: peter-evans/create-pull-request@v6
            with:
              token: ${{ secrets.BOT_TOKEN }}
              branch: ai/airysdark-ai-build
              commit-message: "chore: add AirysDark-AI_build.yml (from probe)"
              title: "AirysDark-AI: add build workflow (from probe)"
              body-path: pr_body_build.md
              labels: automation, ci
    """)
    (WF_DIR / "AirysDark-AI_prob.yml").write_text(yml, encoding="utf-8")
    print(f"✅ Wrote: {WF_DIR}/AirysDark-AI_prob.yml")

# ---------------- PR body (Detector) ----------------

def write_pr_body_detect(types: List[str], evidence: Dict[str, List[str]]):
    lines = []
    lines.append("### AirysDark-AI: detector results")
    lines.append("")
    if types:
        lines.append("**Detected build types:**")
        for t in types:
            lines.append(f"- {t}")
            for ev in evidence.get(t, [])[:6]:  # limit per type to keep PR body short
                lines.append(f"  - {ev}")
    else:
        lines.append("_No build types detected._")
    lines.append("")
    lines.append("**Next steps:**")
    lines.append("1. Edit **`.github/workflows/AirysDark-AI_prob.yml`** and set `env.TARGET` to the build you want (e.g. `android`, `linux`, `cmake`).")
    lines.append("2. Merge this PR.")
    lines.append("3. From the Actions tab, manually run **AirysDark-AI - Probe (LLM builds workflow)**.")
    lines.append("")
    PR_BODY_DETECT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✅ Wrote: {PR_BODY_DETECT}")

# ---------------- Main ----------------

def main():
    types, evidence, dir_hints, sample_files = detect_types()
    print("Detected types:", types)
    write_prob_workflow()
    write_pr_body_detect(types, evidence)
    print("Done. Generated PROBE workflow + scan JSON + PR body.")

if __name__ == "__main__":
    raise SystemExit(main())
