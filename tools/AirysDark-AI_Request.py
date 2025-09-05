#!/usr/bin/env python3
# AirysDark-AI_Request.py
#
# Centralized AI client for the AirysDark-AI tools.
# - Primary: OpenAI (chat.completions)
# - Fallback: llama.cpp CLI
# - Handles long logs via safe truncation
# - Redacts known secrets from logs/context
# - Simple API:
#       text = request_ai(task, context_parts=[...], system="...", want_diff=False)
#   Returns assistant text (and if want_diff=True, a (text, diff_or_None) tuple).
#
# Env (override as needed):
#   PROVIDER=openai|llama
#   OPENAI_API_KEY=...             (required for openai)
#   OPENAI_MODEL=gpt-4o-mini       (default)
#   OPENAI_ORG=...                 (optional)
#   FALLBACK_PROVIDER=llama|none   (default: llama)
#   LLAMA_CPP_BIN=llama-cli
#   MODEL_PATH=models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
#   LLAMA_CTX=4096
#   AI_TEMPERATURE=0.2
#   AI_MAX_PROMPT_TOKENS=2500      (~chars ≈ tokens*4 heuristic)
#   AI_RETRIES=2                   (total attempts per provider)
#
# Notes:
# - We keep the interface text-first so caller scripts can plug any logs they like.
# - We redact obvious secrets (OPENAI_API_KEY / BOT_TOKEN / GH token patterns).
# - We provide extract_unified_diff() helper to consume model outputs that return diffs.

from __future__ import annotations
import os, re, json, time, subprocess, tempfile, pathlib, typing, textwrap

try:
    import requests
except Exception:
    # We tolerate missing requests on callers that won't use OpenAI.
    requests = None  # type: ignore

# -------------------- Config --------------------
PROVIDER          = os.getenv("PROVIDER", "openai").strip().lower()
FALLBACK_PROVIDER = os.getenv("FALLBACK_PROVIDER", "llama").strip().lower()

OPENAI_MODEL      = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_ORG        = os.getenv("OPENAI_ORG", "").strip()

LLAMA_CPP_BIN     = os.getenv("LLAMA_CPP_BIN", "llama-cli")
LLAMA_MODEL_PATH  = os.getenv("MODEL_PATH", "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

LLAMA_CTX         = int(os.getenv("LLAMA_CTX", "4096"))
TEMP              = float(os.getenv("AI_TEMPERATURE", "0.2"))
RETRIES           = int(os.getenv("AI_RETRIES", "2"))
MAX_PROMPT_TOKENS = int(os.getenv("AI_MAX_PROMPT_TOKENS", "2500"))  # 1 token ≈ 4 chars heuristic

# -------------------- Helpers --------------------
def _approx_char_limit(tokens: int) -> int:
    # crude but practical: ~4 chars per token
    return max(512, tokens * 4)

def _truncate(text: str, max_tokens: int) -> str:
    limit = _approx_char_limit(max_tokens)
    if len(text) <= limit:
        return text
    head = text[: int(limit*0.60)]
    tail = text[- int(limit*0.35):]
    return head + "\n\n[...truncated to fit context...]\n\n" + tail

_SECRET_PATTERNS = [
    (re.compile(r"(?:sk-|rk-)[A-Za-z0-9]{20,}"), "[REDACTED_API_KEY]"),        # OpenAI-like keys
    (re.compile(r"ghp_[A-Za-z0-9]{36,}"), "[REDACTED_GH_PAT]"),               # GitHub classic PAT
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "[REDACTED_GH_FG_PAT]"),    # GitHub fine-grained PAT
]
def redact(text: str) -> str:
    out = text
    # direct env values
    for env_key in ("OPENAI_API_KEY", "BOT_TOKEN", "GITHUB_TOKEN"):
        if os.getenv(env_key):
            val = re.escape(os.getenv(env_key, ""))
            if val:
                out = re.sub(val, f"[REDACTED_{env_key}]", out)
    # generic patterns
    for pat, repl in _SECRET_PATTERNS:
        out = pat.sub(repl, out)
    return out

def extract_unified_diff(s: str) -> str | None:
    """
    Try to extract a unified diff (---/+++ with @@ hunks).
    Returns the diff string if found, else None.
    """
    m = re.search(r"(?ms)^--- [^\n]+\n\+\+\+ [^\n]+\n(?:@@.*\n.*)+", s)
    if m:
        return s[m.start():].strip()
    # fallback: find first ---/+++ block
    m2 = re.search(r"(?ms)^--- [^\n]+\n\+\+\+ [^\n]+\n", s)
    if not m2:
        return None
    return s[m2.start():].strip()

def _assemble_prompt(task: str, context_parts: list[str] | None, want_diff: bool) -> str:
    """
    Build a single plain-text prompt for both providers.
    """
    goal = task.strip()
    if not goal:
        goal = "You are a helpful build assistant."

    header = [
        "You are an automated CI assistant.",
        "Read the provided context and return the best possible answer.",
    ]
    if want_diff:
        header.append("If proposing code/config changes, return ONLY a valid unified diff (---/+++ with @@ hunks).")

    blocks = []
    for idx, part in enumerate(context_parts or [], start=1):
        if not part:
            continue
        blocks.append(f"## Context {idx}\n{part.strip()}")

    prompt = "\n\n".join([
        "\n".join(header),
        f"## Task\n{goal}",
        *blocks
    ])

    # Safety: redact secrets and truncate
    prompt = redact(prompt)
    prompt = _truncate(prompt, MAX_PROMPT_TOKENS)
    return prompt

# -------------------- Providers --------------------
def _openai_call(messages: list[dict]) -> str:
    if requests is None:
        raise RuntimeError("requests module not available (needed for OpenAI).")
    if not OPENAI_API_KEY:
        raise RuntimeError("missing OPENAI_API_KEY for OpenAI provider.")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    if OPENAI_ORG:
        headers["OpenAI-Organization"] = OPENAI_ORG

    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": TEMP,
    }

    last_err = None
    for attempt in range(1, RETRIES+1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=180)
            if r.status_code >= 400:
                last_err = f"HTTP {r.status_code}: {r.text}"
                # Backoff for transient 429/5xx
                if r.status_code in (429, 500, 502, 503):
                    time.sleep(1.5 * attempt)
                    continue
                raise RuntimeError(f"OpenAI error: {last_err}")
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = str(e)
            time.sleep(1.0 * attempt)
    raise RuntimeError(f"OpenAI failed after {RETRIES} attempts: {last_err}")

def _llama_call(prompt: str) -> str:
    mp = pathlib.Path(LLAMA_MODEL_PATH)
    if not mp.exists():
        raise RuntimeError(f"llama model not found at: {mp}")
    args = [
        LLAMA_CPP_BIN,
        "-m", str(mp),
        "-p", prompt,
        "-n", "2048",
        "--temp", str(TEMP),
        "-c", str(LLAMA_CTX),
    ]
    last_err = None
    for attempt in range(1, RETRIES+1):
        try:
            p = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if p.returncode != 0:
                last_err = p.stdout
                time.sleep(0.8 * attempt)
                continue
            return p.stdout
        except Exception as e:
            last_err = str(e)
            time.sleep(0.8 * attempt)
    raise RuntimeError(f"llama.cpp failed after {RETRIES} attempts: {last_err}")

# -------------------- Public API --------------------
def request_ai(
    task: str,
    *,
    context_parts: list[str] | None = None,
    system: str | None = None,
    want_diff: bool = False,
    provider: str | None = None,
    fallback_provider: str | None = None,
) -> str | tuple[str, str | None]:
    """
    High-level request:
      - Builds a safe prompt from task + context_parts
      - Calls primary provider (default: env PROVIDER)
      - Falls back (default: llama) on error/quota
      - Returns assistant text (or (text, diff) if want_diff=True)
    """
    provider = (provider or PROVIDER).lower()
    fallback_provider = (fallback_provider or FALLBACK_PROVIDER).lower()

    prompt = _assemble_prompt(task, context_parts, want_diff)

    def _run_primary() -> str:
        if provider == "openai":
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt})
            return _openai_call(msgs)
        elif provider == "llama":
            return _llama_call(prompt)
        else:
            raise RuntimeError(f"Unknown PROVIDER='{provider}'")

    def _run_fallback() -> str:
        if fallback_provider == "llama":
            return _llama_call(prompt)
        elif fallback_provider == "openai":
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt})
            return _openai_call(msgs)
        elif fallback_provider in ("", "none", "null"):
            raise RuntimeError("No fallback provider configured.")
        else:
            raise RuntimeError(f"Unknown FALLBACK_PROVIDER='{fallback_provider}'")

    text = ""
    try:
        text = _run_primary()
    except Exception as e:
        # only fallback for quota/transient or if configured
        try:
            text = _run_fallback()
        except Exception as e2:
            raise RuntimeError(f"AI request failed (primary + fallback): {e} // {e2}")

    if want_diff:
        return text, extract_unified_diff(text)
    return text

# -------------------- CLI (useful for quick tests) --------------------
def _cli():
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="Short instruction for the AI")
    ap.add_argument("--context", action="append", default=[], help="Add context block (repeatable)")
    ap.add_argument("--want-diff", action="store_true", help="Try extracting unified diff from the answer")
    args = ap.parse_args()

    out = request_ai(args.task, context_parts=args.context, want_diff=args.want_diff)
    if isinstance(out, tuple):
        text, diff = out
        print("\n===== AI TEXT =====\n")
        print(text)
        print("\n===== EXTRACTED DIFF =====\n")
        print(diff or "(no diff)")
    else:
        print(out)

if __name__ == "__main__":
    _cli()