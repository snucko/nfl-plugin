#!/usr/bin/env python3
"""
AI Vision Patcher for TRMNL plugin (local/inference-free mode)

Flow:
  1) Ensure TRMNL simulator is up (Docker).
  2) Screenshot the running plugin UI via Playwright (tools/shot.js).
  3) Use a *local* caption model (image->text) to describe UI.
  4) Feed caption + repo inventory + acceptance hint to a *local* text model
     (FLAN-T5 or similar) to ask for a unified diff that fixes layout/data issues.
  5) Validate + apply the patch, rebuild, and rescreenshot.

Artifacts:
  artifacts/plugin_screenshot.png
  artifacts/raw_model_output.txt
  artifacts/last_diff.patch
  artifacts/error.txt (on failure)

Notes:
  - No HF token required; models are downloaded from HF Hub anonymously.
  - Torch >= 2.6 enforced via workflow to satisfy HF security checks.
"""

import os
import sys
import json
import time
import shlex
import pathlib
import subprocess
from typing import Dict

# ---------- Config via env ----------
APP_URL = os.environ.get("APP_URL", "http://localhost:4567")
CONTAINER = os.environ.get("TRMNL_CONTAINER_NAME", "trmnlp")
WAIT_BOOT_SECS = int(os.environ.get("WAIT_BOOT_SECS", "10"))
ATTEMPTS = int(os.environ.get("AI_PATCH_MAX_ATTEMPTS", "3"))
USE_LOCAL_MODE = os.environ.get("USE_LOCAL_MODE", "1") == "1"

CAPTION_MODEL_ID = os.environ.get("CAPTION_MODEL_ID", "nlpconnect/vit-gpt2-image-captioning")
TEXT_MODEL_ID = os.environ.get("TEXT_MODEL_ID", "google/flan-t5-base")
TEXT_MODEL_ARCH = os.environ.get("TEXT_MODEL_ARCH", "seq2seq")  # seq2seq or causal

# Safety / caching
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# Disable fast hf_transfer (avoids requiring hf_transfer package on CI)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# ---------- Paths ----------
ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
ARTIFACTS_DIR = ROOT / "artifacts"
SCREENSHOT_PATH = ARTIFACTS_DIR / "plugin_screenshot.png"
RAW_OUT_PATH = ARTIFACTS_DIR / "raw_model_output.txt"
PATCH_PATH = ARTIFACTS_DIR / "last_diff.patch"

# ---------- Helpers ----------
def run(cmd: str, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess:
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=check, capture_output=capture_output)

def start_sim():
    # start docker container if not running
    run(f"docker rm -f {shlex.quote(CONTAINER)} >/dev/null 2>&1 || true", check=False)
    run(
        f"docker run --name {shlex.quote(CONTAINER)} -d -p 4567:4567 -v \"$PWD:/plugin\" trmnl/trmnlp serve",
        check=True,
    )
    time.sleep(WAIT_BOOT_SECS)

def take_screenshot():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    # Requires tools/shot.js + Playwright chromium installed
    run(f"node tools/shot.js {shlex.quote(APP_URL)} {shlex.quote(str(SCREENSHOT_PATH))}")

def repo_inventory() -> Dict[str, str]:
    out = run("git ls-files", capture_output=True)
    files = out.stdout.decode().strip().splitlines()
    inv = {}
    for rel in files:
        p = ROOT / rel
        try:
            size = p.stat().st_size
        except Exception:
            size = 0
        inv[rel] = f"{size}B"
    return inv

def make_prompt(snap: Dict[str, str], hint: str) -> str:
    inv = "\n".join(f"- {k}" for k in sorted(snap.keys()))
    return (
        "You are an automated code patcher for a TRMNL plugin.\n"
        "Return ONLY a valid unified diff. No prose. No backticks.\n"
        "Your first line MUST be exactly: diff --git a/<path> b/<path>\n"
        "Paths must be repo-relative. Keep changes minimal and targeted.\n\n"
        "Repo file inventory:\n" + inv + "\n\n"
        "Acceptance:\n" + hint + "\n"
    )

def _safe_caption_pipeline(model_id: str):
    """
    Build an image-to-text pipeline. Falls back to non-safetensors checkpoints if needed.
    """
    from transformers import pipeline
    try:
        return pipeline("image-to-text", model=model_id, device="cpu")
    except Exception as e:
        msg = str(e)
        if "safetensors" in msg.lower():
            return pipeline("image-to-text", model=model_id, device="cpu", use_safetensors=False)
        raise

def caption_image(img_path: pathlib.Path) -> str:
    print("🧠 Captioning screenshot…")
    pipe = _safe_caption_pipeline(CAPTION_MODEL_ID)
    out = pipe(str(img_path))
    # standard pipeline returns list of {"generated_text": "..."}
    if isinstance(out, list) and out and "generated_text" in out[0]:
        return out[0]["generated_text"].strip()
    if isinstance(out, str):
        return out.strip()
    return json.dumps(out)

def generate_patch(prompt: str) -> str:
    """
    Use a local text model to produce a unified diff.
    TEXT_MODEL_ARCH = seq2seq (FLAN-T5 etc.) or causal (tiny GPT-like).
    """
    print(f"🧠 Generating patch with {TEXT_MODEL_ID} ({TEXT_MODEL_ARCH})…")

    # Save raw output for debugging
    RAW_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if TEXT_MODEL_ARCH.lower() == "seq2seq":
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tok = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(TEXT_MODEL_ID)
        input_ids = tok(prompt, return_tensors="pt", truncation=True).input_ids
        out_ids = mdl.generate(input_ids, max_new_tokens=1024)
        text = tok.decode(out_ids[0], skip_special_tokens=True)
    else:
        # causal LM path
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
        mdl = AutoModelForCausalLM.from_pretrained(TEXT_MODEL_ID)
        input_ids = tok(prompt, return_tensors="pt", truncation=True).input_ids
        out_ids = mdl.generate(input_ids, max_new_tokens=1024, do_sample=False)
        text = tok.decode(out_ids[0], skip_special_tokens=True)

    RAW_OUT_PATH.write_text(text, encoding="utf-8")
    return text.strip()

def ensure_unified_diff(text: str) -> str:
    # Try to locate a unified diff header inside any surrounding text.
    idx = text.find("diff --git a/")
    if idx == -1:
        raise ValueError("Missing unified diff header in model output.")
    return text[idx:]

def apply_patch(diff_text: str) -> bool:
    PATCH_PATH.write_text(diff_text, encoding="utf-8")
    try:
        run(f"git apply -p0 {shlex.quote(str(PATCH_PATH))}")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ git apply failed:\n")
        try:
            err = e.stderr.decode()
        except Exception:
            err = str(e)
        print(err)
        return False

# ---------- Main ----------
def main():
    # 1) Boot simulator & screenshot
    start_sim()
    take_screenshot()

    # 2) Build prompt from caption + repo inventory
    acceptance = os.environ.get(
        "ACCEPTANCE_HINT",
        "NFL view must be correctly structured and labeled; no misplaced games."
    )
    caption = caption_image(SCREENSHOT_PATH)
    inv = repo_inventory()
    prompt = (
        "Screenshot caption:\n"
        f"{caption}\n\n"
        + make_prompt(inv, acceptance)
    )

    # 3) Try up to N attempts
    for attempt in range(1, ATTEMPTS + 1):
        print(f"\n=== Patch Attempt {attempt}/{ATTEMPTS} ===")
        try:
            model_out = generate_patch(prompt)
            diff_text = ensure_unified_diff(model_out)
        except Exception as e:
            # Persist error for CI logs, then fail fast on final attempt
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            (ARTIFACTS_DIR / "error.txt").write_text(str(e), encoding="utf-8")
            if attempt == ATTEMPTS:
                raise
            else:
                continue

        ok = apply_patch(diff_text)
        if not ok:
            if attempt == ATTEMPTS:
                raise SystemExit(1)
            continue

        # Commit changes and rebuild/screenshot to verify
        run('git add -A && git commit -m "🤖 Local auto-patch for TRMNL NFL plugin" || true', check=False)
        start_sim()
        take_screenshot()
        break

    print("Done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS_DIR / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)
