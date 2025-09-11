#!/usr/bin/env python3
import os
import pathlib
import subprocess
import requests
import time
import json

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
APP_URL = os.environ.get("APP_URL", "http://localhost:4567")
TRMNL_CONTAINER_NAME = os.environ.get("TRMNL_CONTAINER_NAME", "trmnlp")
AI_PATCH_MAX_ATTEMPTS = int(os.environ.get("AI_PATCH_MAX_ATTEMPTS", "3"))
WAIT_BOOT_SECS = int(os.environ.get("WAIT_BOOT_SECS", "10"))
ACCEPTANCE_HINT = os.environ.get("ACCEPTANCE_HINT", "NFL plugin should display correctly")

USE_LOCAL_MODE = os.environ.get("USE_LOCAL_MODE", "0") == "1"
CAPTION_MODEL_ID = os.environ.get("CAPTION_MODEL_ID", "nlpconnect/vit-gpt2-image-captioning")
TEXT_MODEL_ID = os.environ.get("TEXT_MODEL_ID", "google/flan-t5-base")
TEXT_MODEL_ARCH = os.environ.get("TEXT_MODEL_ARCH", "seq2seq")

ARTIFACTS_DIR = pathlib.Path("artifacts")
SCREENSHOT_PATH = ARTIFACTS_DIR / "plugin_screenshot.png"

# Force disable hf_transfer to avoid requiring that package in CI
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def run(cmd, check=True, capture_output=False):
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=check, capture_output=capture_output)

def take_screenshot():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    run(f"node tools/shot.js {APP_URL} {SCREENSHOT_PATH}")

def start_sim():
    run(f"docker rm -f {TRMNL_CONTAINER_NAME} >/dev/null 2>&1 || true", check=False)
    run(f"docker run --name {TRMNL_CONTAINER_NAME} -d -p 4567:4567 -v \"$PWD:/plugin\" trmnl/trmnlp serve")
    time.sleep(WAIT_BOOT_SECS)

def make_prompt(snap: dict[str, str], hint: str) -> str:
    inv = "\n".join(f"- {k}" for k in sorted(snap.keys()))
    return (
        "You are an automated code patcher for a TRMNL plugin.\n"
        "Return ONLY a valid unified diff. No prose. No backticks. No explanations.\n"
        "Your first line MUST be exactly: diff --git a/<path> b/<path>\n"
        "Paths must be relative to repo root shown below.\n\n"
        "Example format (for illustration only):\n"
        "diff --git a/src/full.liquid b/src/full.liquid\n"
        "--- a/src/full.liquid\n"
        "+++ b/src/full.liquid\n"
        "@@ -1,3 +1,4 @@\n"
        "-<div class=\"title\">Old</div>\n"
        "+<!-- patch -->\n"
        "+<div class=\"title\">Old</div>\n"
        " <div class=\"body\">...</div>\n"
        "\n"
        "Repo file inventory:\n" + inv + "\n\n"
        "Acceptance:\n" + hint + "\n"
    )

# -----------------------------------------------------------------------------
# Local caption + diff (transformers pipeline)
# -----------------------------------------------------------------------------
def _safe_caption_pipeline(model=None, model_id=None, **kwargs):
    """
    Build an image-to-text pipeline. Accepts either `model` or `model_id`,
    plus **kwargs. Falls back to use_safetensors=False when checkpoints
    lack safetensors.
    """
    from transformers import pipeline
    mid = model_id or model
    if not mid:
        raise ValueError("Must provide `model_id` or `model`")

    try:
        return pipeline("image-to-text", model=mid, device="cpu", **kwargs)
    except Exception as e:
        msg = str(e)
        if "safetensors" in msg.lower():
            return pipeline("image-to-text", model=mid, device="cpu", use_safetensors=False, **kwargs)
        raise

def call_local_caption_and_diff(image_path: pathlib.Path, prompt: str) -> str:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

    cap = _safe_caption_pipeline(model_id=CAPTION_MODEL_ID)
    captions = cap(str(image_path))
    caption_text = captions[0]["generated_text"] if captions else "(no caption)"

    if TEXT_MODEL_ARCH == "seq2seq":
        tok = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
        model = AutoModelForSeq2SeqLM.from_pretrained(TEXT_MODEL_ID)
        text_pipe = pipeline("text2text-generation", model=model, tokenizer=tok)
    else:
        tok = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(TEXT_MODEL_ID)
        text_pipe = pipeline("text-generation", model=model, tokenizer=tok)

    full_prompt = f"Image description: {caption_text}\n\n{prompt}"
    out = text_pipe(full_prompt, max_new_tokens=300, temperature=0.0)
    return out[0]["generated_text"]

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    take_screenshot()

    for attempt in range(1, AI_PATCH_MAX_ATTEMPTS + 1):
        print(f"\n=== Patch Attempt {attempt}/{AI_PATCH_MAX_ATTEMPTS} ===")
        if USE_LOCAL_MODE:
            diff_text = call_local_caption_and_diff(SCREENSHOT_PATH, ACCEPTANCE_HINT)
        else:
            raise RuntimeError("Remote HF API mode disabled in this build.")

        if not diff_text.strip().startswith("diff --git"):
            print("❌ Model did not return a diff.")
            ARTIFACTS_DIR.joinpath("last_diff.patch").write_text(diff_text, encoding="utf-8")
            continue

        ARTIFACTS_DIR.joinpath("last_diff.patch").write_text(diff_text, encoding="utf-8")

        # Try applying patch
        run("git apply artifacts/last_diff.patch", check=False)
        run('git commit -am "🤖 Local auto-patch for TRMNL NFL plugin" || true', check=False)

        # Rebuild & re-screenshot
        start_sim()
        take_screenshot()
        break

    print("Done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback, sys
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        err = traceback.format_exc()
        (ARTIFACTS_DIR / "error.txt").write_text(err, encoding="utf-8")
        print(err, file=sys.stderr)
        sys.exit(1)
