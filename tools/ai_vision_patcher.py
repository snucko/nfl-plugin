#!/usr/bin/env python3
import os, subprocess, sys, re, traceback
from pathlib import Path
from transformers import pipeline

# ---- Environment ----
APP_URL = "http://localhost:4567"
ARTIFACTS_DIR = Path("artifacts")
SCREENSHOT_PATH = ARTIFACTS_DIR / "plugin_screenshot.png"

# Disable hf_transfer to avoid missing dependency
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# ---- Helpers ----
def run(cmd, check=True, capture_output=False):
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=check, capture_output=capture_output)

def start_sim():
    """Start TRMNL simulator in Docker."""
    run("docker rm -f trmnlp >/dev/null 2>&1 || true", check=False)
    run("docker run --name trmnlp -d -p 4567:4567 -v \"$PWD:/plugin\" trmnl/trmnlp serve")

def take_screenshot():
    run(f"node tools/shot.js {APP_URL} {SCREENSHOT_PATH}")

def _safe_caption_pipeline(model=None, model_id=None, **kwargs):
    """
    Build an image-to-text pipeline. Accepts either `model_id` or `model`.
    Retries without safetensors if necessary.
    """
    mid = model_id or model
    if not mid:
        raise ValueError("Must provide model_id or model")
    try:
        return pipeline("image-to-text", model=mid, device="cpu", **kwargs)
    except Exception as e:
        msg = str(e)
        if "safetensors" in msg.lower():
            return pipeline("image-to-text", model=mid, device="cpu", use_safetensors=False, **kwargs)
        raise

def call_local_caption_and_diff(image_path, prompt):
    cap = _safe_caption_pipeline(model_id="nlpconnect/vit-gpt2-image-captioning")
    caption = cap(str(image_path))[0]["generated_text"]
    print("🧠 Caption:", caption)

    from transformers import pipeline as gen_pipeline
    gen = gen_pipeline("text2text-generation", model="google/flan-t5-base", device="cpu")
    full_prompt = (
        "You are an AI code patching assistant.\n"
        "Given this screenshot caption and task description, output ONLY a valid unified diff.\n"
        "If no change is needed, output an empty diff with correct header.\n\n"
        f"Caption: {caption}\nTask: {prompt}\n"
    )
    out = gen(full_prompt, max_new_tokens=512)[0]["generated_text"]
    (ARTIFACTS_DIR / "raw_model_output.txt").write_text(out, encoding="utf-8")

    # Extract unified diff
    m = re.search(r"(?ms)^(\*\*\* Begin Patch.*End Patch)", out)
    if not m:
        raise ValueError("Missing unified diff header in model output.")
    return m.group(1)

# ---- Main ----
def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # Step 1: screenshot
    start_sim()
    take_screenshot()

    # Step 2: describe patch task
    prompt = "Update plugin code based on screenshot differences."

    # Step 3: caption + diff
    diff_text = call_local_caption_and_diff(SCREENSHOT_PATH, prompt)

    # Step 4: save patch
    patch_file = ARTIFACTS_DIR / "last_diff.patch"
    patch_file.write_text(diff_text, encoding="utf-8")

    # Step 5: apply patch
    run(f"git apply -p0 {patch_file} || true", check=False)
    run('git commit -am "🤖 Local auto-patch for TRMNL NFL plugin" || true', check=False)

    # Step 6: rebuild & rescreenshot
    start_sim()
    take_screenshot()
    print("✅ Done.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        err = traceback.format_exc()
        (ARTIFACTS_DIR / "error.txt").write_text(err, encoding="utf-8")
        print(err, file=sys.stderr)
        sys.exit(1)
