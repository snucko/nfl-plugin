#!/usr/bin/env python3
import os
import pathlib
import subprocess
import time
import requests

# -----------------------------------------------------------------------------
# Environment & defaults
# -----------------------------------------------------------------------------
APP_URL = os.environ.get("APP_URL", "http://localhost:4567")
TRMNL_CONTAINER_NAME = os.environ.get("TRMNL_CONTAINER_NAME", "trmnlp")
AI_PATCH_MAX_ATTEMPTS = int(os.environ.get("AI_PATCH_MAX_ATTEMPTS", "3"))
WAIT_BOOT_SECS = int(os.environ.get("WAIT_BOOT_SECS", "10"))
ACCEPTANCE_HINT = os.environ.get(
    "ACCEPTANCE_HINT",
    "NFL games should render in the correct sections/columns with accurate team/date labels; no misplacements.",
)

# Local mode = use Transformers pipelines instead of HF Inference API
USE_LOCAL_MODE = os.environ.get("USE_LOCAL_MODE", "1") == "1"

# Lighter defaults for CI reliability
CAPTION_MODEL_ID = os.environ.get("CAPTION_MODEL_ID", "nlpconnect/vit-gpt2-image-captioning")
TEXT_MODEL_ID = os.environ.get("TEXT_MODEL_ID", "google/flan-t5-base")
TEXT_MODEL_ARCH = os.environ.get("TEXT_MODEL_ARCH", "seq2seq")  # or "causal"

# Avoid requiring hf_transfer on runners
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

ARTIFACTS_DIR = pathlib.Path("artifacts")
SCREENSHOT_PATH = ARTIFACTS_DIR / "plugin_screenshot.png"
LAST_DIFF_PATH = ARTIFACTS_DIR / "last_diff.patch"
RAW_OUT_PATH = ARTIFACTS_DIR / "raw_model_output.txt"
ERR_TXT = ARTIFACTS_DIR / "error.txt"
SHOT_ERR = ARTIFACTS_DIR / "screenshot_err.txt"

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def run(cmd: str, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess:
    print(f"$ {cmd}")
    try:
        return subprocess.run(cmd, shell=True, check=check, capture_output=capture_output)
    except subprocess.CalledProcessError as e:
        # Preserve stderr/stdout to artifacts for easier debugging
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        if e.stdout:
            (ARTIFACTS_DIR / "last_cmd_stdout.txt").write_text(
                e.stdout.decode("utf-8", errors="ignore"), encoding="utf-8"
            )
        if e.stderr:
            (ARTIFACTS_DIR / "last_cmd_stderr.txt").write_text(
                e.stderr.decode("utf-8", errors="ignore"), encoding="utf-8"
            )
        raise

def take_screenshot():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    # Capture stdout/stderr so Playwright issues land in artifacts/last_cmd_*.txt
    run(f"node tools/shot.js {APP_URL} {SCREENSHOT_PATH}", capture_output=True)

def start_sim():
    # Nuke old container (ignore errors), then start fresh
    run(f"docker rm -f {TRMNL_CONTAINER_NAME} >/dev/null 2>&1 || true", check=False)
    run(
        f"docker run --name {TRMNL_CONTAINER_NAME} "
        f"-d -p 4567:4567 -v \"$PWD:/plugin\" trmnl/trmnlp serve"
    )
    # Give it a head start
    time.sleep(max(1, WAIT_BOOT_SECS))

def wait_http_ready(url: str, timeout: int = 60, interval: float = 1.5) -> bool:
    """Poll URL until we get any non-5xx response."""
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code < 500:  # 2xx/3xx/4xx means server is up
                print(f"✅ App responding at {url} (status {r.status_code})")
                return True
        except Exception as e:
            last_err = e
        time.sleep(interval)
    print(f"⚠️ App did not become ready at {url}: {last_err}")
    return False

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
# Local caption + diff (Transformers pipelines)
# -----------------------------------------------------------------------------
def _safe_caption_pipeline(model=None, model_id=None, **kwargs):
    """
    Build an image-to-text pipeline. Accepts either `model` or `model_id`, plus **kwargs.
    Falls back to use_safetensors=False when checkpoints lack safetensors.
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
            # Retry without safetensors requirement
            return pipeline("image-to-text", model=mid, device="cpu", use_safetensors=False, **kwargs)
        raise

def call_local_caption_and_diff(image_path: pathlib.Path, prompt: str) -> str:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

    # 1) Caption the screenshot
    cap = _safe_caption_pipeline(model_id=CAPTION_MODEL_ID)
    captions = cap(str(image_path))
    caption_text = captions[0].get("generated_text", "") if captions else "(no caption)"

    # 2) Ask text model to emit a unified diff based on caption + instructions
    if TEXT_MODEL_ARCH == "seq2seq":
        tok = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
        model = AutoModelForSeq2SeqLM.from_pretrained(TEXT_MODEL_ID)
        text_pipe = pipeline("text2text-generation", model=model, tokenizer=tok)
        full_prompt = (
            f"Image description: {caption_text}\n\n"
            f"{prompt}\n\n"
            "Remember: output ONLY a valid unified diff starting with 'diff --git'."
        )
        out = text_pipe(full_prompt, max_new_tokens=512)
        text = out[0]["generated_text"]
    else:
        tok = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(TEXT_MODEL_ID)
        text_pipe = pipeline("text-generation", model=model, tokenizer=tok)
        full_prompt = (
            f"Image description: {caption_text}\n\n"
            f"{prompt}\n\n"
            "Return ONLY a valid unified diff starting with 'diff --git'."
        )
        out = text_pipe(full_prompt, max_new_tokens=512, do_sample=False)
        text = out[0]["generated_text"]

    RAW_OUT_PATH.write_text(text, encoding="utf-8")
    return text

# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Start simulator first, then wait for readiness before any screenshot
    start_sim()
    wait_http_ready(APP_URL, timeout=max(20, WAIT_BOOT_SECS + 20))

    # Initial screenshot of current state
    take_screenshot()

    for attempt in range(1, AI_PATCH_MAX_ATTEMPTS + 1):
        print(f"\n=== Patch Attempt {attempt}/{AI_PATCH_MAX_ATTEMPTS} ===")

        if not USE_LOCAL_MODE:
            raise RuntimeError("Remote HF API mode disabled in this build (USE_LOCAL_MODE=1 expected).")

        diff_text = call_local_caption_and_diff(SCREENSHOT_PATH, ACCEPTANCE_HINT)

        # Save model output regardless
        LAST_DIFF_PATH.write_text(diff_text, encoding="utf-8")

        if not diff_text.strip().startswith("diff --git"):
            print("❌ Model did not return a unified diff header. See artifacts/raw_model_output.txt & last_diff.patch.")
            # Try next attempt (lets model refine on the new screenshot)
            continue

        # Apply the patch; do not fail hard here (allow empty/no-op diffs)
        run(f"git apply {LAST_DIFF_PATH}", check=False)
        run('git commit -am "🤖 Local auto-patch for TRMNL NFL plugin" || true', check=False)

        # Rebuild, wait, re-screenshot
        start_sim()
        wait_http_ready(APP_URL, timeout=max(20, WAIT_BOOT_SECS + 20))
        take_screenshot()

        # One patch per run is enough; break here. Increase attempts if you want iterative refinement.
        break

    print("Done.")

# -----------------------------------------------------------------------------
# Entrypoint with error capture
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback, sys
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        err = traceback.format_exc()
        ERR_TXT.write_text(err, encoding="utf-8")
        print(err, file=sys.stderr)
        sys.exit(1)
