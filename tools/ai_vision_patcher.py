#!/usr/bin/env python3
import os, sys, time, base64, pathlib, re, tempfile, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
ALLOWED_DIRS = [ROOT / "src", ROOT / "sample"]
ALLOWED_FILES = {ROOT / ".trmnlp.yml"}
SCREENSHOT_PATH = ROOT / "artifacts" / "plugin_screenshot.png"
SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)

APP_URL = os.getenv("APP_URL", "http://localhost:4567")
CONTAINER_NAME = os.getenv("TRMNL_CONTAINER_NAME", "trmnlp")
START_CMD = os.getenv("START_CMD",
    f"docker rm -f {CONTAINER_NAME} >/dev/null 2>&1 || true && "
    f"docker run --name {CONTAINER_NAME} -d -p 4567:4567 -v \"$PWD:/plugin\" trmnl/trmnlp serve"
)
STOP_CMD = os.getenv("STOP_CMD", f"docker rm -f {CONTAINER_NAME} >/dev/null 2>&1 || true")
WAIT_BOOT_SECS = int(os.getenv("WAIT_BOOT_SECS", "10"))

USE_LOCAL_MODE = os.getenv("USE_LOCAL_MODE", "0") == "1"
MAX_ATTEMPTS = int(os.getenv("AI_PATCH_MAX_ATTEMPTS", "3"))
ENABLE_VALIDATOR = os.getenv("ENABLE_VALIDATOR", "0") == "1"

def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {cmd}")
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if p.stdout: print(p.stdout)
    if p.stderr: print(p.stderr, file=sys.stderr)
    if check and p.returncode != 0:
        raise SystemExit(p.returncode)
    return p

def start_sim():
    run(STOP_CMD, check=False)
    run(START_CMD)
    time.sleep(WAIT_BOOT_SECS)

def ensure_playwright_installed():
    run("command -v npm || (sudo apt-get update && sudo apt-get install -y npm)", check=False)
    run("test -f package.json || npm init -y", check=False)
    run("npm install playwright", check=False)
    run("npx --yes playwright install --with-deps chromium")

def take_screenshot(url: str = APP_URL, out: pathlib.Path = SCREENSHOT_PATH):
    shot_js = ROOT / "tools" / "shot.js"
    shot_js.parent.mkdir(parents=True, exist_ok=True)
    if not shot_js.exists():
        shot_js.write_text("""
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const context = await browser.newContext({ deviceScaleFactor: 2 });
  const page = await context.newPage();
  await page.goto(process.argv[2], { waitUntil: 'networkidle' });
  await page.waitForTimeout(1500);
  await page.screenshot({ path: process.argv[3], fullPage: true, type: 'png' });
  await browser.close();
})();
""")
    ensure_playwright_installed()
    run(f"node {shot_js} {url} {out}")
    if not out.exists():
        raise RuntimeError("Screenshot not captured.")

def list_files():
    files = []
    for d in ALLOWED_DIRS:
        if d.exists():
            for p in d.rglob("*"):
                if p.is_file() and p.suffix in {".liquid",".json",".yml",".yaml",".css",".js",".ts",".tsx",".html"}:
                    files.append(p)
    for p in ALLOWED_FILES:
        if p.exists():
            files.append(p)
    return files

def read_snapshot():
    return {p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8", errors="ignore")
            for p in list_files()}

def make_prompt(snap: dict[str,str], hint: str) -> str:
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

# -------- Local caption + diff (BLIP + FLAN-T5) --------
def call_local_caption_and_diff(image_path: pathlib.Path, prompt: str) -> str:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    cap = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    caption = cap(str(image_path))[0]["generated_text"]
    tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    strict = (
        "You are an automated code patcher. Reply ONLY with a valid unified diff.\n"
        "Start with 'diff --git a/'. No commentary. No backticks.\n\n"
        f"Screenshot caption: {caption}\n\n" + prompt
    )
    inputs = tok(strict, return_tensors="pt", truncation=True)
    out = mdl.generate(**inputs, max_new_tokens=512, do_sample=False, num_beams=1, repetition_penalty=1.05)
    return tok.decode(out[0], skip_special_tokens=True).strip()

# -------- Diff utils --------
def is_valid_diff(diff_text: str) -> bool:
    if not re.search(r"^diff --git a/.+ b/.+", diff_text, re.M):
        return False
    if not re.search(r"^@@ .+ @@", diff_text, re.M):
        return False
    if not re.search(r"^[+-].+", diff_text, re.M):
        return False
    return True

def synthesize_noop_diff() -> str:
    """
    Guaranteed-valid no-op patch: append one comment line to end of src/full.liquid
    Uses a zero-context hunk which always applies:
      @@ -N,0 +N,1 @@
      +<line>
    """
    target = ROOT / "src" / "full.liquid"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text("", encoding="utf-8")
    lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
    n = len(lines) + 1
    header = "diff --git a/src/full.liquid b/src/full.liquid\n--- a/src/full.liquid\n+++ b/src/full.liquid\n"
    hunk = f"@@ -{n},0 +{n},1 @@\n"
    body = "+<!-- autopatch noop: adjust layout later -->\n"
    return header + hunk + body

def diff_targets_allowed_paths(diff_text: str) -> bool:
    ok = True
    for m in re.finditer(r"^\+\+\+ b/(.+)$", diff_text, re.M):
        rel = m.group(1)
        abs_path = ROOT / rel
        within = any(str(abs_path).startswith(str(d) + os.sep) for d in ALLOWED_DIRS) or abs_path in ALLOWED_FILES
        if not within:
            print(f"❌ Blocked diff path: {rel}")
            ok = False
    return ok

def apply_unified_diff(diff_text: str):
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    last = artifacts / "last_diff.patch"

    # log whatever we’re about to try
    last.write_text(diff_text, encoding="utf-8")
    print(f"📝 wrote patch to {last}")

    if not is_valid_diff(diff_text):
        return False, "Invalid diff (missing hunks/+/-)."
    if not diff_targets_allowed_paths(diff_text):
        return False, "Diff targets blocked paths."

    with tempfile.TemporaryDirectory() as td:
        patch = pathlib.Path(td) / "patch.diff"
        patch.write_text(diff_text, encoding="utf-8")
        p = subprocess.run(["git","apply","--index",str(patch)], cwd=ROOT, capture_output=True, text=True)
        if p.returncode != 0:
            return False, f"git apply failed:\n{p.stdout}\n{p.stderr}"
    return True, "Applied."

def main():
    # preflight: Pillow
    try:
        import PIL  # noqa
    except Exception:
        print("Missing Pillow. Install with: pip install pillow", file=sys.stderr)
        sys.exit(2)

    start_sim()
    take_screenshot()

    acceptance_hint = os.getenv(
        "ACCEPTANCE_HINT",
        "Each NFL game must render under the correct section/column with correct team names, dates, and ordering."
    )

    for i in range(1, MAX_ATTEMPTS+1):
        print(f"\n=== Patch Attempt {i}/{MAX_ATTEMPTS} ===")
        prompt = make_prompt(read_snapshot(), acceptance_hint)

        if not USE_LOCAL_MODE:
            print("Remote HF API mode disabled in this build.")
            sys.exit(1)

        print("🧠 Calling local caption+diff (BLIP + FLAN-T5)…")
        diff_text = call_local_caption_and_diff(SCREENSHOT_PATH, prompt)

        if not is_valid_diff(diff_text):
            print("ℹ️ Model did not return a valid unified diff. First 400 chars:\n" + diff_text[:400])
            print("↪︎ Falling back to a safe no-op diff.")
            diff_text = synthesize_noop_diff()

        ok, msg = apply_unified_diff(diff_text)
        print(("✅" if ok else "❌"), msg)
        if not ok:
            # if even the no-op fails (shouldn’t), bail after logging
            if not is_valid_diff(diff_text):
                print("Saved invalid diff at artifacts/last_diff.patch")
            time.sleep(2)
            continue

        run('git config user.name "github-actions"', check=False)
        run('git config user.email "github-actions@github.com"', check=False)
        run('git add -A')
        run('git commit -m "🤖 Local auto-patch for TRMNL NFL plugin" || true', check=False)

        start_sim()
        take_screenshot()
        break

    print("Done.")

if __name__ == "__main__":
    main()
