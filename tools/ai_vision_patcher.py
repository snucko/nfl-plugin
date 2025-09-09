#!/usr/bin/env python3
import os, sys, time, base64, pathlib, re, tempfile, subprocess, requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
ALLOWED_DIRS = [ROOT / "src", ROOT / "sample"]
ALLOWED_FILES = {ROOT / ".trmnlp.yml"}
SCREENSHOT_PATH = ROOT / "artifacts" / "plugin_screenshot.png"
SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)

APP_URL = os.getenv("APP_URL", "http://localhost:4567")
CONTAINER_NAME = os.getenv("TRMNL_CONTAINER_NAME", "trmnlp")
START_CMD = os.getenv(
    "START_CMD",
    f"docker rm -f {CONTAINER_NAME} >/dev/null 2>&1 || true && "
    f"docker run --name {CONTAINER_NAME} -d -p 4567:4567 -v \"$PWD:/plugin\" trmnl/trmnlp serve"
)
STOP_CMD  = os.getenv("STOP_CMD", f"docker rm -f {CONTAINER_NAME} >/dev/null 2>&1 || true")
WAIT_BOOT_SECS = int(os.getenv("WAIT_BOOT_SECS", "8"))

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_URL = os.getenv("HF_MODEL_URL", "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-VL-7B-Instruct")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
MAX_ATTEMPTS = int(os.getenv("AI_PATCH_MAX_ATTEMPTS", "3"))

def run(cmd, check=True):
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

def take_screenshot(url=APP_URL, out=SCREENSHOT_PATH):
    shot_js = ROOT / "tools" / "shot.js"
    shot_js.parent.mkdir(parents=True, exist_ok=True)
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
    run("npx --yes playwright install --with-deps chromium")
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

def encode_image(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()

def make_prompt(snap, hint):
    inv = "\n".join(f"- {k}" for k in sorted(snap.keys()))
    return f"""You are an automated code patcher for a TRMNL plugin (Liquid + YAML/JSON).

Goal:
- Inspect the attached UI screenshot of the plugin at runtime.
- If the visual layout/data placement is incorrect, output a **UNIFIED DIFF** that minimally fixes the plugin so the UI matches expectations.

Allowed files:
- .trmnlp.yml
- src/**/*.liquid
- sample/**/*.json

Rules:
- Return ONLY a valid unified diff (no prose).
- Keep changes minimal and targeted.
- Ensure Liquid compiles and the plugin renders.

Repo file inventory:
{inv}

Acceptance (high-level):
{hint}
"""

def call_hf_vision(image_data_url, prompt):
    payload = {
        "inputs": [
            {"role":"user","content":[
                {"type":"image_url","image_url":{"url": image_data_url}},
                {"type":"text","text": prompt}
            ]}
        ],
        "parameters":{"max_new_tokens":1200,"temperature":0.2}
    }
    r = requests.post(HF_MODEL_URL, headers=HEADERS, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    txt = ""
    if isinstance(data, list):
        txt = data[0].get("generated_text") or data[0].get("content") or ""
    else:
        txt = data.get("generated_text") or data.get("content") or ""
    return (txt or "").strip()

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
    if not re.search(r"^diff --git a/", diff_text, re.M):
        return False, "Missing unified diff header."
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
    if not HF_TOKEN:
        print("Missing HF_TOKEN"); sys.exit(2)

    start_sim()
    take_screenshot()

    acceptance_hint = os.getenv(
        "ACCEPTANCE_HINT",
        "Each NFL game must render under the correct section/column with correct team names, dates, and ordering."
    )

    for i in range(1, MAX_ATTEMPTS+1):
        print(f"\n=== Vision Patch Attempt {i} ===")
        prompt = make_prompt(read_snapshot(), acceptance_hint)
        img = encode_image(SCREENSHOT_PATH)

        print("🧠 Calling Hugging Face vision model…")
        diff_text = call_hf_vision(img, prompt)

        ok, msg = apply_unified_diff(diff_text)
        print(("✅" if ok else "❌"), msg)
        if not ok:
            time.sleep(2)
            continue

        # Commit patch
        run('git config user.name "github-actions"', check=False)
        run('git config user.email "github-actions@github.com"', check=False)
        run('git add -A')
        run('git commit -m "🤖 Vision auto-patch for TRMNL NFL plugin" || true', check=False)

        # Rebuild and rescreenshot
        start_sim()
        take_screenshot()

    print("Done.")

if __name__ == "__main__":
    main()

