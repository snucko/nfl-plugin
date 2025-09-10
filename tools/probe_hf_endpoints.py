#!/usr/bin/env python3
import os, requests, base64, pathlib

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Missing HF_TOKEN"); raise SystemExit(2)
H = {"Authorization": f"Bearer {HF_TOKEN}"}

img_models = [
  "Salesforce/blip-image-captioning-base",
  "Salesforce/blip-image-captioning-large",
  "microsoft/Florence-2-base"
]
text_models = [
  # text/code models that often work on serverless; we’ll take the first 200
  "google/flan-t5-base",
  "gpt2",
  "bigcode/starcoder2-3b",
  "HuggingFaceH4/zephyr-7b-beta",
]

def ping(url, payload):
  r = requests.post(url, headers=H, json=payload, timeout=60)
  return r.status_code, r.text[:250]

def dataurl_png(path):
  b = pathlib.Path(path).read_bytes()
  return "data:image/png;base64," + base64.b64encode(b).decode()

print("\n=== IMAGE → TEXT candidates ===")
img = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
for m in img_models:
  u = f"https://api-inference.huggingface.co/models/{m}"
  code, body = ping(u, {"inputs": img})
  print(f"{m:40} -> {code}")

print("\n=== TEXT → TEXT/CODE candidates ===")
for m in text_models:
  u = f"https://api-inference.huggingface.co/models/{m}"
  code, body = ping(u, {"inputs": "Return the word OK."})
  print(f"{m:40} -> {code}")
