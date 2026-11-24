"""Send audio files to the Soniox async API and save the raw token JSON.
Usage: python3 soniox_transcribe.py out_dir file1.wav [file2.wav ...]
Needs SONIOX_API_KEY in the environment."""
import json, os, sys, time
from pathlib import Path
import requests

API = "https://api.soniox.com"
MODELS = ["stt-async-v3", "stt-async-preview"]

def transcribe(s, path, model):
    with open(path, "rb") as f:
        r = s.post(f"{API}/v1/files", files={"file": (path.name, f)}); r.raise_for_status()
    fid = r.json()["id"]
    r = s.post(f"{API}/v1/transcriptions", json={"model": model, "file_id": fid,
        "language_hints": ["kk", "ru"], "enable_language_identification": True})
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code} {r.text[:300]}")
    tid = r.json()["id"]
    while True:
        st = s.get(f"{API}/v1/transcriptions/{tid}").json()
        if st["status"] == "completed": break
        if st["status"] == "error": raise RuntimeError(st)
        time.sleep(1.5)
    toks = s.get(f"{API}/v1/transcriptions/{tid}/transcript").json()["tokens"]
    s.delete(f"{API}/v1/transcriptions/{tid}"); s.delete(f"{API}/v1/files/{fid}")
    return toks

def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    s = requests.Session(); s.headers["Authorization"] = f"Bearer {os.environ['SONIOX_API_KEY']}"
    for p in map(Path, sys.argv[2:]):
        for m in MODELS:
            try:
                toks = transcribe(s, p, m); break
            except RuntimeError as e:
                print(f"[{p.name}] model {m} failed: {e}")
        else:
            continue
        (out / f"{p.stem}.json").write_text(json.dumps({"model": m, "tokens": toks}, ensure_ascii=False, indent=1))
        print(f"[{p.name}] model={m}: " + "".join(t["text"] for t in toks))

if __name__ == "__main__":
    main()
