"""CS-Audit demo.

python3 demo.py                      two built-in examples + real Soniox examples
python3 demo.py --live raw/memo_b.wav    send a recording to Soniox now and audit it
                                     (needs SONIOX_API_KEY; references from references.tsv)
"""
import json
import os
import sys
from pathlib import Path

import csaudit as ca

det = ca.train_detector(ca.KZ_SENTENCES, ca.CS_SENTENCES)


def show(ref_text, hyp_text):
    ref, hyp = ca.normalise(ref_text), ca.normalise(hyp_text)
    errs = ca.classify(ref, hyp, det)
    e, n = ca.wer_counts(ref, hyp)
    print(f"REFERENCE : {ref_text}")
    print(f"ASR OUTPUT: {hyp_text}")
    print(f"code-switched (detector): {det.is_code_switched(ref)}")
    print(f"WER = {e}/{n} = {e / n:.2f}")
    i = 0
    while i < len(errs):
        _, c, r, h = errs[i]
        if c == "E6":  # group repeated hallucinated words
            k = i
            while k < len(errs) and errs[k][1] == "E6" and errs[k][3] == h:
                k += 1
            print(f"  {c} x{k - i}: repeated/invented '{h}'  ({ca.CLASS_NAMES[c]})")
            i = k
            continue
        print(f"  {c}: ref='{r or '-'}' -> hyp='{h or '-'}'  ({ca.CLASS_NAMES[c]})")
        i += 1
    print()


def builtin():
    print("=== 1. Constructed examples (errors of known type) ===\n")
    show("Мен ертең собеседованиеге барамын", "Мен ертең sobesedovaniege бардым")
    show("Біз бұл мәселені бірге шештік", "Біз бұл мәселені бірге шештік шештік шештік шештік")
    p = Path("real_results.json")
    if p.exists():
        R = json.load(open(p, encoding="utf-8"))
        print(f"=== 2. Real ASR output (Soniox {R['utterances'][0]['model']}, own recordings) ===\n")
        for x in R["utterances"]:
            if x["id"] in ("04", "11", "20"):
                show(x["ref"], x["hyp"])
        s = R["summary"]
        print(f"All 20 phrases: WER Kazakh {s['wer_kz'][2]:.3f}, code-switched {s['wer_cs'][2]:.3f}; "
              f"classes {s['class_counts']}")


def live(audio):
    import soniox_transcribe as st
    import requests
    import soniox_eval as se
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {os.environ['SONIOX_API_KEY']}"
    print(f"Sending {audio} to Soniox ...")
    toks = st.transcribe(s, Path(audio), st.MODELS[0])
    rec = Path(audio).stem
    refs = [r for r in se.load_refs() if r["rec"] == rec]
    hyp = ca.normalise("".join(t["text"] for t in toks))
    for r, h in zip(refs, se.split_hyp(refs, hyp)):
        show(r["text"], " ".join(h))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--live":
        live(sys.argv[2])
    else:
        builtin()
