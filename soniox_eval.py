"""Real-data check: run CS-Audit on Soniox output for 20 read phrases
(10 Kazakh, 10 Kazakh-Russian). Reference text = the phrase the speaker read.

Inputs : references.tsv, soniox_raw/<recording>.json (from soniox_transcribe.py)
Output : real_results.json and a printed report.
Each recording holds several phrases; the hypothesis is split into phrases by
aligning it to the concatenated references.
"""
import json
from collections import Counter
from pathlib import Path
import csaudit as ca

def load_refs(path="references.tsv"):
    rows = [l.rstrip("\n").split("\t") for l in open(path, encoding="utf-8")][1:]
    out = []
    for pid, rec, typ, ref in rows:
        words, langs = ca.parse_marked(ref)
        out.append({"id": pid, "rec": rec, "type": typ, "text": ref.replace("[", "").replace("]", ""),
                    "words": words, "langs": langs})
    return out

def split_hyp(refs, hyp):
    """Assign hypothesis words to phrases via alignment against the joined refs."""
    joined, owner = [], []
    for k, r in enumerate(refs):
        joined += r["words"]; owner += [k] * len(r["words"])
    parts = [[] for _ in refs]
    cur = 0
    for op, i, j in ca.align(joined, hyp):
        if i is not None:
            cur = owner[i]
        if j is not None:
            parts[cur].append(hyp[j])
    return parts

def main():
    refs = load_refs()
    det = ca.train_detector(ca.KZ_SENTENCES, ca.CS_SENTENCES)
    results = []
    for rec in sorted({r["rec"] for r in refs}):
        group = [r for r in refs if r["rec"] == rec]
        raw = json.load(open(f"soniox_raw/{rec}.json", encoding="utf-8"))
        hyp_text = "".join(t["text"] for t in raw["tokens"])
        for r, h in zip(group, split_hyp(group, ca.normalise(hyp_text))):
            errs = ca.classify(r["words"], h, det)
            e, n = ca.wer_counts(r["words"], h)
            results.append({"id": r["id"], "type": r["type"], "ref": " ".join(r["words"]),
                            "hyp": " ".join(h), "errors": e, "n": n,
                            "classes": [(c, rw, hw) for _, c, rw, hw in errs],
                            "cs_gold": "ru" in r["langs"],
                            "cs_pred_nb": det.is_code_switched(r["words"]),
                            "cs_pred_rule": any(ca.letter_rule_lang(w) == "ru" for w in r["words"]),
                            "model": raw["model"]})
    print(f"Soniox model: {results[0]['model']}\n")
    for x in results:
        flag = "" if x["errors"] == 0 else "  <- " + ", ".join(
            f"{c}: {rw or '-'} -> {hw or '-'}" for c, rw, hw in x["classes"])
        print(f"[{x['id']}] {x['type']}  REF: {x['ref']}\n          HYP: {x['hyp']}{flag}")
    print()
    summ = {}
    for t in ("kz", "cs"):
        e = sum(x["errors"] for x in results if x["type"] == t)
        n = sum(x["n"] for x in results if x["type"] == t)
        summ[f"wer_{t}"] = (e, n, e / n)
        print(f"WER {t}: {e}/{n} = {e/n:.3f}")
    cnt = Counter(c for x in results for c, _, _ in x["classes"])
    summ["class_counts"] = {c: cnt[c] for c in ca.CLASSES}
    print("Error classes (CS-Audit):", summ["class_counts"])
    for name in ("nb", "rule"):
        c = Counter((x["cs_gold"], x[f"cs_pred_{name}"]) for x in results)
        p, r, f = ca.prf(c[(True, True)], c[(False, True)], c[(True, False)])
        summ[f"detector_{name}"] = {"P": p, "R": r, "F1": f}
        print(f"Code-switch detector ({name}) on 20 real references: P {p:.2f} R {r:.2f} F1 {f:.2f}")
    json.dump({"summary": summ, "utterances": results}, open("real_results.json", "w"),
              ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
