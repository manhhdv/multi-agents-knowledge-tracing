"""Reproduce the MathDial/CoMTA turn counts and multi-KC shares quoted in the paper.

Mirrors dialogue_kt.data_loading.load_annotated_data (typical_cutoff=1, CoMTA fold 1)
and dialogue_kt.kt_data_loading.apply_annotations without importing torch.
Usage: python split_counts.py /path/to/dialogue-kt  [out.csv]
"""
import sys
from ast import literal_eval
import pandas as pd

REPO = sys.argv[1] if len(sys.argv) > 1 else "../../../kse2026_api_eval/third_party/dialogue-kt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "00_run_info/split_counts.csv"
CONV = {c: literal_eval for c in ["dialogue", "meta_data", "annotation"]}


def apply_annotations(sample):
    dialogue = [dict(t) for t in sample["dialogue"]]
    anno = dict(sample["annotation"])
    if "error" in anno:
        return None
    if dialogue[0]["turn"] == 0:
        anno["turn 0"] = {"correct": None, "kcs": []}
    for t in dialogue:
        a = anno[f"turn {t['turn']}"]
        corr, kcs = a["correct"], a["kcs"]
        corr = None if not kcs else corr
        kcs = [] if corr is None else kcs
        t["correct"], t["kcs"] = corr, kcs
    if dialogue[-1]["kcs"]:
        md = sample["meta_data"]
        if "expected_result" in md:
            dialogue[-1]["correct"] = md["expected_result"] == "Answer Accepted"
        elif "self_correctness" in md and dialogue[-1]["correct"] is not None:
            sc = md["self_correctness"]
            if sc == "Yes":
                dialogue[-1]["correct"] = True
            elif sc == "Yes, but I had to reveal the answer":
                dialogue[-1]["correct"] = None
            elif sc == "No":
                dialogue[-1]["correct"] = False
    return dialogue


def count(df, skip_first):
    n_kcs, failed, with_rows = [], 0, 0
    for _, s in df.iterrows():
        try:
            d = apply_annotations(s)
        except Exception:
            d = None
        if not d:
            failed += 1
            continue
        first, before = True, len(n_kcs)
        for t in d:
            if t["correct"] is None:
                continue
            if skip_first and first:
                first = False
                continue
            first = False
            n_kcs.append(len(t["kcs"]))
        with_rows += len(n_kcs) > before
    return n_kcs, failed, with_rows


def typical(r):
    return r["meta_data"]["self_typical_confusion"] >= 1 and r["meta_data"]["self_typical_interactions"] >= 1


tr = pd.read_csv(f"{REPO}/data/annotated/mathdial_train_atc.csv", converters=CONV).sample(frac=1, random_state=221)
tr = tr[tr.apply(typical, axis=1)]
te = pd.read_csv(f"{REPO}/data/annotated/mathdial_test_atc.csv", converters=CONV)
te = te[te.apply(typical, axis=1)]
co = pd.read_csv(f"{REPO}/data/annotated/comta_atc.csv", converters=CONV)
cs = co.sample(frac=1, random_state=221)
frames = {
    ("mathdial", "train"): tr[: int(0.8 * len(tr))],
    ("mathdial", "val"): tr[int(0.8 * len(tr)):],
    ("mathdial", "test"): te,
    ("comta", "all"): co,
    ("comta", "fold1_val"): cs[int(len(cs) * 0.65): int(len(cs) * 0.8)],
    ("comta", "fold1_test"): cs[int(len(cs) * 0.8):],
}
rows = []
for (ds, split), df in frames.items():
    for skip in (False, True):
        k, failed, with_rows = count(df, skip)
        rows.append(dict(dataset=ds, split=split, skip_first_labelled_turn=skip, dialogues=len(df),
                         dialogues_failed=failed, dialogues_with_turns=with_rows, turns=len(k),
                         multi_kc_turns=sum(x > 1 for x in k), multi_kc_share=sum(x > 1 for x in k) / len(k)))
for ds, splits in [("mathdial", ["train", "val", "test"])]:
    sub = [r for r in rows if r["dataset"] == ds and not r["skip_first_labelled_turn"] and r["split"] in splits]
    n, m = sum(r["turns"] for r in sub), sum(r["multi_kc_turns"] for r in sub)
    rows.append(dict(dataset=ds, split="all", skip_first_labelled_turn=False, dialogues=sum(r["dialogues"] for r in sub),
                     dialogues_failed=sum(r["dialogues_failed"] for r in sub),
                     dialogues_with_turns=sum(r["dialogues_with_turns"] for r in sub),
                     turns=n, multi_kc_turns=m, multi_kc_share=m / n))
out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(out.to_string(index=False))
