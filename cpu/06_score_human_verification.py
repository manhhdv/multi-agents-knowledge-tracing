"""06 - CHAY SAU KHI GAN NHAN. Neo cac con so dong thuan vao phan quyet cua nguoi.
Dong y kien #3 (ca ba ban nhan xet): 74.4%% / 75.4%% / 69.2%% hien la dong thuan voi nhan GPT-4o,
chua phai do chinh xac so voi dung/sai da duoc nguoi xac minh.

Cach dung:
  python3 06_score_human_verification.py --merge   # gop A+B -> dien san adjudication, danh dau bat dong
  (hai nguoi thong nhat cot 'final' cho cac dong bat dong)
  python3 06_score_human_verification.py           # cham diem

QUAN TRONG: mau 139 luot duoc PHAN TANG theo 6 mau dong thuan voi san 15 luot/tang, nen
ty le tho KHONG phai uoc luong cho 1.985 luot. Script tu dong ap trong so tang
(strata_weights.csv) de ra uoc luong cho toan bo tap test, kem bootstrap trong tang."""
import os, sys, json, math
import numpy as np, pandas as pd

def _find_root():
    env = os.environ.get("MATHKT_ROOT")
    if env and os.path.isdir(os.path.join(env, "results")):
        return os.path.abspath(env)
    seeds = []
    try: seeds.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError: pass
    seeds.append(os.getcwd())
    for s in seeds:
        p = s
        for _ in range(6):
            for q in (p, os.path.join(p, "MathKT-Agent_SoICT2026")):   # goi tai lap chep san trong revision_kit
                if os.path.isdir(os.path.join(q, "results")) and os.path.isdir(os.path.join(q, "raw_runs")):
                    return q
            p = os.path.dirname(p)
    raise SystemExit("Khong tim thay goc goi (can thu muc chua ca results/ va raw_runs/). "
                     "Dat bien moi truong MATHKT_ROOT.")

def _outdir():
    try: return os.path.dirname(os.path.abspath(__file__))
    except NameError: return os.getcwd()

ROOT = _find_root(); OUT = _outdir()
def rp(*a): return os.path.join(ROOT, *a)
def op(*a): return os.path.join(OUT, *a)

import argparse
HERE = OUT
A_P, B_P = op("xac_minh_A.xlsx"), op("xac_minh_B.xlsx")
ADJ_P = op("adjudication_xac_minh.xlsx")
for p in (A_P, B_P, ADJ_P):
    if not os.path.exists(p):
        # cho phep dat file gan nhan canh script hoac trong ../labelling/
        alt = os.path.join(OUT, "..", "labelling", os.path.basename(p))
        if os.path.exists(alt): globals()[["A_P","B_P","ADJ_P"][(A_P,B_P,ADJ_P).index(p)]] = alt

VALID = {"dung", "sai", "chua_ro"}
def read_labels(path, col="dung_sai (dung/sai/chua_ro)"):
    d = pd.read_excel(path, sheet_name="Xac minh")
    lab = d[col].astype(str).str.strip().str.lower().replace({"nan": ""})
    return d[["item_id"]].assign(label=lab)

def do_merge():
    A, B = read_labels(A_P), read_labels(B_P)
    adj = pd.read_excel(ADJ_P, sheet_name="Thong nhat")
    m = adj.drop(columns=["dung_sai_A", "dung_sai_B", "trang_thai", "final"], errors="ignore") \
           .merge(A.rename(columns={"label": "dung_sai_A"}), on="item_id") \
           .merge(B.rename(columns={"label": "dung_sai_B"}), on="item_id")
    m["trang_thai"] = np.where(m.dung_sai_A == m.dung_sai_B, "dong_thuan", "bat_dong")
    m["final"] = np.where(m.trang_thai == "dong_thuan", m.dung_sai_A, "")
    cols = ["stt","item_id","dialogue","turn_id","de_bai","ngu_canh","luot_hoc_sinh","dap_an",
            "dung_sai_A","ghi_chu_A","dung_sai_B","ghi_chu_B","trang_thai","final","final_ghi_chu"]
    m = m.reindex(columns=[c for c in cols if c in m.columns])
    outp = op("adjudication_xac_minh_MERGED.xlsx")
    with pd.ExcelWriter(outp, engine="openpyxl") as w:
        m.to_excel(w, sheet_name="Thong nhat", index=False)
    n_dis = int((m.trang_thai == "bat_dong").sum())
    from sklearn.metrics import cohen_kappa_score
    k = cohen_kappa_score(m.dung_sai_A, m.dung_sai_B)
    print("Gop xong: %d dong, dong thuan tho %.1f%% (Cohen kappa %.3f), con %d dong can thong nhat."
          % (len(m), 100 * (m.trang_thai == "dong_thuan").mean(), k, n_dis))
    print("-> %s  (dien cot 'final' cho %d dong bat_dong roi luu de len adjudication_xac_minh.xlsx)"
          % (os.path.basename(outp), n_dis))

def do_score():
    src = op("adjudication_xac_minh_MERGED.xlsx") if os.path.exists(op("adjudication_xac_minh_MERGED.xlsx")) else ADJ_P
    adj = pd.read_excel(src, sheet_name="Thong nhat")
    adj["final"] = adj["final"].astype(str).str.strip().str.lower()
    miss = adj[~adj["final"].isin(VALID)]
    if len(miss):
        raise SystemExit("Con %d dong chua co nhan 'final' hop le (dung/sai/chua_ro): %s"
                         % (len(miss), list(miss.item_id)[:10]))
    key = pd.read_csv(rp("supplementary", "human_verification", "human_verification_key.csv"))
    wts = pd.read_csv(op("strata_weights.csv") if os.path.exists(op("strata_weights.csv"))
                      else os.path.join(OUT, "strata_weights.csv"), index_col=0)
    d = adj[["item_id", "final"]].merge(key, on="item_id")
    d = d[d["final"] != "chua_ro"].copy()
    d["human"] = np.where(d["final"] == "dung", "correct", "incorrect")
    d["w"] = d.pattern.map(wts.weight)
    JJ = ("correct", "incorrect")
    systems = {"GPT-4o reference label": d.label.map({1: "correct", 0: "incorrect"}),
               "SAVA": d.sava_rerun.astype(str), "Numeric regex": d.numeric_regex.astype(str),
               "LLM judge (binary)": d.llm_judge_binary.astype(str),
               "LLM judge (three-way)": d.llm_judge_three_way.astype(str)}
    rows = []
    rng = np.random.default_rng(221)
    for name, v in systems.items():
        judged = v.isin(JJ)
        sub = d[judged]; vv = v[judged]
        hit = (vv == sub.human).astype(float).values; w = sub.w.values
        est = float(np.sum(hit * w) / np.sum(w))
        cov = float(np.sum(d.w[judged]) / np.sum(d.w))
        boots = []
        for _ in range(2000):                      # bootstrap TRONG tung tang
            idx = np.concatenate([rng.choice(np.where(sub.pattern.values == p)[0],
                                             (sub.pattern.values == p).sum(), replace=True)
                                  for p in sub.pattern.unique()])
            boots.append(np.sum(hit[idx] * w[idx]) / np.sum(w[idx]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append(dict(system=name, n_judged_in_sample=int(judged.sum()),
                         weighted_coverage=cov, accuracy_vs_human=est, ci_low=lo, ci_high=hi))
    out = pd.DataFrame(rows)
    out.to_csv(op("out_06_human_anchored_accuracy.csv"), index=False)
    matched_analysis(d, adj, systems, JJ)
    print("Do chinh xac so voi phan quyet cua NGUOI (da ap trong so tang, uoc luong cho 1.985 luot):")
    print(out.round(4).to_string(index=False))
    print("\nSo sanh voi cac con so dong-thuan-voi-GPT-4o trong bai: SAVA 0.744 / judge 0.754 / regex 0.692")
    print("-> out_06_human_anchored_accuracy.csv")

def matched_analysis(d, adj, systems, JJ):
    """So sanh tren CUNG tap luot: luot SAVA cham va luot SAVA tu choi (trong so tang, bootstrap trong tang)."""
    from sklearn.metrics import cohen_kappa_score
    a, b = adj.dung_sai_A.astype(str).str.strip().str.lower(), adj.dung_sai_B.astype(str).str.strip().str.lower()
    agree = dict(raw_agreement=float((a == b).mean()), cohen_kappa=float(cohen_kappa_score(a, b)),
                 n_disagreements=int((a != b).sum()), n_unclear_final=int((adj["final"] == "chua_ro").sum()))
    rng = np.random.default_rng(221)
    def boot(mask, v, v2=None, B=2000):
        base = d[mask]; pats = base.pattern.values; out = []
        for _ in range(B):
            idx = np.concatenate([rng.choice(np.where(pats == p)[0], (pats == p).sum()) for p in np.unique(pats)])
            s_ = base.iloc[idx]; w = s_.w.values
            h1 = (v[s_.index].values == s_.human.values).astype(float)
            if v2 is None: out.append((h1 * w).sum() / w.sum())
            else:
                h2 = (v2[s_.index].values == s_.human.values).astype(float); out.append(((h1 - h2) * w).sum() / w.sum())
        return np.percentile(out, [2.5, 97.5])
    names = {"GPT-4o reference label": "GPT-4o reference label", "SAVA": "SAVA", "Numeric regex": "Numeric regex",
             "LLM judge (binary)": "LLM judge (binary)", "LLM judge (three-way)": "LLM judge (three-way)"}
    sava_j = systems["SAVA"].isin(JJ)
    rows = []
    for subset, mask in [("sava_judges", sava_j), ("sava_declines", ~sava_j)]:
        share = float(d.w[mask].sum() / d.w.sum())
        for name, v in systems.items():
            if subset == "sava_declines" and name == "SAVA": continue
            m = mask & v.isin(JJ); sub = d[m]
            acc = float((((v[sub.index] == sub.human).astype(float)) * sub.w).sum() / sub.w.sum())
            lo, hi = boot(m, v)
            rows.append(dict(subset=subset, weighted_share=share, system=name, n=int(m.sum()), accuracy_vs_human=acc, ci_low=lo, ci_high=hi))
        if subset == "sava_judges":
            jb = systems["LLM judge (binary)"]; both = mask & jb.isin(JJ)
            diff = float((((jb == d.human).astype(float) - (systems["SAVA"] == d.human).astype(float))[both] * d.w[both]).sum() / d.w[both].sum())
            lo, hi = boot(both, jb, systems["SAVA"])
            rows.append(dict(subset=subset, weighted_share=share, system="judge (binary) minus SAVA, paired", n=int(both.sum()), accuracy_vs_human=diff, ci_low=lo, ci_high=hi))
        else:
            h = d[mask]; maj = h.human.value_counts().idxmax()
            rows.append(dict(subset=subset, weighted_share=share, system="majority class (%s)" % maj, n=len(h),
                             accuracy_vs_human=float(((h.human == maj) * h.w).sum() / h.w.sum()), ci_low=np.nan, ci_high=np.nan))
    out = pd.DataFrame(rows); out.to_csv(op("out_06_human_matched.csv"), index=False)
    json.dump(agree, open(op("out_06_annotator_agreement.json"), "w"), indent=1)
    print("Hai nguoi xac minh:", {k: round(v, 3) if isinstance(v, float) else v for k, v in agree.items()})
    print("Tren cung tap luot (luot SAVA cham / luot SAVA tu choi):")
    print(out.round(4).to_string(index=False))
    print("-> out_06_human_matched.csv, out_06_annotator_agreement.json\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true", help="gop nhan cua A va B vao file thong nhat")
    do_merge() if ap.parse_args().merge else do_score()
