"""09 — Do thi tien quyet cho 103 KC cua MathDial tu Achieve the Core Coherence Map.

KC cua MathDial la nguyen van mo ta chuan Common Core
theo Achieve the Core (LLMKT tag_src="atc"), nen khop duoc chinh xac voi ma chuan va lien ket
"progress from" do chuyen gia soan.

Nguon: allenai/achieve-the-core (Hugging Face, ODC-BY 1.0), file standards.jsonl, chep tai
out/external/achieve-the-core_standards.jsonl. Trich dan: arXiv 2408.04226 + Achieve the Core.

Cach dung canh:
  1. Khop KC -> chuan: trung khop sau chuan hoa; neu khong, KC la tien to cua mo ta ATC
     (ATC them chu thich "Grade N expectations ... limited to ...").
  2. Chuan con (sub-standard) gop vao KC cha.
  3. Di nguoc "progress from"; gap KC khac thi them canh va dung; gap chuan ngoai tap 103 KC
     thi di xuyen qua (co lap do thi qua node trung gian).
  4. Kiem phi chu trinh va khong co canh tu lop cao ve lop thap; rut gon bac cau.

Dau ra (out/): prereq_graph_mathdial_atc.json (nap vao cpu/08), prereq_edges_atc.csv,
kc_ccss_map.csv.
"""
import os, re, json, unicodedata, collections
import pandas as pd

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

ROOT = _find_root()
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(KIT, "out")
STD = os.path.join(OUT, "external", "achieve-the-core_standards.jsonl")

KCS = sorted({k for split in ["test", "val"]
              for r in json.load(open(os.path.join(ROOT, "raw_runs", "mathdial", "qlora", "seed221", split, "raw_logit_gaps.json")))
              for k in r["kcs"]})
S = {s["id"]: s for s in map(json.loads, open(STD))}
EMPTY = {"progress from": [], "progress to": [], "related": []}
conn = lambda sid: S[sid]["connections"] if isinstance(S.get(sid, {}).get("connections"), dict) else EMPTY
norm = lambda t: re.sub(r"[^a-z0-9]+", " ", unicodedata.normalize("NFKC", t).lower()).strip()
cands = [s for s in S.values() if s["level"] in ("Standard", "Sub-standard")]

# 1) KC -> ma chuan
kc2id, how = {}, {}
for k in KCS:
    nk = norm(k)
    ids = [s["id"] for s in cands if norm(s["description"]) == nk]
    kind = "exact"
    if not ids:
        ids, kind = [s["id"] for s in cands if norm(s["description"]).startswith(nk)], "prefix"
    assert len(ids) == 1, ("KC khong khop duy nhat", k[:80], ids)
    kc2id[k], how[k] = ids[0], kind
id2kc = {v: k for k, v in kc2id.items()}
assert len(id2kc) == len(KCS), "hai KC khop cung mot chuan"

def owner(sid):
    while sid:
        if sid in id2kc: return id2kc[sid]
        sid = S.get(sid, {}).get("parent")
    return None

def owned(kc):
    out, st = [], [kc2id[kc]]
    while st:
        n = st.pop(); out.append(n); st.extend(S.get(n, {}).get("children") or [])
    return out

def grade(k):
    sid = kc2id[k]
    return 0 if sid.startswith("K.") else int(sid.split(".")[0]) if sid[0].isdigit() else 9   # 9 = trung hoc

def domain(k):
    sid = kc2id[k]
    return sid.split(".")[1] if sid[0] in "K0123456789" else sid.split(".")[0]

# 2-3) canh tien quyet, co lap qua chuan ngoai tap KC
edges = set()
for b in KCS:
    seen, st = set(), [p for n in owned(b) for p in conn(n)["progress from"]]
    while st:
        n = st.pop()
        if n in seen or n not in S: continue
        seen.add(n); o = owner(n)
        if o == b: continue
        if o is not None: edges.add((o, b)); continue
        st.extend(conn(n)["progress from"])

succ = collections.defaultdict(set)
for a, b in edges: succ[a].add(b)

def reach(a, skip=None):
    seen, st = set(), [n for n in succ[a] if (a, n) != skip]
    while st:
        n = st.pop()
        if n in seen: continue
        seen.add(n); st.extend(succ[n])
    return seen

# 4) tu kiem + rut gon bac cau
assert not [e for e in edges if e[0] in reach(e[1])], "do thi ATC co chu trinh"
red = {(a, b) for a, b in edges if b not in reach(a, (a, b))}
assert not [e for e in red if grade(e[0]) > grade(e[1])], "co canh tu lop cao ve lop thap"

graph = {b: sorted(a for a, bb in red if bb == b) for b in sorted({b for _, b in red})}
json.dump({"prerequisites": graph, "n_kcs": len(KCS), "n_edges": len(red),
           "method": "Achieve the Core Coherence Map 'progress from' (allenai/achieve-the-core, ODC-BY), "
                     "contracted through standards outside the 103 KCs, sub-standards merged into parent KC, "
                     "transitive reduction"},
          open(os.path.join(OUT, "prereq_graph_mathdial_atc.json"), "w"), indent=1, ensure_ascii=False)
pd.DataFrame([dict(kc=k, ccss_id=kc2id[k], grade=grade(k), domain=domain(k), match=how[k]) for k in KCS]) \
  .to_csv(os.path.join(OUT, "kc_ccss_map.csv"), index=False)

pd.DataFrame([dict(prereq=a, prereq_id=kc2id[a], kc=b, kc_id=kc2id[b], grade_gap=grade(b) - grade(a))
              for a, b in sorted(red, key=lambda e: (kc2id[e[1]], kc2id[e[0]]))]) \
  .to_csv(os.path.join(OUT, "prereq_edges_atc.csv"), index=False)

print("Khop KC:", len(kc2id), dict(collections.Counter(how.values())))
print("Lop:", dict(sorted(collections.Counter(map(grade, KCS)).items())))
print("Canh ATC (co lap):", len(edges), "-> sau rut gon bac cau:", len(red),
      "|", len(graph), "KC co tien quyet; 0 chu trinh; 0 canh nguoc lop")
print("-> out/prereq_graph_mathdial_atc.json, out/prereq_edges_atc.csv, out/kc_ccss_map.csv")
