"""LLM-judge baseline for SAVA, as run on Colab (NVIDIA L4; torch 2.11.0, transformers 5.17.0, bitsandbytes 0.50.2,
accelerate 1.15.0). GEN_MODEL / GEN_TOKENIZER are Llama-3.1-8B-Instruct loaded in 4-bit NF4 exactly as the pedagogical
agents (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
bnb_4bit_compute_dtype=torch.float16), tokenizer padding_side="left", pad_token = eos_token).

J1 rebuilds the 1,985 MathDial test turns from the released LLMKT data at commit c61f335 and checks them against the
paper's export (sha256 8bc0df3e...). J2 runs the judge in two prompt variants (three-way, binary) in batches of 8 with
greedy decoding and max_new_tokens=5. J3 exports the verdicts; the export hash is
2d40dd51017720f28a0082b6fd2c3d919ede98a2d1994fb761b1b74dc6c5b2e8. Analysis: analyze_llm_judge_baseline.py.
"""

# ---------------- J1 ----------------
import io, html, json, re, hashlib, urllib.request
import pandas as pd
from ast import literal_eval

URL = ("https://raw.githubusercontent.com/umass-ml4ed/dialogue-kt/c61f335f89005161b6ef439872cc1735bee26745/"
       "data/annotated/mathdial_test_atc.csv")
test_df = pd.read_csv(io.StringIO(urllib.request.urlopen(URL, timeout=300).read().decode()),
                      converters={c: literal_eval for c in ["dialogue", "meta_data", "annotation"]})
test_df = test_df[test_df.apply(lambda r: r["meta_data"]["self_typical_confusion"] >= 1
                                and r["meta_data"]["self_typical_interactions"] >= 1, axis=1)]
NUMBER = r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?"


def final_number(text):
    lines = [line.strip() for line in html.unescape(str(text)).splitlines() if line.strip()]
    if not lines:
        return None
    matches = re.findall(NUMBER, lines[-1])
    return matches[-1].replace(",", "") if matches else None


def apply_annotations(sample, apply_na=True):   # dialogue_kt/kt_data_loading.py at c61f335
    dialogue = sample["dialogue"]
    anno = sample["annotation"]
    if "error" in anno:
        return None
    if dialogue[0]["turn"] == 0:
        anno["turn 0"] = {"correct": None, "kcs": []}
    for dia_turn in dialogue:
        anno_turn = anno[f"turn {dia_turn['turn']}"]
        corr, kcs = anno_turn["correct"], anno_turn["kcs"]
        if apply_na:
            corr = None if not kcs else corr
            kcs = [] if corr is None else kcs
        dia_turn["correct"] = dia_turn["og_correct"] = corr
        dia_turn["kcs"] = kcs
    if dialogue[-1]["kcs"]:
        if "expected_result" in sample["meta_data"]:
            dialogue[-1]["correct"] = sample["meta_data"]["expected_result"] == "Answer Accepted"
        elif "self_correctness" in sample["meta_data"]:
            if dialogue[-1]["correct"] is not None:
                if sample["meta_data"]["self_correctness"] == "Yes":
                    dialogue[-1]["correct"] = True
                elif sample["meta_data"]["self_correctness"] == "Yes, but I had to reveal the answer":
                    dialogue[-1]["correct"] = None
                elif sample["meta_data"]["self_correctness"] == "No":
                    dialogue[-1]["correct"] = False
    return dialogue


TURNS = []
for idx, sample in test_df.iterrows():
    dialogue = apply_annotations(sample)
    if not dialogue:
        continue
    first, history = True, []
    for turn in dialogue:
        teacher = str(turn.get("teacher") or "").strip()
        if turn["correct"] is not None:
            if first:
                first = False
            else:
                TURNS.append({"dialogue": str(idx), "turn_id": int(turn["turn"]), "label": int(turn["correct"]),
                              "student_text": str(turn["student"]),
                              "reference": final_number(sample["meta_data"]["correct_solution"]),
                              "history": "\n".join(history + ([f"Tutor: {teacher}"] if teacher else []))[-2500:],
                              "problem": sample["meta_data"].get("question")})
        if teacher:
            history.append(f"Tutor: {teacher}")
        history.append(f"Student: {turn['student']}")
TURN_KEY = [[t["dialogue"], t["turn_id"], t["student_text"], t["reference"], t["label"]] for t in TURNS]
assert hashlib.sha256(json.dumps(TURN_KEY).encode()).hexdigest() == \
    "8bc0df3ece2c38a8dbf75eb63e13b1da9e8885e122f709326448f530fb073c66"

# ---------------- J2 ----------------
import time, torch

JUDGE_SYSTEM = "You are a mathematics teacher grading a student's turn in a tutoring dialogue. Reply with one word only."


def judge_prompt(t, three_way):
    options = ("Answer with exactly one word: correct, incorrect, or undetermined (use undetermined only if the turn "
               "contains no mathematical claim that can be checked)." if three_way
               else "Answer with exactly one word: correct or incorrect.")
    return (f"Problem: {t['problem']}\nReference final answer: {t['reference']}\nDialogue so far:\n{t['history']}\n\n"
            f"Student's latest turn: {t['student_text']}\n\nIs the mathematics in the student's latest turn correct? {options}")


def parse_verdict(text):
    match = re.search(r"[a-z]+", str(text).lower())
    return {"correct": "correct", "incorrect": "incorrect", "undetermined": "undetermined"}.get(
        match.group(0) if match else "", "unparsed")


def generate_judge(prompts):
    chats = [GEN_TOKENIZER.apply_chat_template([{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": p}],
                                               tokenize=False, add_generation_prompt=True) for p in prompts]
    encoded = GEN_TOKENIZER(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(GEN_MODEL.device)
    with torch.inference_mode():
        generated = GEN_MODEL.generate(**encoded, max_new_tokens=5, do_sample=False, temperature=None, top_p=None,
                                       pad_token_id=GEN_TOKENIZER.pad_token_id)
    return [GEN_TOKENIZER.decode(row, skip_special_tokens=True) for row in generated[:, encoded["input_ids"].shape[1]:]]


JUDGE_RAW = globals().get("JUDGE_RAW") or {"three_way": [], "binary": []}
started = time.time()
for variant, three_way in [("three_way", True), ("binary", False)]:
    while len(JUDGE_RAW[variant]) < len(TURNS) and time.time() - started < 600:   # resumable in 10-minute runs
        i = len(JUDGE_RAW[variant])
        JUDGE_RAW[variant].extend(generate_judge([judge_prompt(t, three_way) for t in TURNS[i:i + 8]]))

# ---------------- J3 ----------------
import gzip, base64

PARSED = {v: [parse_verdict(x) for x in JUDGE_RAW[v]] for v in JUDGE_RAW}
export = {"turn_key_sha256": hashlib.sha256(json.dumps(TURN_KEY).encode()).hexdigest(), "system_prompt": JUDGE_SYSTEM,
          "three_way": PARSED["three_way"], "binary": PARSED["binary"],
          "raw_three_way": JUDGE_RAW["three_way"], "raw_binary": JUDGE_RAW["binary"]}
canonical = json.dumps(export, ensure_ascii=True, sort_keys=True)
print("sha256 of export:", hashlib.sha256(canonical.encode()).hexdigest())
print(base64.b64encode(gzip.compress(canonical.encode(), mtime=0)).decode())
