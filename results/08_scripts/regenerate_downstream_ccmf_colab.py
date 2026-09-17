"""Colab re-run of the Error Analyzer and Feedback Generator with CCMF mastery (SoICT 2026 paper).

Run on a Colab GPU (the earlier regeneration used an NVIDIA L4 with torch 2.11.0, transformers 5.17.0,
bitsandbytes 0.50.2, accelerate 1.15.0). Upload `inputs_ccmf.json` (from
ket_qua_bai_bao/06_feedback/regenerated_cases/ccmf_full_rerun/) to /content, add HF_TOKEN (Llama licence) and
OPENROUTER_API_KEY (GPT-4o reference judge) to Colab Secrets, then run R1-R6 in order.

Everything between the "verbatim" markers is copied unchanged from colab_full_framework_replication.ipynb
(cells 2, 10, 24, 32, 35), so prompts, decoding (greedy, max_new_tokens=320, batch 8), validators, rule-based
fallbacks, leakage and band checks, and the judge are identical to the paper's pipeline. Only the mastery values,
weakest KC, band and next KC in the inputs differ: they come from the integrated loop re-run with CCMF.
R6 prints the sha256 of the exported results; pass it with the saved .b64 file to 08_scripts/apply_ccmf_full_rerun.py.
"""

# ---------------- R1: environment and generator (identical to load_generator with PEDAGOGY_USE_KT_ADAPTER=False) ----------------
import base64, gzip, hashlib, json, os, re, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import pandas as pd
STUB = os.environ.get("CCMF_RERUN_STUB") == "1"   # local pipeline test only; never set on Colab
if not STUB:
    import torch, transformers, bitsandbytes, accelerate
    from google.colab import userdata
    from huggingface_hub import login
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    login(token=userdata.get("HF_TOKEN"))
    print("torch", torch.__version__, "| transformers", transformers.__version__, "| bitsandbytes", bitsandbytes.__version__,
          "| accelerate", accelerate.__version__, "| GPU", torch.cuda.get_device_name(0))

# ---------------- R2: verbatim notebook definitions ----------------
# >>> verbatim start
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

GEN_MAX_NEW_TOKENS = 320

GEN_BATCH_SIZE = 8

PEDAGOGY_BACKEND = "local"

PEDAGOGY_OPENROUTER_MODEL = "openai/gpt-4o-mini"

JUDGE_MODEL = "openai/gpt-4o"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

NUMBER = r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?"

PEDAGOGY_SYSTEM = "You are a mathematics tutor. Reply with a single JSON object only."

ERROR_CLASSES = ["conceptual", "procedural", "calculation", "careless", "other"]

SEVERITIES = ["low", "medium", "high"]

FEEDBACK_KEYS = ["feedback_text", "scaffolding_question", "mastery_adaptation", "pedagogical_strategy", "next_step_hint"]

BAND_KEYWORDS = {"low": ("worked", "re-teach", "reteach", "example", "demonstrat"),
                 "medium": ("scaffold", "guiding", "guided"),
                 "high": ("hint", "self-explanation", "self explanation")}


def to_fraction(text):
    try:
        return Fraction(str(text).replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def openrouter_key():
    try:
        from google.colab import userdata
        key = userdata.get("OPENROUTER_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("OPENROUTER_API_KEY")


def openrouter_post(payload, retries=4):
    key = openrouter_key()
    assert key, "Add OPENROUTER_API_KEY to Colab Secrets (key icon in the left sidebar) and enable notebook access."
    request = urllib.request.Request(OPENROUTER_URL, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                                              "X-Title": "SoICT2026 replication"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read())["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as error:
            if error.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(5 * 2 ** attempt)
                continue
            raise RuntimeError(f"OpenRouter HTTP {error.code}: {error.read()[:300]!r}") from None
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(5 * 2 ** attempt)
                continue
            raise


def generate_texts(prompts):
    if PEDAGOGY_BACKEND == "openrouter":
        def one(prompt):
            return openrouter_post({"model": PEDAGOGY_OPENROUTER_MODEL, "temperature": 0, "max_tokens": GEN_MAX_NEW_TOKENS,
                                    "messages": [{"role": "system", "content": PEDAGOGY_SYSTEM},
                                                 {"role": "user", "content": prompt}]})
        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(one, prompts))
        print(f"generated {len(outputs)}/{len(prompts)} via OpenRouter ({PEDAGOGY_OPENROUTER_MODEL})")
        return outputs
    assert PEDAGOGY_BACKEND == "local", PEDAGOGY_BACKEND
    return generate_texts_local(prompts)


def generate_texts_local(prompts):
    outputs = []
    for start in range(0, len(prompts), GEN_BATCH_SIZE):
        batch = prompts[start:start + GEN_BATCH_SIZE]
        chats = [GEN_TOKENIZER.apply_chat_template([{"role": "system", "content": PEDAGOGY_SYSTEM},
                                                    {"role": "user", "content": p}],
                                                   tokenize=False, add_generation_prompt=True) for p in batch]
        encoded = GEN_TOKENIZER(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(GEN_MODEL.device)
        with torch.inference_mode():
            generated = GEN_MODEL.generate(**encoded, max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=False,
                                           temperature=None, top_p=None, pad_token_id=GEN_TOKENIZER.pad_token_id)
        for row in generated[:, encoded["input_ids"].shape[1]:]:
            outputs.append(GEN_TOKENIZER.decode(row, skip_special_tokens=True))
        print(f"generated {len(outputs)}/{len(prompts)}")
    return outputs


def parse_json_object(text):
    match = re.search(r"\{.*\}", str(text), flags=re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def short_kc(kc, limit=160):
    kc = str(kc)
    return kc if len(kc) <= limit else kc[:limit - 3] + "..."


def validate_error_analysis(obj):
    if not obj:
        return None
    category = str(obj.get("error_category", "")).strip().lower()
    severity = str(obj.get("severity", "")).strip().lower()
    explanation, concepts, suggestions = obj.get("explanation"), obj.get("affected_concepts"), obj.get("instructional_suggestions")
    if category not in ERROR_CLASSES or severity not in SEVERITIES:
        return None
    if not isinstance(explanation, str) or not explanation.strip():
        return None
    if not isinstance(concepts, list) or not isinstance(suggestions, list) or not suggestions:
        return None
    return {"error_category": category, "explanation": explanation.strip(), "severity": severity,
            "affected_concepts": [str(c) for c in concepts], "instructional_suggestions": [str(s) for s in suggestions]}


def error_analyzer_prompt(item):
    mastery = ", ".join(f"{short_kc(k, 60)}: {v:.2f}" for k, v in item["mastery"].items()) or "not available"
    return (
        "Analyse the student's error in a mathematics tutoring dialogue.\n"
        f"Problem: {item.get('problem') or 'not provided'}\n"
        f"Reference answer: {item.get('reference') or 'not provided'}\n"
        f"Dialogue so far:\n{item['history']}\n\n"
        f"Student turn to analyse: {item['student_text']}\n"
        f"Extracted student expression (symbolic verifier): {item.get('extracted')}\n"
        f"Knowledge components involved: {'; '.join(short_kc(k) for k in item['kcs'])}\n"
        f"Current mastery estimates: {mastery}\n\n"
        "Category definitions: conceptual = misunderstanding of a concept or of the relation between quantities; "
        "procedural = a wrong, missing or misordered step of a method; calculation = an arithmetic slip inside an "
        "otherwise correct method; careless = a copying, sign or reading slip; other = none of these.\n"
        "Return JSON with exactly these keys: error_category (one of conceptual, procedural, calculation, careless, other), "
        "explanation (string), affected_concepts (list of KC names from the list above), severity (low|medium|high), "
        "instructional_suggestions (list of 2-3 strings)."
    )


def rule_based_error_analysis(item):
    # Port of kse2026_code/pedagogy.py ErrorAnalyzer.rule_based
    text = str(item["student_text"]).strip().lower()
    extracted, reference = item.get("extracted"), item.get("reference_parsed")
    mastery, kcs = item["mastery"], item["kcs"]
    category, explanation = "other", "The error could not be attributed to a specific mechanism from the text alone."
    try:
        a, b = float(extracted), float(reference)
        if abs(a + b) < 1e-9 and b != 0:
            category, explanation = "careless", f"The answer {a:g} has the right magnitude but the wrong sign."
        elif abs(a - b) <= 1:
            category, explanation = "careless", f"The answer {a:g} is off by {abs(a - b):g} from {b:g}."
        elif b != 0 and (abs(a / b - 2) < 1e-9 or abs(a / b - 0.5) < 1e-9):
            category, explanation = "procedural", f"The answer {a:g} is {b:g} scaled by two: a step was applied or omitted."
        elif b != 0 and abs(a * b - 1) < 1e-9:
            category, explanation = "conceptual", f"The answer {a:g} is the reciprocal of {b:g}: the relation was inverted."
        elif b != 0 and abs(a - b) / max(1.0, abs(b)) < 0.15:
            category, explanation = "calculation", f"The answer {a:g} is close to {b:g}: an arithmetic slip."
        else:
            category, explanation = "procedural", f"The answer {a:g} differs substantially from {b:g}."
    except (TypeError, ValueError):
        symbolic_ext = extracted is not None and re.search(r"[a-z]", str(extracted)) is not None
        symbolic_ref = reference is not None and re.search(r"[a-z]", str(reference)) is not None
        if symbolic_ext and symbolic_ref:
            category, explanation = "procedural", "A rule was applied to only part of the expression."
        elif symbolic_ext:
            category, explanation = "procedural", "The student stopped at an intermediate algebraic form."
        elif any(w in text for w in ("because", "since", "means", "should", "rule", "always")):
            category, explanation = "conceptual", "The explanation reveals a misconception about the underlying rule."
    weakest = min((mastery.get(k, 0.5) for k in kcs), default=0.5)
    if category in ("careless", "calculation"):
        severity = "low" if weakest >= 0.5 else "medium"
    else:
        severity = "high" if weakest < 0.4 else "medium"
    affected = [k for k in kcs if mastery.get(k, 0.5) < 0.7] or list(kcs)
    return {"error_category": category, "explanation": explanation, "severity": severity,
            "affected_concepts": affected, "instructional_suggestions": ["Ask the student to explain the step that produced this answer."]}


def run_error_analyzer(items):
    outputs = generate_texts([error_analyzer_prompt(item) for item in items]) if items else []
    records = []
    for item, text in zip(items, outputs):
        parsed = validate_error_analysis(parse_json_object(text))
        rule = rule_based_error_analysis(item)
        final = parsed or rule
        records.append({"set": item["set"], "dataset": item["dataset"], "dialogue": item["dialogue"],
                        "turn_id": item["turn_id"], "label": item["label"], "sava_verdict": item["sava_verdict"],
                        "kcs": json.dumps(item["kcs"]), "valid_json": parsed is not None,
                        "source": "llm" if parsed else "rule_fallback", "category": final["error_category"],
                        "severity": final["severity"], "explanation": final["explanation"],
                        "suggestions": json.dumps(final["instructional_suggestions"]),
                        "category_rule_based": rule["error_category"], "raw_output": str(text)[:4000]})
    return pd.DataFrame(records)


def mastery_band(value):
    return "low" if value < 0.4 else ("medium" if value < 0.7 else "high")


def feedback_prompt(case):
    mastery = ", ".join(f"{short_kc(k, 60)}: {v:.2f}" for k, v in case["mastery"].items()) or "not available"
    diagnosis = json.dumps(case["error_analysis"]) if case["error_analysis"] else "none (answer correct or undetermined)"
    guidance = {"low": "worked example and re-teaching of the concept",
                "medium": "scaffolded prompting with a guiding question",
                "high": "a minimal hint and self-explanation"}[case["band"]]
    return (
        "Generate adaptive tutoring feedback for the student's latest turn.\n"
        f"Problem: {case.get('problem') or 'not provided'}\nReference answer: {case['reference']}\n"
        f"Dialogue so far:\n{case['history']}\n\nStudent turn: {case['student_text']}\n"
        f"Verified correctness y (1 correct, 0 incorrect, None undetermined): {case['y']}\n"
        f"Mastery estimates: {mastery}\n"
        f"Weakest knowledge component: {short_kc(case['target_kc'])} (mastery band: {case['band']}; use {guidance})\n"
        f"Error analysis: {diagnosis}\nPlanned next knowledge component: {short_kc(case['next_kc'])}\n\n"
        "Do not reveal the final answer. Return JSON with exactly these keys: feedback_text, scaffolding_question, "
        "mastery_adaptation, pedagogical_strategy, next_step_hint."
    )


def validate_feedback(obj):
    if not obj or not all(isinstance(obj.get(k), str) and obj[k].strip() for k in FEEDBACK_KEYS):
        return None
    return {k: obj[k].strip() for k in FEEDBACK_KEYS}


def rule_based_feedback(case):
    # Port of kse2026_code/pedagogy.py FeedbackGenerator.rule_based
    target, band = short_kc(case["target_kc"], 80), case["band"]
    strategy = {"low": "worked example and re-teaching of the concept",
                "medium": "scaffolded prompting with a guiding question",
                "high": "minimal hint and self-explanation"}[band]
    if case["y"] == 1:
        text, question = f"Correct. You handled {target} well in this step.", "Can you explain why each step is valid?"
    elif case["y"] == 0:
        explanation = (case["error_analysis"] or {}).get("explanation", "")
        text, question = f"Not quite. {explanation} Look again at the step involving {target}.", "What is the very next step after your last line, and why?"
    else:
        text, question = "I see your reasoning so far. Let us make it concrete with a value.", "What number do you get at the end of that step?"
    return {"feedback_text": text, "scaffolding_question": question,
            "mastery_adaptation": f"Mastery of the weakest KC is in the {band} band.",
            "pedagogical_strategy": strategy, "next_step_hint": f"Next we will work on {short_kc(case['next_kc'], 80)}."}


def leaks_answer(feedback, reference, y):
    if y == 1 or reference is None:
        return float("nan")
    target = to_fraction(reference)
    text = " ".join(feedback[k] for k in ["feedback_text", "scaffolding_question", "next_step_hint"])
    return float(any(to_fraction(x) == target for x in re.findall(NUMBER, text)))


def band_matches(strategy, band):
    return float(any(word in strategy.lower() for word in BAND_KEYWORDS[band]))


def judge_request(case, feedback):
    prompt = (
        "Rate the tutor feedback below on three criteria, each an integer from 1 (poor) to 5 (excellent).\n"
        "correctness: the mathematics and the judgement of the student's answer are right.\n"
        "relevance: the feedback addresses this student's turn, error and weakest skill.\n"
        "clarity: the feedback is clear and appropriate for the learner.\n\n"
        f"Problem: {case.get('problem')}\nReference answer: {case['reference']}\n"
        f"Dialogue so far:\n{case['history'][-1500:]}\nStudent turn: {case['student_text']}\n"
        f"Feedback: {json.dumps(feedback)}\n\n"
        'Return JSON: {"correctness": int, "relevance": int, "clarity": int, "rationale": str}'
    )
    return {"model": JUDGE_MODEL, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": "You are an expert mathematics teacher evaluating tutoring feedback."},
                         {"role": "user", "content": prompt}]}


def call_judge(request):
    scores = parse_json_object(openrouter_post(request)) or {}
    return {k: scores.get(k) if isinstance(scores.get(k), int) and 1 <= scores.get(k) <= 5 else None
            for k in ["correctness", "relevance", "clarity"]}
# <<< verbatim end

if STUB:   # local pipeline test: canned model outputs, no GPU; never active on Colab
    def generate_texts_local(prompts):
        ea = {"error_category": "procedural", "explanation": "stub", "affected_concepts": [], "severity": "low",
              "instructional_suggestions": ["stub"]}
        fb = {k: "stub scaffolded guiding question" for k in FEEDBACK_KEYS}
        return [json.dumps(ea if p.startswith("Analyse the student") else fb) for p in prompts]

if not STUB:
    GEN_TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
    GEN_TOKENIZER.pad_token = GEN_TOKENIZER.eos_token
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                      bnb_4bit_compute_dtype=torch.float16)
    GEN_MODEL = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=quantization,
                                                     dtype=torch.float16, device_map={"": 0})
    GEN_MODEL.eval()

# ---------------- R3: inputs ----------------
INPUT_PATH = os.environ.get("CCMF_RERUN_INPUTS", "/content/inputs_ccmf.json")
EXPECTED_INPUT_SHA256 = "9e8ef4786f3a57d18a89769dd02c3a621b333f87d36c2139ee5959986249ca50"
raw_inputs = open(INPUT_PATH).read()
input_sha = hashlib.sha256(raw_inputs.encode()).hexdigest()
assert input_sha == EXPECTED_INPUT_SHA256, f"inputs_ccmf.json differs from the local file: {input_sha}"
INPUTS = json.loads(raw_inputs)
EA_ITEMS, CASES = INPUTS["error_analyzer_items"], INPUTS["feedback_cases"]
assert len(EA_ITEMS) == 201 and len(CASES) == 30
print("inputs verified:", len(EA_ITEMS), "Error Analyzer items,", len(CASES), "feedback cases")

# ---------------- R4: Error Analyzer (run_error_analyzer, notebook cell 33) ----------------
ea_df = run_error_analyzer(EA_ITEMS)
print(ea_df.groupby("set").agg(n=("valid_json", "size"), valid_json_rate=("valid_json", "mean")))

# ---------------- R5: Feedback Generator (notebook cell 36 with CCMF inputs) ----------------
analysed = {(r.dialogue, int(r.turn_id)): r for r in ea_df.itertuples()}
cases = []
for c in CASES:
    diagnosis = analysed.get((c["dialogue"], int(c["turn_id"]))) if c["verdict"] == "incorrect" else None
    assert c["verdict"] != "incorrect" or diagnosis is not None, (c["dialogue"], c["turn_id"])
    cases.append({**c, "error_analysis": {"error_category": diagnosis.category, "explanation": diagnosis.explanation}
                  if diagnosis is not None else None})
outputs = generate_texts([feedback_prompt(c) for c in cases])
records, judge_requests = [], []
for case, text in zip(cases, outputs):
    parsed = validate_feedback(parse_json_object(text))
    feedback = parsed or rule_based_feedback(case)
    records.append({"dialogue": case["dialogue"], "turn_id": case["turn_id"], "verdict": case["verdict"],
                    "band": case["band"], "valid_json": parsed is not None,
                    "source": "llm" if parsed else "rule_fallback",
                    "answer_leak": leaks_answer(feedback, case["reference"], case["y"]),
                    "strategy_band_match": band_matches(feedback["pedagogical_strategy"], case["band"]),
                    "feedback_words": len(feedback["feedback_text"].split()), **feedback, "raw_output": str(text)[:4000]})
    judge_requests.append(judge_request(case, feedback))
feedback_df = pd.DataFrame(records)
if not STUB:
    assert openrouter_key(), "add OPENROUTER_API_KEY to Colab Secrets"
    scores = [call_judge(request) for request in judge_requests]
    feedback_df = pd.concat([feedback_df, pd.DataFrame(scores)], axis=1)
print(feedback_df[["dialogue", "turn_id", "verdict", "band", "valid_json", "source", "strategy_band_match"]].to_string(index=False))

# ---------------- R6: export ----------------
results = {"inputs_sha256": input_sha, "error_analysis": json.loads(ea_df.to_json(orient="records")),
           "feedback": json.loads(feedback_df.to_json(orient="records"))}
payload = json.dumps(results, ensure_ascii=True, sort_keys=True).encode()
RESULT_SHA256 = hashlib.sha256(payload).hexdigest()
OUT_B64 = os.environ.get("CCMF_RERUN_OUT_B64", "/content/results_ccmf.b64")
with open(OUT_B64, "w") as handle:
    handle.write(base64.b64encode(gzip.compress(payload)).decode())
print("results sha256:", RESULT_SHA256, "| saved to", OUT_B64)
if not STUB:
    from google.colab import files
    files.download(OUT_B64)
