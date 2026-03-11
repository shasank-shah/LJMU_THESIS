import os
import re
import json
import ast
import random
import argparse
import subprocess
import base64
import mimetypes
import requests
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime

# =========================
# 0) CONFIG / DATA MODELS
# =========================

@dataclass
class Sample:
    image_path: str
    code_path: str
    language: str
    id: str


@dataclass
class Metrics:
    exact_match: float
    token_accuracy: float
    codebleu_like: float
    compilation_success: float
    ast_success: float


# =========================
# 1) OLLAMA RESOLUTION (CLI + API)
# =========================

def resolve_ollama_bin(user_path: str | None = None) -> str:
    if user_path and os.path.exists(user_path):
        return user_path

    env_path = os.environ.get("OLLAMA_BIN")
    if env_path and os.path.exists(env_path):
        return env_path

    try:
        import shutil
        p = shutil.which("ollama")
        if p and os.path.exists(p):
            return p
    except Exception:
        pass

    common = [
        "/usr/local/bin/ollama",
        "/opt/homebrew/bin/ollama",
        "/Applications/Ollama.app/Contents/Resources/ollama",
    ]
    for p in common:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        "Could not find 'ollama' binary. Provide --ollama_bin or set env var OLLAMA_BIN."
    )


def run_ollama_cli(
    ollama_bin: str,
    model: str,
    prompt: str,
    timeout_s: int = 900,
) -> str:
    """
    Text-only inference via CLI:
      ollama run <model> "<prompt>"
    """
    cmd = [ollama_bin, "run", model, prompt]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f"Ollama CLI failed.\nCMD: {' '.join(cmd)}\nSTDERR:\n{proc.stderr}")
    return proc.stdout


# OpenAI-compatible Ollama API base
OLLAMA_API_BASE = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_API_BASE}/v1/chat/completions"


def _guess_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "image/png"


def image_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b = f.read()
    b64 = base64.b64encode(b).decode("utf-8")
    mime = _guess_mime(image_path)
    return f"data:{mime};base64,{b64}"


def run_ollama_vlm_api(
    model: str,
    prompt: str,
    image_path: str,
    timeout_s: int = 900,
    max_tokens: int = 650,
    temperature: float = 0.0,
) -> str:
    """
    Vision inference via OpenAI-compatible API:
      POST /v1/chat/completions with messages content including image_url.
    """
    img_url = image_to_data_url(image_path)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# =========================
# 2) DATASET: COLLECT + LOAD
# =========================

def infer_language_from_ext(code_path: str) -> str:

    name = os.path.basename(code_path).lower()

    # detect language from prefix first
    if name.startswith("python_"):
        return "python"
    if name.startswith("java_"):
        return "java"
    if name.startswith("javascript_"):
        return "javascript"
    if name.startswith("cpp_"):
        return "cpp"
    if name.startswith("html_") or name.startswith("html_css_"):
        return "html"

    # fallback to extension
    ext = os.path.splitext(code_path)[1].lower()

    return {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".c": "c",
        ".cs": "csharp",
        ".go": "go",
        ".rs": "rust",
        ".php": "php",
        ".rb": "ruby",
        ".kt": "kotlin",
        ".swift": "swift",
        ".sql": "sql",
        ".html": "html",
        ".css": "css",
    }.get(ext, "unknown")

def collect_dataset(dataset_dir: str) -> List[Sample]:

    img_dir = os.path.join(dataset_dir, "images")
    code_dir = os.path.join(dataset_dir, "code")

    if not os.path.isdir(img_dir) or not os.path.isdir(code_dir):
        raise FileNotFoundError("Expected dataset_dir/images and dataset_dir/code")

    samples = []

    images = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]

    code_files = os.listdir(code_dir)

    # build map: 0001 -> 0001.cpp
    code_map = {}
    for cf in code_files:
        stem = os.path.splitext(cf)[0]
        code_map[stem] = cf

    for img_name in images:

        stem = os.path.splitext(img_name)[0]

        if stem not in code_map:
            print(f"WARNING: No ground truth for image {img_name}")
            continue

        img_path = os.path.join(img_dir, img_name)
        code_file = code_map[stem]
        code_path = os.path.join(code_dir, code_file)

        language = infer_language_from_ext(code_path)

        samples.append(
            Sample(
                image_path=img_path,
                code_path=code_path,
                language=language,
                id=stem
            )
        )

    print(f"Loaded {len(samples)} samples")

    return samples

def split_dataset(samples: List[Sample], train=0.75, val=0.15, seed=42):
    rnd = random.Random(seed)
    rnd.shuffle(samples)
    n = len(samples)
    n_train = int(n * train)
    n_val = int(n * val)
    return samples[:n_train], samples[n_train:n_train + n_val], samples[n_train + n_val:]


# =========================
# 3) CLEANING / VALIDATION
# =========================

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+-]*\s*|\s*```$", re.MULTILINE)

def normalise_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")

def strip_markdown_fences(s: str) -> str:
    return _FENCE_RE.sub("", s).strip()

def remove_numeric_spam(s: str) -> str:
    return re.sub(r"(,\s*\d+){25,}", "", s)

def remove_duplicate_adjacent_lines(s: str) -> str:
    lines = s.splitlines()
    out = []
    for ln in lines:
        if out and ln == out[-1]:
            continue
        out.append(ln)
    return "\n".join(out)

'''
def normalise_model_output(s: str) -> str:
    s = normalise_newlines(s)
    s = strip_markdown_fences(s)
    s = remove_numeric_spam(s)
    s = remove_duplicate_adjacent_lines(s)
    return s.strip()
'''

def normalise_model_output(s: str) -> str:
    s = normalise_newlines(s)

    # remove markdown fences
    s = strip_markdown_fences(s)

    # remove separators
    s = re.sub(r"^\s*-{3,}\s*$", "", s, flags=re.MULTILINE)

    # remove leading/trailing separators
    s = re.sub(r"^-+\n", "", s)
    s = re.sub(r"\n-+$", "", s)

    # remove numeric spam
    s = remove_numeric_spam(s)

    # remove duplicate lines
    s = remove_duplicate_adjacent_lines(s)

    return s.strip()

def python_ast_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False

def python_compiles(code: str) -> bool:
    try:
        compile(code, "<pred>", "exec")
        return True
    except Exception:
        return False

def brackets_balanced(s: str) -> bool:
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    opens = set(pairs.values())
    for ch in s:
        if ch in opens:
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return not stack

def is_valid(pred: str, lang: str) -> bool:
    lang = (lang or "unknown").lower()
    if lang == "python":
        return python_ast_ok(pred)
    return brackets_balanced(pred)


# =========================
# 4) VLM + CODER + RETRY
# =========================

def vlm_extract_code_api(vision_model: str, image_path: str, language_hint: str, timeout_s: int) -> str:
    prompt = f"""
You are performing VISUAL TRANSCRIPTION of source code from an image.

STRICT RULES:
- Output ONLY the code visible in the image
- Preserve indentation exactly
- Do NOT add type hints or comments
- Do NOT 'improve' or rewrite code
- Do NOT complete missing code
- Output raw code only

Language hint: {language_hint}

Stop when the visible code ends.
(Keep output concise; do not repeat lines.)
"""
    out = run_ollama_vlm_api(
        model=vision_model,
        prompt=prompt,
        image_path=image_path,
        timeout_s=timeout_s,
        max_tokens=650,
        temperature=0.0,
    )
    return normalise_model_output(out)


def coder_reconstruct_cli(ollama_bin: str, coder_model: str, raw_code: str, language_hint: str, timeout_s: int) -> str:
    prompt = f"""
You are a CODE RECONSTRUCTION module.

The text below was extracted from an image and may contain transcription errors.
Fix ONLY obvious transcription/syntax/indentation issues.

STRICT RULES:
- Do NOT change program logic
- Do NOT rename variables/functions/classes
- Do NOT add new functionality
- Do NOT add type hints or comments
- Do NOT delete lines unless they are clear garbage
- Preserve the code as close as possible to the input

Language hint: {language_hint}

INPUT CODE:
--------------------
{raw_code}
--------------------

Return corrected code only.
"""
    out = run_ollama_cli(ollama_bin, coder_model, prompt, timeout_s=timeout_s)
    return normalise_model_output(out)


def extract_with_retry_loop(
    ollama_bin: str,
    vision_model: str,
    coder_model: str,
    sample: Sample,
    timeout_s: int,
    max_retries: int = 2,
) -> Tuple[str, Dict]:
    meta = {"tries": [], "final_valid": False}

    raw = vlm_extract_code_api(vision_model, sample.image_path, sample.language, timeout_s)
    repaired = coder_reconstruct_cli(ollama_bin, coder_model, raw, sample.language, timeout_s)

    v = is_valid(repaired, sample.language)
    meta["tries"].append({"stage": "vlm_api_then_coder_cli", "valid": v})
    if v:
        meta["final_valid"] = True
        return repaired, meta

    cur = repaired
    for i in range(1, max_retries + 1):
        cur = coder_reconstruct_cli(ollama_bin, coder_model, cur, sample.language, timeout_s)
        v = is_valid(cur, sample.language)
        meta["tries"].append({"stage": f"coder_retry_{i}", "valid": v})
        if v:
            meta["final_valid"] = True
            return cur, meta

    return cur, meta


# =========================
# 5) METRICS
# =========================

def read_ground_truth(code_path: str) -> str:
    with open(code_path, "r", encoding="utf-8", errors="ignore") as f:
        return normalise_newlines(f.read()).rstrip()

def normalise_for_compare(s: str) -> str:
    s = normalise_newlines(s).rstrip()
    s = re.sub(r"[ \t]+", " ", s)
    return s

def exact_match(pred: str, gold: str) -> int:
    return int(normalise_for_compare(pred) == normalise_for_compare(gold))

def tokenise_code(s: str) -> List[str]:
    return re.findall(r"[A-Za-z_]\w*|\d+|==|!=|<=|>=|->|[^\s]", s)

def token_accuracy(pred: str, gold: str) -> float:
    p = tokenise_code(pred)
    g = tokenise_code(gold)
    if not g:
        return 0.0
    m = min(len(p), len(g))
    correct = sum(1 for i in range(m) if p[i] == g[i])
    return correct / max(len(g), 1)

def ngram_precision(pred_tokens: List[str], gold_tokens: List[str], n: int) -> float:
    def ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    p_ng = ngrams(pred_tokens, n)
    g_ng = ngrams(gold_tokens, n)
    if not p_ng:
        return 0.0
    g_counts = {}
    for ng in g_ng:
        g_counts[ng] = g_counts.get(ng, 0) + 1
    match = 0
    p_counts = {}
    for ng in p_ng:
        p_counts[ng] = p_counts.get(ng, 0) + 1
    for ng, c in p_counts.items():
        match += min(c, g_counts.get(ng, 0))
    return match / max(len(p_ng), 1)

def codebleu_like(pred: str, gold: str) -> float:
    p = tokenise_code(pred)
    g = tokenise_code(gold)
    if not g:
        return 0.0
    precisions = [ngram_precision(p, g, n) for n in [1, 2, 3, 4]]
    return sum(precisions) / 4.0

def compilation_success(pred: str, lang: str) -> int:
    lang = (lang or "unknown").lower()
    if lang == "python":
        return int(python_compiles(pred))
    return int(brackets_balanced(pred))

def ast_success(pred: str, lang: str) -> int:
    lang = (lang or "unknown").lower()
    if lang == "python":
        return int(python_ast_ok(pred))
    return int(brackets_balanced(pred))


# =========================
# 6) SAVE OUTPUTS
# =========================

def save_results(out_dir: str, exp_name: str, split_name: str, metrics: Metrics, results: List[Dict]):
    os.makedirs(out_dir, exist_ok=True)
    mdir = os.path.join(out_dir, "metrics")
    rdir = os.path.join(out_dir, "predictions")
    os.makedirs(mdir, exist_ok=True)
    os.makedirs(rdir, exist_ok=True)

    with open(os.path.join(mdir, f"{exp_name}_{split_name}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics.__dict__, f, indent=2)

    with open(os.path.join(rdir, f"{exp_name}_{split_name}_predictions.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_generated_code_files(out_dir: str, split_name: str, results: List[Dict]):
    gen_root = os.path.join(out_dir, "generated_code", split_name)
    os.makedirs(gen_root, exist_ok=True)

    ext_map = {
        "python": ".py",
        "javascript": ".js",
        "typescript": ".ts",
        "java": ".java",
        "cpp": ".cpp",
        "c": ".c",
        "csharp": ".cs",
        "go": ".go",
        "rust": ".rs",
        "php": ".php",
        "ruby": ".rb",
        "kotlin": ".kt",
        "swift": ".swift",
        "sql": ".sql",
        "html": ".html",
        "css": ".css",
    }

    for r in results:
        pred_code = r["prediction"]
        sample_id = r["id"]
        language = (r.get("language") or "unknown").lower()
        ext = ext_map.get(language, ".txt")
        out_path = os.path.join(gen_root, sample_id + ext)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(pred_code)


# =========================
# 7) RUN PIPELINE
# =========================

def run_pipeline_on_samples(
    samples: List[Sample],
    exp_name: str,
    ollama_bin: str,
    vision_model: str,
    coder_model: str,
    timeout_s: int,
    max_retries: int,
) -> Tuple[Metrics, List[Dict]]:

    preds = []
    results = []

    for s in samples:

        # -----------------------------
        # Start timestamp
        # -----------------------------
        start_time = datetime.now()

        gold = read_ground_truth(s.code_path)

        if exp_name != "E4":
            raise ValueError("This script implements E4 only.")

        pred, meta = extract_with_retry_loop(
            ollama_bin=ollama_bin,
            vision_model=vision_model,
            coder_model=coder_model,
            sample=s,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )

        # -----------------------------
        # End timestamp
        # -----------------------------
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        em = exact_match(pred, gold)
        ta = token_accuracy(pred, gold)
        cb = codebleu_like(pred, gold)
        comp = compilation_success(pred, s.language)
        ast_ok = ast_success(pred, s.language)

        results.append({
            "id": s.id,
            "language": s.language,
            "image_path": s.image_path,
            "code_path": s.code_path,

            # timestamps
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration,

            # metrics
            "exact_match": em,
            "token_accuracy": ta,
            "codebleu_like": cb,
            "compilation_success": comp,
            "ast_success": ast_ok,

            "prediction": pred,
            "debug": meta,
        })

        preds.append((em, ta, cb, comp, ast_ok))

        print(f"Processed {s.id} | Duration: {duration:.2f}s")

    if not preds:
        return Metrics(0.0, 0.0, 0.0, 0.0, 0.0), results

    n = len(preds)

    return Metrics(
        exact_match=sum(x[0] for x in preds) / n,
        token_accuracy=sum(x[1] for x in preds) / n,
        codebleu_like=sum(x[2] for x in preds) / n,
        compilation_success=sum(x[3] for x in preds) / n,
        ast_success=sum(x[4] for x in preds) / n,
    ), results


# =========================
# 8) MAIN
# =========================

def main():
    parser = argparse.ArgumentParser(description="Thesis Pipeline: Qwen-VL(API) -> Qwen-Coder(CLI) -> Validation -> Metrics")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--exp", default="E4", choices=["E4"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate_code", action="store_true")
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--timeout_s", type=int, default=900)
    parser.add_argument("--ollama_bin", default=None)
    parser.add_argument("--vision_model", default="qwen2.5vl:latest")
    parser.add_argument("--coder_model", default="qwen2.5-coder:7b")
    args = parser.parse_args()

    ollama_bin = resolve_ollama_bin(args.ollama_bin)

    samples = collect_dataset(args.dataset_dir)
    if not samples:
        raise RuntimeError("No samples found. Check dataset structure.")

    train_set, val_set, test_set = split_dataset(samples, seed=args.seed)

    for split_name, split_samples in [("train", train_set), ("val", val_set), ("test", test_set)]:
        metrics, results = run_pipeline_on_samples(
            samples=split_samples,
            exp_name=args.exp,
            ollama_bin=ollama_bin,
            vision_model=args.vision_model,
            coder_model=args.coder_model,
            timeout_s=args.timeout_s,
            max_retries=args.max_retries,
        )

        print(f"\n[{args.exp}] {split_name.upper()} METRICS")
        print(f"  Exact Match:          {metrics.exact_match:.4f}")
        print(f"  Token Accuracy:       {metrics.token_accuracy:.4f}")
        print(f"  CodeBLEU-like:        {metrics.codebleu_like:.4f}")
        print(f"  Compilation Success:  {metrics.compilation_success:.4f}")
        print(f"  AST Success:          {metrics.ast_success:.4f}")

        save_results(args.out_dir, args.exp, split_name, metrics, results)

        if args.generate_code:
            save_generated_code_files(args.out_dir, split_name, results)

    print(f"\nSaved outputs to: {args.out_dir}")
    print(f"Ollama bin: {ollama_bin}")
    print(f"Vision model (API): {args.vision_model}")
    print(f"Coder model (CLI): {args.coder_model}")


if __name__ == "__main__":
    main()

