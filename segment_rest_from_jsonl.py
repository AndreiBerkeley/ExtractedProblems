#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
segment_rest_from_jsonl.py
--------------------------
Read Mathpix-derived all_contests.jsonl (one line per PDF),
and re-segment ONLY the specified contests/years into problem–solution pairs
with GPT-4o using specialized prompts and exact expected counts.

Target set:
- TSTST 2018, 2021, 2023      -> exactly 9 problems each (numbers 1..9)
- TST   2016, 2023, 2025       -> exactly 6 problems each (numbers 1..6)
- EGMO  2020                   -> exactly 6 problems (1..6)
- IMOSL 2022 (33 problems), 2024 (31 problems) -> labels A/C/G/N + numbers, e.g., A1, C3, ...

Output:
- One JSONL line per {contest, year, problem_number} in rest_contests.jsonl

Usage:
  export OPENAI_API_KEY=...
  python segment_rest_from_jsonl.py all_contests.jsonl -o rest_contests.jsonl --workers 10 --retries 2

Dependencies:
  pip install regex
"""

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import regex

# ------------------------
# Target set + expectations (strict)
# ------------------------
STRICT_EXPECTATIONS: Dict[Tuple[str, str], int] = {
    ("EGMO",  "2020"): 6,
    ("TST",   "2016"): 6,
    ("TST",   "2023"): 6,
    ("TST",   "2025"): 6,
    ("TSTST", "2018"): 9,
    ("TSTST", "2021"): 9,
    ("TSTST", "2023"): 9,
    ("IMOSL", "2022"): 33,
    ("IMOSL", "2024"): 31,
}

TARGET_KEYS = set(STRICT_EXPECTATIONS.keys())

# ------------------------
# Data model
# ------------------------
@dataclass
class Pair:
    id: str
    contest: str
    year: str
    problem_number: str  # allow "A3"
    problem: str
    solution: str
    source_pdf: str
    notes: str = ""

# ------------------------
# Helpers
# ------------------------
def best_content(rec: dict) -> Tuple[str, str]:
    # prefer 'content' (what we wrote by default), else markdown->latex->text
    for key in ("content", "markdown", "latex", "text"):
        val = rec.get(key)
        if isinstance(val, str) and val.strip():
            return val, key
    return "", "none"

def clamp(s: str, max_chars: int = 180_000) -> str:
    return s if len(s) <= max_chars else s[:max_chars] + "\n\n[TRUNCATED]"

# ------------------------
# Specialized prompts
# ------------------------
def prompt_for(contest: str, year: str, content_format: str, content: str, expected: int) -> str:
    cU = contest.upper()

    base_rules = [
        f"Contest: {contest}",
        f"Year: {year}",
        f"- EXPECTED NUMBER OF PROBLEMS: EXACTLY {expected}. If fewer are present in the text, output exactly those found; do not invent content.",
        "- Extract EVERY problem **and** its MATCHED official solution from the provided text.",
        '- If a solution is not present in the text, set "solution" to "" (do NOT invent).',
        "- Keep LaTeX/math as-is.",
        "- Output ONLY valid JSON with the schema:",
        '{ "problems": [ { "problem_number": "<string or integer>", "problem": "<full problem>", "solution": "<full solution or empty>" } ] }',
        "- JSON only. No commentary, no markdown.",
    ]

    # Tailor by contest
    if cU == "IMOSL":
        extra = [
            "- This is the IMO Shortlist (IMOSL). Problems are labeled by letter in {A,C,G,N} and a number, e.g., A1, C3, G2, N5.",
            '- Use those codes EXACTLY as "problem_number" (e.g., "A3").',
            "- Use the official solutions associated to each problem (often under a 'Solution' header).",
        ]
    elif cU in {"TST", "TSTST"}:
        extra = [
            "- This is a USA TST/TSTST set. Problems are numbered 1..6 (TST) or 1..9 (TSTST).",
            '- Use the numeric index as "problem_number" (e.g., "1", "2", ...).',
            "- Solutions typically start under 'Solutions to Day X' sections; match them to the correct problem.",
        ]
    elif cU == "EGMO":
        extra = [
            "- This is EGMO. Problems are numbered 1..6. Use the numeric index as problem_number.",
            "- Match each problem to its official solution. EGMO PDFs often present problems, then solutions labeled 'Solutions to Problem k'.",
        ]
    else:
        extra = []

    rules = "\n".join(base_rules + extra)

    return f"""
You are given the full {content_format} content of a math contest PDF (already OCR'd).

{rules}

TEXT START
{clamp(content)}
TEXT END
JSON ONLY.
""".strip()

# ------------------------
# OpenAI call (GPT-4o)
# ------------------------
def call_openai(prompt: str) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a meticulous extractor. Output valid JSON only; no commentary."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        # fallback import signature for older installs
        import openai  # type: ignore
        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a meticulous extractor. Output valid JSON only; no commentary."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content)

# ------------------------
# Conservative fallback (regex)
# ------------------------
H_IMOSL = regex.compile(r"(?:(?:^|\n)([ACGN]\d{1,2})\s*[:\.\)])", flags=regex.IGNORECASE)
H_PROB = regex.compile(r"(?:(?:^|\n)Problem\s*(?:No\.|#)?\s*(\d{1,2})\s*[:\.\-])", flags=regex.IGNORECASE)
H_SOL  = regex.compile(r"(?:(?:^|\n)(?:Solution|Solutions?)(?:\s*(?:to)?\s*(?:Problem)?\s*(\d{1,2}))?\s*[:\.\-])", flags=regex.IGNORECASE)

def naive_pairs(contest: str, text: str) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    t = text

    if contest.upper() == "IMOSL":
        labels = list(H_IMOSL.finditer(t))
        for i, m in enumerate(labels):
            pn = m.group(1).upper()
            start = m.end()
            end = labels[i + 1].start() if i + 1 < len(labels) else len(t)
            chunk = t[start:end].strip()
            parts = regex.split(r"(?i)\bSolution\b", chunk, maxsplit=1)
            prob = parts[0].strip()
            sol = parts[1].strip() if len(parts) > 1 else ""
            out.append((pn, prob, sol))
        return out

    probs = list(H_PROB.finditer(t))
    for i, m in enumerate(probs):
        pn = m.group(1)
        start = m.end()
        end = probs[i + 1].start() if i + 1 < len(probs) else len(t)
        chunk = t[start:end].strip()
        ms = H_SOL.search(chunk)
        if ms:
            sn = ms.group(1) or ""
            if sn and pn and sn.strip() != pn.strip():
                prob = chunk
                sol = ""
            else:
                prob = chunk[:ms.start()].strip()
                sol = chunk[ms.end():].strip()
        else:
            prob, sol = chunk, ""
        out.append((pn, prob, sol))
    return out

# ------------------------
# Extraction per record
# ------------------------
def extract_for_record(rec: dict, expected: int, retries: int = 2) -> List[Pair]:
    contest = (rec.get("contest") or "UNKNOWN").upper()
    year = str(rec.get("year") or "UNKNOWN")
    filename = rec.get("filename") or rec.get("source_pdf") or "UNKNOWN.pdf"

    text, fmt = best_content(rec)
    if not text.strip():
        return [Pair(
            id=f"{contest}-{year}-UNKNOWN",
            contest=contest, year=year, problem_number="UNKNOWN",
            problem="", solution="", source_pdf=filename,
            notes="empty-content"
        )]

    last_err = None
    # 1) GPT-4o attempts (strict expected count guidance in prompt)
    for attempt in range(retries + 1):
        try:
            prompt = prompt_for(contest, year, fmt, text, expected)
            data = call_openai(prompt)
            items = data.get("problems", [])
            out: List[Pair] = []
            for it in items:
                pn_raw = it.get("problem_number", "")
                pn = str(pn_raw).strip() if pn_raw is not None else ""
                pid = f"{contest}-{year}-{pn or 'UNKNOWN'}"
                out.append(Pair(
                    id=pid,
                    contest=contest,
                    year=year,
                    problem_number=pn,
                    problem=it.get("problem", "") or "",
                    solution=it.get("solution", "") or "",
                    source_pdf=filename,
                ))
            if out:
                return out
            last_err = "gpt-empty"
        except Exception as e:
            last_err = f"gpt-error: {e}"
            time.sleep(1.2 * (attempt + 1))

    # 2) Fallback regex splitter (best effort)
    naive = naive_pairs(contest, text)
    if naive:
        out: List[Pair] = []
        for pn, prob, sol in naive:
            pid = f"{contest}-{year}-{pn or 'UNKNOWN'}"
            out.append(Pair(
                id=pid,
                contest=contest,
                year=year,
                problem_number=str(pn),
                problem=prob,
                solution=sol,
                source_pdf=filename,
                notes="fallback-regex",
            ))
        return out

    # 3) Stub if nothing
    return [Pair(
        id=f"{contest}-{year}-UNKNOWN",
        contest=contest, year=year, problem_number="UNKNOWN",
        problem="", solution="",
        source_pdf=filename,
        notes=last_err or "no-pairs-found",
    )]

# ------------------------
# Writer (thread-safe)
# ------------------------
_write_lock = threading.Lock()
def write_jsonl_line(out_path: Path, rec: Dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False)
    with _write_lock:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Segment ONLY the specified contests/years from all_contests.jsonl into per-problem pairs with GPT-4o."
    )
    ap.add_argument("input", help="Path to all_contests.jsonl")
    ap.add_argument("-o", "--output", default="rest_contests.jsonl",
                    help="Output JSONL (merged per-problem pairs)")
    ap.add_argument("--workers", type=int, default=10, help="Parallel workers (default 10)")
    ap.add_argument("--retries", type=int, default=2, help="GPT retries per record (default 2)")
    ap.add_argument("--resume", action="store_true", help="Skip problems already present in output by id")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Input file not found: {in_path}")

    out_path = Path(args.output)
    if not args.resume and out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume index
    existing = set()
    if args.resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if "id" in obj:
                        existing.add(obj["id"])
                except Exception:
                    pass

    # Load input JSONL records, filter to target set
    records: List[dict] = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            contest = (obj.get("contest") or "").upper()
            year = str(obj.get("year") or "")
            if (contest, year) in TARGET_KEYS:
                records.append(obj)

    if not records:
        print("[INFO] No matching records for the specified contests/years.")
        return

    print(f"[INFO] Loaded {len(records)} target PDF-level records")
    print(f"[INFO] Writing per-problem lines to: {out_path}")

    # Process
    total_written = 0

    def _one(rec: dict):
        nonlocal total_written
        contest = (rec.get("contest") or "UNKNOWN").upper()
        year = str(rec.get("year") or "UNKNOWN")
        exp = STRICT_EXPECTATIONS.get((contest, year), 0)
        pairs = extract_for_record(rec, expected=exp, retries=args.retries)

        # Filter duplicates if resuming
        if existing:
            pairs = [p for p in pairs if p.id not in existing]

        for p in pairs:
            write_jsonl_line(out_path, asdict(p))
        total_written += len(pairs)
        src = rec.get("filename") or rec.get("source_pdf") or "UNKNOWN"
        print(f"[OK] {contest}-{year} ({src}): wrote {len(pairs)} lines (expected ~{exp})")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(_one, records))

    print(f"[DONE] problem_lines_written={total_written}, output={out_path}")

if __name__ == "__main__":
    main()
