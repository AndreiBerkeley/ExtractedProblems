#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
segment_from_jsonl.py
---------------------
Read Mathpix-derived records from all_contests.jsonl and segment each record's
CONTENT into {problem, solution} pairs. Writes ONE JSONL LINE PER PROBLEM.

- Uses GPT-4o for ALL contests, including IMOSL (as requested).
- Adds contest-specific expectations as hints (APMO=5; EGMO/USAMO/IMO=6; TST/TSTST in {3,6,9,12,...}).
- Thread-safe writer, multithreaded.
- Regex fallback splitter if GPT fails or returns invalid JSON.

INPUT JSONL (one line per PDF) — typically produced by the texify pipeline:
{
  "contest": "EGMO",
  "year": "2024",
  "filename": "egmo_2024.pdf",
  "job_id": "cnv_....",
  "content_format": "markdown",
  "content": "## Problem 1 ... ## Solution 1 ...",
  "notes": "",
  // optionally if you used --keep-all
  "markdown": "...",
  "latex": "...",
  "text": "..."
}

OUTPUT JSONL (one line per PROBLEM):
{
  "id": "EGMO-2024-1" or "IMOSL-2019-A3",
  "contest": "EGMO",
  "year": "2024",
  "problem_number": "1" | "A3",
  "problem": "<problem statement>",
  "solution": "<solution text or empty>",
  "source_pdf": "egmo_2024.pdf",
  "notes": ""
}

Usage:
  export OPENAI_API_KEY=...
  python segment_from_jsonl.py all_contests.jsonl -o out/pairs.jsonl --workers 10

Dependencies:
  pip install regex
"""

import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import regex  # better for large/complex patterns

# ------------------------
# Contest expectations (hints)
# ------------------------
CONTEST_PROFILES: Dict[str, Dict] = {
    "APMO":  {"expected_counts": {5}},
    "EGMO":  {"expected_counts": {6}},
    "USAMO": {"expected_counts": {6}},
    "IMO":   {"expected_counts": {6}},
    "TST":   {"expected_counts": {3, 6, 9, 12}},
    "TSTST": {"expected_counts": {3, 6, 9, 12}},
    # IMOSL: uses A/C/G/N labels; number varies → still handled by GPT with clear guidance
}

IMOSL_LABEL = regex.compile(r"\b([ACGN])(\d{1,2})\b", flags=regex.IGNORECASE)

# ------------------------
# Data model
# ------------------------
@dataclass
class Pair:
    id: str
    contest: str
    year: str
    problem_number: str  # might be "A3" etc
    problem: str
    solution: str
    source_pdf: str
    notes: str = ""

# ------------------------
# Content selection
# ------------------------
def best_content(rec: dict) -> Tuple[str, str]:
    """
    Prefer the 'content' field (what texify wrote by default), else fall back
    to markdown → latex → text.
    Returns (text, format_name)
    """
    for key in ("content", "markdown", "latex", "text"):
        val = rec.get(key)
        if isinstance(val, str) and val.strip():
            return val, key
    return "", "none"

def clamp(s: str, max_chars: int = 160_000) -> str:
    return s if len(s) <= max_chars else s[:max_chars] + "\n\n[TRUNCATED]"

# ------------------------
# OpenAI call (GPT-4o)
# ------------------------
def build_prompt(contest: str, year: str, content_format: str, content: str) -> str:
    profile = CONTEST_PROFILES.get(contest.upper(), {})
    expected = profile.get("expected_counts", set())
    if not expected:
        exp_line = "- EXPECTED NUMBER OF PROBLEMS: (unknown); extract all you find.\n"
    elif len(expected) == 1:
        exp_line = f"- EXPECTED NUMBER OF PROBLEMS: exactly {list(expected)[0]}.\n"
    else:
        exp_line = f"- EXPECTED NUMBER OF PROBLEMS: one of {{{', '.join(map(str, sorted(expected)))}}}.\n"

    imosl_note = ""
    if contest.upper() == "IMOSL":
        imosl_note = (
            "- This is the IMO Shortlist (IMOSL). Problems are labelled by a letter in {A,C,G,N} "
            "for Algebra/Combinatorics/Geometry/Number Theory and a number, e.g., A1, C3, G2, N5.\n"
            "- Use those codes as 'problem_number' exactly (e.g., 'A3').\n"
        )

    return f"""
You are given the full {content_format} content of a math contest PDF.

Contest: {contest}
Year: {year}
{exp_line}{imosl_note}
Task:
- Extract EVERY problem and its MATCHED official solution from the text.
- If a solution is not present, set solution to "" (do NOT invent).
- Keep all LaTeX/math as-is; do not normalize or rewrite.
- Return ONLY valid JSON with this schema:

{{
  "problems": [
    {{
      "problem_number": "<string or integer; IMOSL must be codes like 'A1'>",
      "problem": "<full problem statement>",
      "solution": "<full official solution, or '' if truly missing>"
    }}
  ]
}}

IMPORTANT:
- Make sure the solution belongs to the same problem_number.
- Do not include editorial commentary or headers unrelated to the problem.
- JSON only. No markdown, no explanations.

TEXT START
{clamp(content)}
TEXT END
JSON ONLY.
""".strip()

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
                {"role": "system", "content": "You are a careful extractor. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        # fallback import style
        import openai  # type: ignore
        client = openai.OpenAI(api_key=api_key)  # same signature in recent versions
        resp = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a careful extractor. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content)

# ------------------------
# Fallback regex splitter (very conservative)
# ------------------------
H_PROB = regex.compile(
    r"(?:(?:^|\n)(?:Problem\s*(?:\#|No\.?)?\s*|P(?:rob)?\s*)?(\d{1,2})\s*[:\.\-]|^##\s*Problem\s*(\d{1,2}))",
    flags=regex.IGNORECASE
)
H_SOL = regex.compile(
    r"(?:(?:^|\n)(?:Solution|Sol\.?|Proof)\s*(\d{1,2})?\s*[:\.\-]|^##\s*Solution\s*(\d{1,2}))",
    flags=regex.IGNORECASE
)
H_IMOSL = regex.compile(r"(?:(?:^|\n)([ACGN]\d{1,2})\s*[:\.\)]|^##\s*([ACGN]\d{1,2}))", flags=regex.IGNORECASE)

def naive_split_pairs(contest: str, text: str) -> List[Tuple[str, str, str]]:
    """
    Extremely simple fallback:
    - For IMOSL: find A1/C3/G2/N5 headings, split until next label; split 'Solution' inside each chunk.
    - For others: find Problem k ... (until next Problem), split at Solution (k) if present.
    Returns list of (problem_number, problem_text, solution_text).
    """
    out: List[Tuple[str, str, str]] = []
    t = text

    if contest.upper() == "IMOSL":
        matches = list(H_IMOSL.finditer(t))
        for i, m in enumerate(matches):
            pn = (m.group(1) or m.group(2) or "").upper()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(t)
            chunk = t[start:end].strip()
            # split by Solution marker within chunk
            parts = regex.split(r"(?i)\bSolution\b|(?i)^##\s*Solution\b", chunk, maxsplit=1)
            prob = parts[0].strip()
            sol = parts[1].strip() if len(parts) > 1 else ""
            if pn:
                out.append((pn, prob, sol))
        return out

    # Non-IMOSL: generic Problem N / Solution N headings
    # First, find all problem starts
    # We'll capture number from group 1 or 2
    iters = list(H_PROB.finditer(t))
    for i, m in enumerate(iters):
        pn = m.group(1) or m.group(2) or ""
        start = m.end()
        end = iters[i + 1].start() if i + 1 < len(iters) else len(t)
        chunk = t[start:end].strip()

        # Look for solution inside this chunk
        ms = H_SOL.search(chunk)
        if ms:
            # If solution header has a number, ensure it matches (when present)
            sn = ms.group(1) or ms.group(2) or ""
            if sn and pn and sn.strip() != pn.strip():
                # mismatched labelled solution; keep all as problem, set empty solution
                prob = chunk
                sol = ""
            else:
                prob = chunk[:ms.start()].strip()
                sol = chunk[ms.end():].strip()
        else:
            prob = chunk
            sol = ""
        if pn:
            out.append((pn, prob, sol))
    return out

# ------------------------
# Worker logic
# ------------------------
def extract_pairs_for_record(rec: dict, *, retries: int = 2) -> List[Pair]:
    contest = (rec.get("contest") or "UNKNOWN").upper()
    year = str(rec.get("year") or "UNKNOWN")
    filename = rec.get("filename") or rec.get("source_pdf") or "UNKNOWN.pdf"

    text, fmt = best_content(rec)
    if not text.strip():
        # no content available
        return [Pair(
            id=f"{contest}-{year}-UNKNOWN",
            contest=contest, year=year, problem_number="UNKNOWN",
            problem="", solution="", source_pdf=filename,
            notes="empty-content"
        )]

    # 1) Try GPT-4o
    last_err = None
    for attempt in range(retries + 1):
        try:
            prompt = build_prompt(contest, year, fmt, text)
            data = call_openai(prompt)
            out_pairs: List[Pair] = []
            for item in data.get("problems", []):
                pn_raw = item.get("problem_number", "")
                pn = str(pn_raw).strip() if pn_raw is not None else ""
                pid = f"{contest}-{year}-{pn or 'UNKNOWN'}"
                out_pairs.append(Pair(
                    id=pid,
                    contest=contest,
                    year=year,
                    problem_number=pn,
                    problem=item.get("problem", "") or "",
                    solution=item.get("solution", "") or "",
                    source_pdf=filename,
                ))
            if out_pairs:
                return out_pairs
            last_err = "gpt-empty"
        except Exception as e:
            last_err = f"gpt-error: {e}"
            time.sleep(1.5 * (attempt + 1))

    # 2) Fallback: naive regex splitter
    naive = naive_split_pairs(contest, text)
    if naive:
        out_pairs = []
        for pn, prob, sol in naive:
            pid = f"{contest}-{year}-{pn or 'UNKNOWN'}"
            out_pairs.append(Pair(
                id=pid,
                contest=contest,
                year=year,
                problem_number=str(pn),
                problem=prob,
                solution=sol,
                source_pdf=filename,
                notes="fallback-regex",
            ))
        return out_pairs

    # 3) Still nothing → write a stub record
    return [Pair(
        id=f"{contest}-{year}-UNKNOWN",
        contest=contest, year=year, problem_number="UNKNOWN",
        problem="", solution="",
        source_pdf=filename,
        notes=last_err or "no-pairs-found",
    )]

# ------------------------
# Thread-safe writer
# ------------------------
_write_lock = threading.Lock()
def write_jsonl_line(out_path: Path, rec: Dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False)
    with _write_lock:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

# ------------------------
# Main runner
# ------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Segment all_contests.jsonl into per-problem JSONL using GPT-4o (all contests, including IMOSL)."
    )
    parser.add_argument("input", help="Path to all_contests.jsonl")
    parser.add_argument("-o", "--output", default="out/pairs.jsonl",
                        help="Output per-problem JSONL (merged)")
    parser.add_argument("--workers", type=int, default=10, help="Parallel workers (default 10)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip problems that already exist in output (by id)")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Input file not found: {in_path}")

    out_path = Path(args.output)
    if not args.resume:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink()

    # Build an index of existing ids if resuming
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
                    continue

    # Load all input records
    records: List[dict] = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] Skipping invalid JSONL line: {e}", file=sys.stderr)

    print(f"[INFO] Loaded {len(records)} PDF-level records from {in_path}")
    print(f"[INFO] Writing per-problem lines to {out_path}")

    ok_pairs = 0
    files_done = 0

    def _one(rec: dict):
        nonlocal ok_pairs, files_done
        pairs = extract_pairs_for_record(rec)
        # If resuming, filter out already-present ids
        if existing:
            pairs = [p for p in pairs if p.id not in existing]
        for p in pairs:
            write_jsonl_line(out_path, asdict(p))
        files_done += 1
        ok_pairs += len(pairs)
        src = rec.get("filename") or rec.get("source_pdf") or "UNKNOWN"
        print(f"[OK] {src}: wrote {len(pairs)} problem lines")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(_one, records))

    print(f"[DONE] files_processed={files_done}, problem_lines_written={ok_pairs}, output={out_path}")

if __name__ == "__main__":
    main()
