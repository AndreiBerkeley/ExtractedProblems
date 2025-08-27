#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
parse_apmo_from_jsonl.py
------------------------
Parse APMO entries in all_contests.jsonl into per-problem JSONL with schema
compatible with pairs.jsonl. Includes any trailing "Note:" lines as part of the
problem statement and concatenates multiple solution sections into one
"solution" field.

Usage:
  python parse_apmo_from_jsonl.py /path/to/all_contests.jsonl \
    -o /path/to/apmo_problems.jsonl
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Pair:
    id: str
    contest: str
    year: str
    problem_number: str
    problem: str
    solution: str
    source_pdf: str
    notes: str = ""


def best_content(rec: Dict) -> Tuple[str, str]:
    for key in ("content", "markdown", "latex", "text"):
        val = rec.get(key)
        if isinstance(val, str) and val.strip():
            return val, key
    return "", "none"


class ApmoParser:
    """Heuristic parser for APMO PDFs converted to text/markdown.

    Rules:
    - Problem header detection prefers explicit forms like "\\section*{Problem N}",
      "Problem N.", or "## Problem N".
    - Also accepts top-level numeric headings like "N." when not inside a
      solution section.
    - Problem text includes everything after its header up to (but not
      including) the first Solution heading. This preserves any trailing
      "Note:" lines as part of the problem statement.
    - Solution text comprises all content from the first Solution heading up to
      the next problem header (or end of document). Multiple Solution headings
      (e.g., Solution 1, Solution 2b) are concatenated with blank lines.
    """

    # Problem header patterns
    _re_section_prob = re.compile(r"^\s*\\section\*\{\s*Problem\s+(\d{1,2})\s*\}\s*$", re.IGNORECASE)
    _re_md_prob = re.compile(r"^\s*##\s*Problem\s+(\d{1,2})\b", re.IGNORECASE)
    _re_plain_prob = re.compile(r"^\s*Problem\s+(\d{1,2})\s*[:\.)]", re.IGNORECASE)
    _re_numeric_prob = re.compile(r"^\s*(\d{1,2})\s*\.[ \t]", re.IGNORECASE)

    # Solution heading patterns
    _re_section_solution = re.compile(r"^\s*\\section\*\{\s*Solution[^}]*\}\s*$", re.IGNORECASE)
    _re_md_solution = re.compile(r"^\s*##\s*Solution\b", re.IGNORECASE)
    _re_solution = re.compile(r"^\s*Solution\b", re.IGNORECASE)

    def parse(self, text: str) -> List[Tuple[str, str, str]]:
        lines = text.splitlines()

        def is_problem_header(line: str, allow_numeric: bool) -> Optional[Tuple[str, int]]:
            m = self._re_section_prob.match(line)
            if m:
                return m.group(1), m.end()
            m = self._re_md_prob.match(line)
            if m:
                return m.group(1), m.end()
            m = self._re_plain_prob.match(line)
            if m:
                return m.group(1), m.end()
            if allow_numeric:
                m = self._re_numeric_prob.match(line)
                if m:
                    return m.group(1), m.end()
            return None

        def is_solution_header(line: str) -> bool:
            return bool(
                self._re_section_solution.match(line)
                or self._re_md_solution.match(line)
                or self._re_solution.match(line)
            )

        results: List[Tuple[str, str, str]] = []
        mode: str = "idle"  # idle | in_problem | in_solution
        current_pn: Optional[str] = None
        problem_buf: List[str] = []
        solution_buf: List[str] = []
        last_line_was_blank: bool = True
        last_was_solution_heading: bool = False

        # To reduce false positives of numeric headings within solutions, only
        # allow numeric-based problem detection when not currently in a solution.
        for raw_line in lines:
            line = raw_line.rstrip("\n")

            # First, try non-numeric problem headers
            pn_non_numeric = is_problem_header(line, allow_numeric=False)
            # Then try numeric headers with context-sensitive allowance inside solutions
            pn_numeric = None
            if pn_non_numeric is None:
                pn_candidate = is_problem_header(line, allow_numeric=True)
                if pn_candidate is not None:
                    if mode != "in_solution":
                        pn_numeric = pn_candidate
                    else:
                        # Inside a solution: accept numeric as a new problem if it
                        # appears right after a Solution heading OR after a blank line.
                        # This matches APMO 2022 formatting where problems are
                        # numbered 1., 2., ... under a top-level Solution section.
                        if last_was_solution_heading or last_line_was_blank:
                            pn_numeric = pn_candidate

            pn_match = pn_non_numeric or pn_numeric
            if pn_match is not None:
                pn, header_end_idx = pn_match
                if current_pn is not None:
                    results.append(
                        (
                            current_pn,
                            _rstrip_blanklines("\n".join(problem_buf)),
                            _rstrip_blanklines("\n".join(solution_buf)),
                        )
                    )
                current_pn = pn
                problem_buf = []
                solution_buf = []
                mode = "in_problem"
                last_was_solution_heading = False
                # Capture any text on the same line after the header
                remainder = line[header_end_idx:].lstrip()
                if remainder:
                    problem_buf.append(remainder)
                continue

            if is_solution_header(line) and current_pn is not None:
                # Start or continue solution aggregation
                if mode != "in_solution" and solution_buf and solution_buf[-1] != "":
                    solution_buf.append("")
                mode = "in_solution"
                # Keep the solution heading to preserve structure for multiple solutions
                solution_buf.append(line)
                last_was_solution_heading = True
                last_line_was_blank = False
                continue

            if mode == "in_solution" and current_pn is not None:
                solution_buf.append(line)
            elif mode == "in_problem" and current_pn is not None:
                problem_buf.append(line)
            else:
                # idle: ignore content until the first problem header
                pass

            last_line_was_blank = (line.strip() == "")

        if current_pn is not None:
            results.append(
                (
                    current_pn,
                    _rstrip_blanklines("\n".join(problem_buf)),
                    _rstrip_blanklines("\n".join(solution_buf)),
                )
            )

        # Filter out entries with empty problem text
        results = [r for r in results if r[1].strip()]
        return results


def _rstrip_blanklines(s: str) -> str:
    lines = s.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def extract_apmo_pairs(rec: Dict) -> List[Pair]:
    if (rec.get("contest") or "").upper() != "APMO":
        return []
    year = str(rec.get("year") or "UNKNOWN")
    print(year)
    filename = rec.get("filename") or rec.get("source_pdf") or "UNKNOWN.pdf"
    content, fmt = best_content(rec)
    if not content.strip():
        return []

    parser = ApmoParser()
    triples = parser.parse(content)

    pairs: List[Pair] = []
    for pn, prob, sol in triples:
        pid = f"APMO-{year}-{pn}"
        pairs.append(
            Pair(
                id=pid,
                contest="APMO",
                year=year,
                problem_number=str(pn),
                problem=prob.strip(),
                solution=sol.strip(),
                source_pdf=filename,
                notes="",
            )
        )
    return pairs


def run(input_path: Path, output_path: Path) -> Tuple[int, int]:
    written = 0
    total_inputs = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if (obj.get("contest") or "").upper() != "APMO":
                continue
            total_inputs += 1
            pairs = extract_apmo_pairs(obj)
            for p in pairs:
                fout.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
                written += 1
    return total_inputs, written


def main():
    parser = argparse.ArgumentParser(description="Parse APMO entries into per-problem JSONL (apmo_problems.jsonl)")
    parser.add_argument("input", help="Path to all_contests.jsonl")
    parser.add_argument("-o", "--output", default="apmo_problems.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Input file not found: {in_path}")
    out_path = Path(args.output)
    total_files, total_pairs = run(in_path, out_path)
    print(f"[DONE] apmo_files_processed={total_files}, problem_lines_written={total_pairs}, output={out_path}")


if __name__ == "__main__":
    main()


