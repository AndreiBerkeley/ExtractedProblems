#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_evaluate.py
-------------------
Collect requested problems from existing JSONL sources and write them to
evaluate.jsonl. Prints basic statistics per contest and per year.

Sources used:
- pairs.jsonl (default all contests)
- apmo_problems.jsonl (preferred for APMO)

Requested sets:
- IMO:    2024, 2023, 2022
- USAMO:  2024, 2023, 2022
- APMO:   2025, 2024, 2023, 2022
- EGMO:   2025, 2024, 2023, 2022
- Putnam: 2023, 2022
- TST:    2025, 2023, 2022

Usage:
  python collect_evaluate.py \
    --pairs pairs.jsonl --apmo apmo_problems.jsonl \
    -o evaluate.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


REQUESTS: Dict[str, Set[str]] = {
    "IMO": {"2024", "2023", "2022"},
    "USAMO": {"2024", "2023", "2022"},
    "APMO": {"2025", "2024", "2023", "2022"},
    "EGMO": {"2025", "2024", "2023", "2022"},
    "PUTNAM": {"2023", "2022"},
    "TST": {"2025", "2023", "2022"},
}


def load_jsonl(path: Path) -> Iterable[Dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def want_record(rec: Dict) -> bool:
    contest = str(rec.get("contest") or "").strip()
    year = str(rec.get("year") or "").strip()
    if not contest or not year:
        return False
    want_years = REQUESTS.get(contest)
    return want_years is not None and year in want_years


def main():
    parser = argparse.ArgumentParser(description="Collect requested problems into evaluate.jsonl and report stats")
    parser.add_argument("--pairs", default="pairs.jsonl", help="Path to pairs.jsonl")
    parser.add_argument("-o", "--output", default="evaluate.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    out_path = Path(args.output)

    # Load all candidates
    pairs = [rec for rec in load_jsonl(pairs_path) if want_record(rec)]

    # Prefer APMO entries from apmo_problems.jsonl when available
    by_id: Dict[str, Dict] = {}
    for rec in pairs:
        rid = str(rec.get("id") or "")
        if not rid:
            continue
        by_id[rid] = rec

    # Filter to requested contests explicitly (in case pairs has extras)
    selected: List[Dict] = [r for r in by_id.values() if want_record(r)]

    # Sort deterministically: by contest, year, problem_number (numeric if possible)
    def pn_key(v: str) -> Tuple[int, str]:
        try:
            return (0, f"{int(v):03d}")
        except Exception:
            return (1, v)

    selected.sort(key=lambda r: (
        str(r.get("contest") or ""),
        str(r.get("year") or ""),
        pn_key(str(r.get("problem_number") or "")),
    ))

    # Write evaluate.jsonl
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        for rec in selected:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Stats
    total = len(selected)
    by_contest: Dict[str, int] = defaultdict(int)
    by_cy: Dict[Tuple[str, str], int] = defaultdict(int)
    for r in selected:
        c = str(r.get("contest") or "")
        y = str(r.get("year") or "")
        by_contest[c] += 1
        by_cy[(c, y)] += 1

    print(f"[DONE] wrote {total} problems to {out_path}")
    print("By contest:")
    for c in sorted(by_contest.keys()):
        print(f"  {c}: {by_contest[c]}")
    print("By contest-year:")
    for (c, y) in sorted(by_cy.keys()):
        print(f"  {c} {y}: {by_cy[(c, y)]}")


if __name__ == "__main__":
    main()


