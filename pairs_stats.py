#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pairs_stats.py
---------------
Print statistics for all problems in pairs.jsonl:
- Total count
- Counts by contest
- Counts by contest-year

Usage:
  python pairs_stats.py --pairs pairs.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable


def load_jsonl(path: Path) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def main():
    parser = argparse.ArgumentParser(description="Show statistics for pairs.jsonl")
    parser.add_argument("--pairs", default="pairs.jsonl", help="Path to pairs.jsonl")
    args = parser.parse_args()

    path = Path(args.pairs)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    total = 0
    by_contest: Dict[str, int] = defaultdict(int)
    by_cy: Dict[str, int] = defaultdict(int)

    for rec in load_jsonl(path):
        total += 1
        c = str(rec.get("contest") or "")
        y = str(rec.get("year") or "")
        by_contest[c] += 1
        by_cy[f"{c} {y}"] += 1

    print(f"Total problems: {total}")
    print("By contest:")
    for c in sorted(by_contest.keys()):
        print(f"  {c}: {by_contest[c]}")
    print("By contest-year:")
    for cy in sorted(by_cy.keys()):
        print(f"  {cy}: {by_cy[cy]}")


if __name__ == "__main__":
    main()


