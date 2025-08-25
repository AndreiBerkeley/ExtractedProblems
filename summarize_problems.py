#!/usr/bin/env python3
"""
Summarize problem statistics from a JSONL file (default: pairs.jsonl).
Reports counts for contests, years, sources, and whether entries contain images via \includegraphics.
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Any, Iterable, Tuple


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def normalize_source(value: str) -> str:
    if not value:
        return "Unknown"
    # If it's a path or URL, use the last path segment; else return as-is
    tail = value.strip().split("?")[0]
    tail = tail.rstrip("/")
    if "/" in tail:
        tail = tail.split("/")[-1]
    return tail or "Unknown"


def has_includegraphics(text: str) -> bool:
    if not text:
        return False
    # Look for LaTeX includegraphics command
    return re.search(r"\\includegraphics\b", text) is not None


def summarize(path: str) -> Tuple[int, Counter, Counter, Counter, int]:
    total = 0
    contests: Counter = Counter()
    years: Counter = Counter()
    sources: Counter = Counter()
    images = 0

    for obj in read_jsonl(path):
        total += 1
        contests[obj.get("contest", "Unknown")] += 1
        years[str(obj.get("year", "Unknown"))] += 1
        sources[normalize_source(obj.get("source_pdf", "Unknown"))] += 1

        if has_includegraphics(obj.get("problem", "")) or has_includegraphics(obj.get("solution", "")):
            images += 1

    return total, contests, years, sources, images


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize problem statistics from JSONL.")
    parser.add_argument("--file", "-f", default="pairs.jsonl", help="Path to JSONL file (default: pairs.jsonl)")
    parser.add_argument("--top", type=int, default=50, help="Show up to N entries per category (default: 50)")
    args = parser.parse_args()

    path = args.file
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    total, contests, years, sources, images = summarize(path)

    def print_counter(title: str, counter: Counter, limit: int) -> None:
        print(f"\n{title} (unique: {len(counter)}):")
        for key, cnt in counter.most_common(limit):
            print(f"  {key}: {cnt}")

    print(f"File: {os.path.abspath(path)}")
    print(f"Total problems: {total}")
    print(f"Contains images (\\includegraphics): {images} ({(images/total*100 if total else 0):.1f}%)")

    print_counter("Contests", contests, args.top)
    print_counter("Years", Counter(dict(sorted(years.items(), key=lambda kv: (kv[0])))), args.top)
    print_counter("Sources (source_pdf)", sources, args.top)


if __name__ == "__main__":
    main() 