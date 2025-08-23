#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
texify.py
---------
Best-practice Mathpix PDF conversion:
- Use /v3/pdf to process PDFs and fetch structured outputs.
- Prefer result.markdown; fallback to latex_styled, then text.
- Write per-contest JSONL and (optionally) a merged JSONL.

Usage:
  export MATHPIX_APP_ID="..."
  export MATHPIX_APP_KEY="..."

  python texify.py pdf_contests \
    -o out_pdf \
    --recursive \
    --workers 8 \
    --merge \
    --prefer markdown \
    --rate-per-min 30

Requires:
  pip install requests
  (optional) pip install mpxpy
"""

import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# Optional SDK (we fall back to HTTP if absent)
try:
    import mpxpy as mpx
    HAS_SDK = True
except Exception:
    HAS_SDK = False

MPX_BASE = "https://api.mathpix.com"
PDF_POST = f"{MPX_BASE}/v3/pdf"
PDF_GET  = f"{MPX_BASE}/v3/pdf/{{id}}"

CONTEST_MAP = {"APMO","EGMO","USAMO","IMO","IMOSL","TST","TSTST"}
STEM_RE = re.compile(r"(?i)([A-Za-z]+)[\-_ ]*((?:19|20)\d{2})")

def infer_contest_year(pdf_path: Path) -> Tuple[str, str]:
    """Infer contest name and year from filename"""
    stem = pdf_path.stem.lower()  # Convert to lowercase for easier matching
    
    # Try the regex pattern first
    m = STEM_RE.search(stem)
    if m:
        contest_raw = m.group(1).upper()
        year = m.group(2)
        
        # Handle special cases
        if contest_raw in {"IMOSL", "IMO-SL", "IMO_SHORTLIST", "IMO-SHORTLIST", "IMOS"}:
            contest_raw = "IMOSL"
        
        # Check if it's a known contest
        if contest_raw in CONTEST_MAP:
            return contest_raw, year
    
    # Fallback: try to extract contest and year separately
    contest = "OTHER"
    year = "UNKNOWN"
    
    # Look for known contest names
    for known_contest in CONTEST_MAP:
        if known_contest.lower() in stem:
            contest = known_contest
            break
    
    # Look for year pattern
    year_match = re.search(r'((?:19|20)\d{2})', stem)
    if year_match:
        year = year_match.group(1)
    
    return contest, year

def mpx_headers() -> Dict[str, str]:
    """Get Mathpix API headers"""
    app_id = os.getenv("MATHPIX_APP_ID")
    app_key = os.getenv("MATHPIX_APP_KEY")
    if not app_id or not app_key:
        raise RuntimeError("Set MATHPIX_APP_ID and MATHPIX_APP_KEY environment variables.")
    return {"app_id": app_id, "app_key": app_key}

class RateLimiter:
    """Thread-safe rate limiter"""
    def __init__(self, rate_per_min: float):
        self.interval = max(0.1, 60.0 / max(rate_per_min, 0.1))
        self._lock = threading.Lock()
        self._last = 0.0
        
    def acquire(self):
        with self._lock:
            now = time.time()
            wait = (self._last + self.interval) - now
            if wait > 0:
                time.sleep(wait)
                now = time.time()
            self._last = now

_rate_limiter: Optional[RateLimiter] = None

# ---------- HTTP path ----------
def http_submit_pdf(pdf_bytes: bytes) -> str:
    """Submit PDF to Mathpix API and return job ID"""
    if _rate_limiter: 
        _rate_limiter.acquire()
    
    headers = mpx_headers()
    files = {"file": ("document.pdf", pdf_bytes, "application/pdf")}
    # Fix: Don't specify conversion_formats - let Mathpix use defaults
    data = {"options_json": json.dumps({})}
    
    try:
        r = requests.post(PDF_POST, headers=headers, files=files, data=data, timeout=120)
        
        if r.status_code == 401:
            raise RuntimeError("Unauthorized: check MATHPIX_APP_ID / MATHPIX_APP_KEY.")
        elif r.status_code == 429:
            raise RuntimeError("Rate limit exceeded. Please wait or reduce rate-per-min.")
        
        r.raise_for_status()
        j = r.json()
        
        # Accept either key
        conv_id = j.get("conversion_id") or j.get("pdf_id")
        if not conv_id:
            raise RuntimeError(f"Mathpix did not return conversion_id/pdf_id: {j}")
        return conv_id
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"HTTP request failed: {e}")

def http_poll_pdf(job_id: str, timeout_s: int = 900, interval_s: float = 3.0) -> Dict:
    """Poll Mathpix API for job completion and get the converted content"""
    headers = mpx_headers()
    t0 = time.time()
    last_status = ""
    
    while True:
        try:
            # Check status first
            r = requests.get(PDF_GET.format(id=job_id), headers=headers, timeout=60)
            r.raise_for_status()
            j = r.json()
            
            status = (j.get("status") or j.get("state") or "").lower()
            
            if status != last_status:
                print(f"[INFO] Job {job_id}: status = {status}")
                last_status = status
            
            if status in {"completed", "success"}:
                # Now get the actual converted content
                print(f"[INFO] Job {job_id} completed, fetching content...")
                
                # Try to get Mathpix Markdown (.mmd) content first
                mmd_url = f"{MPX_BASE}/v3/pdf/{job_id}.mmd"
                content_response = requests.get(mmd_url, headers=headers, timeout=60)
                
                if content_response.status_code == 200:
                    mmd_content = content_response.text
                    return {
                        "status": "completed",
                        "mmd": mmd_content,
                        "markdown": mmd_content,  # Use mmd as markdown
                        "text": mmd_content      # Use mmd as text fallback
                    }
                else:
                    print(f"[WARN] Could not fetch .mmd content: {content_response.status_code}")
                    # Fall back to original status response
                    return j
                    
            if status == "error":
                error_msg = j.get("error", {}).get("message", "Unknown error")
                raise RuntimeError(f"Mathpix job error: {error_msg}")
                
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"Timed out waiting for Mathpix job {job_id}")
                
            time.sleep(interval_s)
            
        except requests.exceptions.RequestException as e:
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"Timed out with request error: {e}")
            print(f"[WARN] Polling error, retrying: {e}")
            time.sleep(interval_s * 2)  # Wait longer on request errors

# ---------- SDK path (optional) ----------
def sdk_convert_to_result(pdf_bytes: bytes, timeout_s: int = 900) -> Dict:
    """Convert PDF using Mathpix SDK"""
    if _rate_limiter: 
        _rate_limiter.acquire()
    
    app_id = os.getenv("MATHPIX_APP_ID")
    app_key = os.getenv("MATHPIX_APP_KEY")
    client = mpx.Client(app_id=app_id, app_key=app_key)
    pdf = client.pdf_from_bytes(pdf_bytes)
    pdf.wait_until_complete(timeout=timeout_s)
    
    result = {
        "markdown": pdf.to_md_text() or "",
        "latex_styled": pdf.to_latex_text() or "",
        "text": pdf.to_text() or "",
    }
    return {"status": "completed", "result": result}

# ---------- content selection ----------
def choose_content(result: Dict, prefer: str) -> Tuple[str, str, Dict]:
    """Choose the best content format from Mathpix result"""
    md = result.get("markdown", "")
    latex = result.get("latex_styled", "")
    text = result.get("text", "")
    
    # Handle both string and dict formats for each field
    if isinstance(md, dict):
        md = md.get("text", "")
    if isinstance(latex, dict):
        latex = latex.get("text", "")
    if isinstance(text, dict):
        text = text.get("text", "")
    
    # Ensure all are strings
    md = str(md) if md else ""
    latex = str(latex) if latex else ""
    text = str(text) if text else ""
    
    order = {
        "markdown": [("markdown", md), ("latex", latex), ("text", text)],
        "latex":    [("latex", latex), ("markdown", md), ("text", text)],
        "text":     [("text", text), ("markdown", md), ("latex", latex)],
    }[prefer]
    
    for fmt, val in order:
        if val and val.strip():
            return val, fmt, {"markdown": md, "latex": latex, "text": text}
    
    return "", "none", {"markdown": md, "latex": latex, "text": text}

# ---------- writers (thread-safe) ----------
_write_locks: Dict[str, threading.Lock] = {}
_write_locks_lock = threading.Lock()
_merge_lock = threading.Lock()

def _file_lock(path: Path) -> threading.Lock:
    """Get or create a lock for a specific file path"""
    key = str(path.resolve())
    with _write_locks_lock:
        if key not in _write_locks:
            _write_locks[key] = threading.Lock()
        return _write_locks[key]

def write_jsonl_line(path: Path, record: dict):
    """Thread-safely write a JSON line to file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _file_lock(path)
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def write_per_contest_and_merged(out_dir: Path, contest: str, rec: dict, merged_path: Optional[Path]):
    """Write record to both contest-specific and merged files"""
    contest_file = out_dir / f"{contest}.jsonl"
    write_jsonl_line(contest_file, rec)
    
    if merged_path is not None:
        with _merge_lock:
            write_jsonl_line(merged_path, rec)

# ---------- file collection ----------
def collect_pdfs(root: Path, recursive: bool) -> List[Path]:
    """Collect all PDF files from the given path"""
    if not root.exists():
        print(f"[ERROR] Path does not exist: {root}")
        return []
    
    if root.is_file() and root.suffix.lower() == ".pdf":
        return [root]
    
    if root.is_dir():
        if recursive:
            pdfs = list(root.rglob("*.pdf"))
        else:
            pdfs = list(root.glob("*.pdf"))
        return sorted(pdfs)
    
    return []

# ---------- worker ----------
def process_pdf(pdf_path: Path, out_dir: Path, merged_path: Optional[Path],
                prefer: str, keep_all: bool, retries: int, use_sdk: bool):
    """Process a single PDF file"""
    contest, year = infer_contest_year(pdf_path)
    
    print(f"[INFO] Processing {pdf_path.name} (contest: {contest}, year: {year})")

    try:
        pdf_bytes = pdf_path.read_bytes()
        print(f"[INFO] Read {len(pdf_bytes)} bytes from {pdf_path.name}")
    except Exception as e:
        error_rec = {
            "contest": contest, 
            "year": year, 
            "filename": pdf_path.name,
            "job_id": None, 
            "content_format": "none", 
            "content": "",
            "notes": f"read-error: {e}"
        }
        write_per_contest_and_merged(out_dir, contest, error_rec, merged_path)
        print(f"[ERROR] Failed to read {pdf_path.name}: {e}")
        return

    attempt = 0
    job_id = None
    
    while attempt < retries:
        try:
            print(f"[INFO] Attempt {attempt + 1}/{retries} for {pdf_path.name}")
            
            if use_sdk and HAS_SDK:
                print(f"[INFO] Using SDK for {pdf_path.name}")
                j = sdk_convert_to_result(pdf_bytes)
            else:
                print(f"[INFO] Submitting {pdf_path.name} to Mathpix API...")
                job_id = http_submit_pdf(pdf_bytes)
                print(f"[INFO] Got job ID: {job_id} for {pdf_path.name}")
                
                print(f"[INFO] Polling for completion of {pdf_path.name}...")
                j = http_poll_pdf(job_id)

            print(f"[INFO] Conversion completed for {pdf_path.name}")
            
            # For PDF API, the result data is at the top level, not in a nested "result" field
            # Extract content directly from the response
            if "mmd" in j:
                # Use .mmd content if available (Mathpix Markdown format)
                result = {
                    "markdown": j.get("mmd", ""),
                    "latex_styled": "",  # Not available in PDF response
                    "text": j.get("mmd", "")  # Use mmd as fallback for text
                }
            else:
                # Try to extract other format data from response
                result = {
                    "markdown": j.get("markdown", ""),
                    "latex_styled": j.get("latex_styled", ""),
                    "text": j.get("text", "")
                }
            
            # If no content found, list what keys are available
            if not any(result.values()):
                print(f"[DEBUG] Available response keys: {list(j.keys())}")
                print(f"[DEBUG] Response sample: {str(j)[:500]}...")
                raise RuntimeError("No content found in Mathpix PDF response")
            
            content, fmt, extras = choose_content(result, prefer)
            
            rec = {
                "contest": contest,
                "year": year,
                "filename": pdf_path.name,
                "job_id": job_id,
                "content_format": fmt,
                "content": content,
                "content_length": len(content),
                "notes": "",
            }
            
            if keep_all:
                rec.update({
                    "markdown": extras.get("markdown", ""),
                    "latex": extras.get("latex", ""),
                    "text": extras.get("text", ""),
                })
            
            write_per_contest_and_merged(out_dir, contest, rec, merged_path)
            print(f"[SUCCESS] Processed {pdf_path.name} -> {fmt} format, {len(content)} chars")
            return
            
        except Exception as e:
            attempt += 1
            error_msg = str(e)
            print(f"[ERROR] Attempt {attempt} failed for {pdf_path.name}: {error_msg}")
            
            if attempt >= retries:
                error_rec = {
                    "contest": contest, 
                    "year": year, 
                    "filename": pdf_path.name,
                    "job_id": job_id, 
                    "content_format": "none", 
                    "content": "",
                    "notes": f"mathpix-error: {error_msg}"
                }
                write_per_contest_and_merged(out_dir, contest, error_rec, merged_path)
                print(f"[FAILED] Exhausted retries for {pdf_path.name}")
                return
            
            wait = 2.0 * (2 ** (attempt - 1))  # Exponential backoff
            print(f"[WARN] {pdf_path.name}: {error_msg} -> retrying in {wait:.1f}s")
            time.sleep(wait)

# ---------- main ----------
def main():
    p = argparse.ArgumentParser(
        description="Convert PDFs with Mathpix /v3/pdf (or SDK), prefer Markdown; write per-contest + merged JSONL."
    )
    p.add_argument("input", help="PDF file or folder")
    p.add_argument("-o", "--out-dir", default="out_pdf", help="Output folder")
    p.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    p.add_argument("--workers", type=int, default=8, help="Parallel workers (default 8)")
    p.add_argument("--merge", action="store_true", help="Also write a merged JSONL of all records")
    p.add_argument("--merge-file", default=None, help="Merged JSONL path (default: <out-dir>/all_contests.jsonl)")
    p.add_argument("--prefer", choices=["markdown","latex","text"], default="markdown",
                   help="Preferred field to store as 'content' (default markdown)")
    p.add_argument("--keep-all", action="store_true", help="Also store markdown/latex/text fields")
    p.add_argument("--rate-per-min", type=float, default=30.0, help="Global rate limit")
    p.add_argument("--retries", type=int, default=3, help="Retries for submit/poll")
    p.add_argument("--sdk", action="store_true", help="Use Mathpix SDK if installed (mpxpy)")
    args = p.parse_args()

    # Validate credentials early
    try:
        headers = mpx_headers()
        print(f"[INFO] Mathpix credentials validated")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # Collect PDFs
    src = Path(args.input)
    pdfs = collect_pdfs(src, recursive=args.recursive)
    if not pdfs:
        print("[ERROR] No PDFs found.", file=sys.stderr)
        sys.exit(1)

    # Setup output paths
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    merged_path = None
    if args.merge:
        merged_path = Path(args.merge_file) if args.merge_file else out_dir / "all_contests.jsonl"
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        # Clear existing merged file
        if merged_path.exists():
            merged_path.unlink()
            print(f"[INFO] Cleared existing merged file: {merged_path}")

    # Setup rate limiter
    global _rate_limiter
    _rate_limiter = RateLimiter(args.rate_per_min)

    use_sdk = bool(args.sdk and HAS_SDK)
    if args.sdk and not HAS_SDK:
        print("[WARN] SDK requested but mpxpy not available, using HTTP API")

    print(f"[INFO] Found {len(pdfs)} PDF(s)")
    print(f"[INFO] Workers: {args.workers}")
    print(f"[INFO] Prefer: {args.prefer}")
    print(f"[INFO] Keep all formats: {args.keep_all}")
    print(f"[INFO] Using SDK: {use_sdk}")
    print(f"[INFO] Output directory: {out_dir}")
    if merged_path:
        print(f"[INFO] Merged JSONL: {merged_path}")
    print(f"[INFO] Rate limit: ~{args.rate_per_min:.1f}/min")
    print(f"[INFO] Retries per file: {args.retries}")

    def _process_one(pdf_path: Path):
        """Wrapper for processing one PDF with error handling"""
        t0 = time.time()
        try:
            process_pdf(pdf_path, out_dir, merged_path, args.prefer, args.keep_all, args.retries, use_sdk)
            elapsed = time.time() - t0
            print(f"[COMPLETE] {pdf_path.name} processed in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[FATAL] {pdf_path.name} failed after {elapsed:.2f}s: {e}", file=sys.stderr)

    # Process all PDFs
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(_process_one, pdfs))
    
    total_time = time.time() - start_time
    print(f"\n[SUMMARY] Processed {len(pdfs)} PDFs in {total_time:.1f}s")
    print(f"[SUMMARY] Output saved to: {out_dir}")
    if merged_path:
        print(f"[SUMMARY] Merged file: {merged_path}")

if __name__ == "__main__":
    main()
