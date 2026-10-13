#!/usr/bin/env python3
"""Compare a benchmark output file with its native (NVIDIA) reference.

Text files: integers must match exactly; other numbers must agree within
|out - ref| <= atol + rtol * |ref|;
everything else must match exactly. Different GPUs and compilers legitimately
differ in the last printed digit (FMA contraction, libm), so an exact diff
reports correct results as failures.
Binary files: byte-exact.

Usage: compare_results.py OUT REF [--rtol R] [--atol A]
Exit status: 0 = match, 1 = mismatch, 2 = missing/unreadable file.
"""
import argparse
import re
import sys

NUM = re.compile(r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")
INT = re.compile(r"[-+]?\d+")


def tokens(text):
    # Split every whitespace-separated word into numeric and non-numeric parts,
    # so "Distance=0.000123" compares "Distance=" exactly and the number loosely.
    for word in text.split():
        for part in NUM.split(word):
            if part:
                yield part


def compare(out_path, ref_path, rtol, atol):
    try:
        out_b, ref_b = open(out_path, "rb").read(), open(ref_path, "rb").read()
    except OSError as e:
        print(f"MISSING {e.filename}")
        return 2
    try:
        out_t, ref_t = out_b.decode(), ref_b.decode()
    except UnicodeDecodeError:
        ok = out_b == ref_b
        print("OK (binary, exact)" if ok else
              f"DIFF (binary) {sum(a != b for a, b in zip(out_b, ref_b)) + abs(len(out_b) - len(ref_b))} bytes differ")
        return 0 if ok else 1

    out_k, ref_k = list(tokens(out_t)), list(tokens(ref_t))
    if len(out_k) != len(ref_k):
        print(f"DIFF token count {len(out_k)} vs reference {len(ref_k)}")
        return 1
    bad, worst, first = 0, 0.0, None
    for i, (o, r) in enumerate(zip(out_k, ref_k)):
        if INT.fullmatch(r) and INT.fullmatch(o):
            # integers (keys, indices, counts) are exact quantities
            if int(o) != int(r):
                bad += 1
                first = first or (i, o, r)
        elif NUM.fullmatch(r) and NUM.fullmatch(o):
            fo, fr = float(o), float(r)
            err = abs(fo - fr)
            if err > atol + rtol * abs(fr):
                bad += 1
                first = first or (i, o, r)
            worst = max(worst, err / abs(fr) if fr else err)
        elif o != r:
            bad += 1
            first = first or (i, o, r)
    if bad:
        print(f"DIFF {bad}/{len(ref_k)} tokens, first #{first[0]}: {first[1]!r} vs {first[2]!r}")
        return 1
    print(f"OK (max rel err {worst:.2e})")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("ref")
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--atol", type=float, default=1e-6)
    a = ap.parse_args()
    sys.exit(compare(a.out, a.ref, a.rtol, a.atol))
