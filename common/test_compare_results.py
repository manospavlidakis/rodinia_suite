#!/usr/bin/env python3
"""Self-check for compare_results.py: python3 common/test_compare_results.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_results import compare


def check(out, ref, want, rtol=1e-4, atol=1e-6):
    with tempfile.TemporaryDirectory() as d:
        o, r = os.path.join(d, "o"), os.path.join(d, "r")
        for p, s in ((o, out), (r, ref)):
            open(p, "wb").write(s if isinstance(s, bytes) else s.encode())
        got = compare(o, r, rtol, atol)
        assert got == want, (out, ref, got, want)


check("0  840187  6001\n", "0  840187  6001\n", 0)               # exact ints
check("0  840188  6001\n", "0  840187  6001\n", 1)               # int off by one
check("400\t405.177\n", "400\t405.178\n", 0)                      # last-digit float
check("400\t405.9\n", "400\t405.178\n", 1)                        # real float error
check("RAFAEL --> Distance=1.2345\n", "RAFAEL --> Distance=1.2345\n", 0)
check("MICHAEL --> Distance=1.2345\n", "RAFAEL --> Distance=1.2345\n", 1)  # text exact
check("1.0e-09\n", "1.1e-09\n", 0)                                # near zero: atol
check("1 2 3\n", "1 2\n", 1)                                      # length mismatch
check(b"\x80\x81\x82", b"\x80\x81\x82", 0)                        # binary exact
check(b"\x80\x81\x82", b"\x80\x81\x83", 1)
print("compare_results self-check passed")
