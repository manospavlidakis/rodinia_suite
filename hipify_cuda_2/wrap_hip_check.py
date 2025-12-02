#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

EXTS = {".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".hip", ".hpp", ".hh", ".hxx", ".h"}

# HIP calls that return hipError_t (nodiscard in many headers)
HIP_FUNCS = [
    "hipMalloc",
    "hipFree",
    "hipMemcpy",
    "hipMemcpyAsync",
    "hipMemset",
    "hipMemsetAsync",
    "hipMallocHost",
    "hipHostMalloc",
    "hipFreeHost",
    "hipStreamCreate",
    "hipStreamCreateWithFlags",
    "hipStreamDestroy",
    "hipDeviceSynchronize",
    "hipGetDeviceCount",
    "hipSetDevice",
    "hipGetDeviceProperties",
    "hipEventCreate",
    "hipEventCreateWithFlags",
    "hipEventRecord",
    "hipEventSynchronize",
    "hipEventDestroy",
    "hipGetLastError",
    "hipPeekAtLastError",
]

# Matches: optional leading whitespace + hipFunc( ... );
# We do a conservative "single statement on one line" transform.
CALL_RE = re.compile(
    r"""
    ^(?P<indent>\s*)                                # indentation
    (?P<prefix>.*?)                                 # anything before call (conservative filters applied later)
    (?P<func>""" + "|".join(map(re.escape, HIP_FUNCS)) + r""")
    \s*\(
      (?P<args>.*)
    \)\s*;
    (?P<suffix>\s*)$
    """,
    re.VERBOSE,
)

# Quick filters: don't rewrite if line is already HIP_CHECK, or is a macro definition, etc.
SKIP_PATTERNS = [
    re.compile(r"\bHIP_CHECK\s*\("),
    re.compile(r"^\s*#"),               # preprocessor lines
    re.compile(r"^\s*//"),              # full-line comment
]

# If the call is used in a control/expr context, don't rewrite automatically:
# e.g., err = hipMalloc(...);, if (hipMalloc(...)) ..., return hipMalloc(...);
UNSAFE_CONTEXT = re.compile(
    r"""
    (\=)|
    (\breturn\b)|
    (\bif\s*\()|
    (\bwhile\s*\()|
    (\bfor\s*\()|
    (\bcatch\s*\()|
    (\bthrow\b)
    """,
    re.VERBOSE,
)

def transform_line(line: str) -> str:
    if any(p.search(line) for p in SKIP_PATTERNS):
        return line

    m = CALL_RE.match(line)
    if not m:
        return line

    indent = m.group("indent")
    prefix = m.group("prefix")

    # If there's any meaningful prefix tokens, be conservative.
    # Allow only whitespace in prefix; otherwise it might be something like:
    #   (void)hipMalloc(...);  -> we *can* wrap, but keep it conservative.
    if prefix.strip() not in ("", "(void)"):
        # If it is something like "(void)" prefix, we can still wrap.
        return line

    # If used in an unsafe context, skip.
    if UNSAFE_CONTEXT.search(line):
        return line

    func = m.group("func")
    args = m.group("args").strip()

    # Avoid wrapping HIP_CHECK(hipX(...)); if someone wrote it oddly
    if "HIP_CHECK" in line:
        return line

    return f"{indent}HIP_CHECK({func}({args}));\n"

def process_file(path: Path, write: bool) -> tuple[bool, int]:
    original = path.read_text(errors="ignore").splitlines(keepends=True)
    changed = False
    out = []
    count = 0
    for line in original:
        new_line = transform_line(line)
        if new_line != line:
            changed = True
            count += 1
        out.append(new_line)

    if changed and write:
        path.write_text("".join(out))
    return changed, count

def main():
    ap = argparse.ArgumentParser(description="Wrap standalone HIP runtime calls with HIP_CHECK(...)")
    ap.add_argument("root", help="Root directory to process")
    ap.add_argument("--write", action="store_true", help="Actually modify files (otherwise dry-run)")
    ap.add_argument("--ext", action="append", default=None, help="Extra file extension to include (repeatable)")
    args = ap.parse_args()

    exts = set(EXTS)
    if args.ext:
        exts |= {e if e.startswith(".") else "." + e for e in args.ext}

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"ERROR: path not found: {root}")

    total_files = 0
    total_changes = 0
    touched_files = 0

    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() not in exts:
                continue
            total_files += 1
            try:
                changed, n = process_file(p, args.write)
            except Exception as e:
                print(f"ERROR reading {p}: {e}")
                continue
            if changed:
                touched_files += 1
                total_changes += n
                print(f"{'MOD' if args.write else 'DRY'} {p}  (+{n})")

    print(f"\nScanned {total_files} files. {'Modified' if args.write else 'Would modify'} {touched_files} files, {total_changes} lines.")

if __name__ == "__main__":
    main()
