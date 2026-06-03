#!/usr/bin/env python3
"""Aggregate a samply profile into self-time per source line.

Reads a samply-recorded Firefox-profiler JSON, buckets every leaf sample by its
binary offset, batch-symbolizes the distinct offsets with `atos` against a dSYM,
and prints self-time rolled up by symbol and by `file:line`.

Usage:
    analyze_profile.py <profile.json> <dSYM-bundle-or-DWARF> [binary-name]

`binary-name` defaults to "profile_dijkstra" and must match the recorded binary
(used both to locate the DWARF inside a .dSYM bundle and to pick its samples out
of the system libraries).
"""

import collections
import json
import os
import re
import subprocess
import sys


def text_vmaddr(macho: str) -> int:
    """Read the __TEXT segment's vmaddr so offsets can be turned into static
    addresses for `atos` (normally 0x100000000 for arm64 executables)."""
    out = subprocess.run(
        ["otool", "-l", macho], capture_output=True, text=True
    ).stdout
    in_text = False
    for line in out.splitlines():
        s = line.split()
        if "segname" in s and "__TEXT" in s:
            in_text = True
        elif in_text and s and s[0] == "vmaddr":
            return int(s[1], 16)
    return 0x100000000


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)

    profile_path = sys.argv[1]
    dsym = sys.argv[2]
    binname = sys.argv[3] if len(sys.argv) > 3 else "profile_dijkstra"

    dwarf = dsym
    if os.path.isdir(dsym):
        dwarf = os.path.join(dsym, "Contents", "Resources", "DWARF", binname)
    if not os.path.exists(dwarf):
        sys.exit(f"DWARF not found: {dwarf}")

    base = text_vmaddr(dwarf)

    j = json.load(open(profile_path))
    libs = j["libs"]
    our_libs = {i for i, l in enumerate(libs) if l.get("name") == binname}
    if not our_libs:
        sys.exit(f"no lib named {binname!r} in {profile_path}")

    th = j["threads"][0]
    strings = th["stringArray"]
    ft, fn, rt = th["frameTable"], th["funcTable"], th["resourceTable"]
    sk_frame = th["stackTable"]["frame"]
    samples = th["samples"]
    stacks = samples["stack"]
    weights = samples.get("weight") or [1] * len(stacks)
    fr_func, fr_addr = ft["func"], ft["address"]
    fn_res, rt_lib = fn["resource"], rt["lib"]

    def lib_of_func(fu: int):
        r = fn_res[fu]
        return rt_lib[r] if (r is not None and r >= 0) else None

    our_off = collections.Counter()  # binary offset -> samples
    other = collections.Counter()  # (lib, func) -> samples (system libs)
    total = 0
    for si, w in zip(stacks, weights):
        if si is None:
            continue
        w = w or 1
        total += w
        fi = sk_frame[si]
        fu = fr_func[fi]
        lib = lib_of_func(fu)
        if lib in our_libs:
            our_off[fr_addr[fi]] += w
        else:
            name = libs[lib]["name"] if lib is not None else "?"
            other[(name, strings[fn["name"][fu]])] += w

    if total == 0:
        sys.exit("no samples in profile")

    in_bin = sum(our_off.values())
    print(f"total leaf samples = {total}")
    print(f"in {binname} binary = {in_bin} ({100 * in_bin / total:.1f}%)")
    print(f"in other libs       = {total - in_bin} ({100 * (total - in_bin) / total:.1f}%)\n")

    print("=== TOP NON-BINARY (system) SELF TIME ===")
    for (lib, name), c in other.most_common(6):
        print(f"{100 * c / total:6.2f}%  {c:7d}  {lib}: {name[:50]}")

    # Batch-symbolize the distinct offsets (static address = vmaddr + offset).
    offs = sorted(our_off)
    args = ["atos", "-o", dwarf, "-arch", "arm64"] + [hex(base + o) for o in offs]
    out = subprocess.run(args, capture_output=True, text=True).stdout.splitlines()

    def parse(line: str):
        m = re.search(r"\(([^()]+\.[a-z]+):(\d+)\)\s*$", line)
        file_line = f"{m.group(1)}:{m.group(2)}" if m else None
        sym = line.split(" (in ")[0]
        if "find_path_in_heatmap" in sym:
            sym = "find_path_in_heatmap"
        else:
            sym = re.sub(r"::h[0-9a-f]+$", "", sym)[:46]
        return sym, file_line

    by_line = collections.Counter()
    by_sym = collections.Counter()
    for o, line in zip(offs, out):
        c = our_off[o]
        sym, file_line = parse(line)
        by_sym[sym] += c
        by_line[(file_line, sym)] += c

    print("\n=== SELF TIME BY SYMBOL (within binary) ===")
    for sym, c in by_sym.most_common(8):
        print(f"{100 * c / total:6.2f}%  {c:7d}  {sym}")

    print("\n=== HOT LINES (.rs) — self time ===")
    rows = [(fl, sym, c) for (fl, sym), c in by_line.items() if fl and ".rs:" in fl]
    for fl, sym, c in sorted(rows, key=lambda r: -r[2])[:18]:
        print(f"{100 * c / total:6.2f}%  {c:7d}  {fl:26s} [{sym}]")


if __name__ == "__main__":
    main()
