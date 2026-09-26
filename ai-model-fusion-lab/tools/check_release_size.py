"""Release size check (M3 spec §2, §34).

The release zip must stay under the 120 MB HARD cap, and must never contain
real model weights (safetensors/bin/h5/onnx/ckpt) or HF caches. Prints
PASS/FAIL and exits 0/1 accordingly.
"""
from __future__ import annotations

import os
import sys
import zipfile

LIMIT_MB = 120.0
WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".h5",
               ".msgpack", ".npz")  # npz = component tensor store, never bundled


def zip_stats(path: str):
    total = 0
    offenders = []
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            total += info.file_size
            name = info.filename.lower()
            if name.endswith(WEIGHT_EXTS) or "huggingface" in name:
                offenders.append((info.filename, info.file_size))
    return total, offenders


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = os.path.join(os.path.dirname(root), "fusionlab_data")
    zip_path = os.path.join(os.path.dirname(root), "ai-model-fusion-lab-release.zip")

    print("RELEASE SIZE CHECK (cap: %.0f MB hard)" % LIMIT_MB)
    print("-" * 60)
    ok = True

    entries = []
    if os.path.isdir(root):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if d not in ("__pycache__", ".git", "node_modules")]
            for f in filenames:
                p = os.path.join(dirpath, f)
                entries.append((p, os.path.getsize(p)))
    for name in ("registry.json", "tokenizer.json", "experiments.jsonl",
                 "lineage.json", "private_suites.json"):
        p = os.path.join(data, name)
        if os.path.exists(p):
            entries.append((p, os.path.getsize(p)))
    # small LOCAL-TRAINED synthetic model metadata+weights are allowed as
    # plain JSON/NPY (they are ours, Apache/MIT-clean, tiny); real third-party
    # weights are NOT. Check nothing bundled exceeds 2 MB per file.
    big = [(p, s) for p, s in entries if s > 2 * 1024 * 1024]
    total = sum(s for _, s in entries)

    for p, s in sorted(entries, key=lambda e: -e[1])[:10]:
        print(f"  {s/1024:10.1f} KB  {os.path.relpath(p, os.path.dirname(root))}")
    print("-" * 60)
    print(f"  release content total: {total/1024/1024:.2f} MB "
          f"({len(entries)} files)")

    if total > LIMIT_MB * 1024 * 1024:
        print(f"FAIL: release content {total/1024/1024:.2f} MB exceeds "
              f"{LIMIT_MB:.0f} MB hard cap")
        ok = False
    else:
        print(f"PASS: {total/1024/1024:.2f} MB <= {LIMIT_MB:.0f} MB hard cap")

    if big:
        print("FAIL: files above 2 MB per-file limit (possible real weights):")
        for p, s in big[:10]:
            print(f"   {s/1024/1024:.2f} MB  {os.path.relpath(p, os.path.dirname(root))}")
        ok = False
    else:
        print("PASS: no file above the 2 MB per-file limit (no real weights)")

    if os.path.exists(zip_path):
        total_z, off = zip_stats(zip_path)
        mb = total_z / 1024 / 1024
        line = f"existing zip {os.path.basename(zip_path)}: {mb:.2f} MB"
        if mb > LIMIT_MB or off:
            print(f"FAIL: {line}" + (f" — weight-like entries: {off[:5]}" if off else ""))
            ok = False
        else:
            print(f"PASS: {line}")
    else:
        print(f"NOTE: no existing zip at {zip_path} (built at release time)")

    print("-" * 60)
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
