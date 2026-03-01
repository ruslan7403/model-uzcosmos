#!/usr/bin/env python3
"""Extract model_snapshot.tar.gz (if needed) and print the sign classes from gallery.json."""
import json
import os
import subprocess
import sys

# Run from repo root or scripts/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARBALL = os.path.join(ROOT, "model_snapshot.tar.gz")
GALLERY_JSON = os.path.join(ROOT, "gallery.json")

def main():
    os.chdir(ROOT)
    if not os.path.exists(TARBALL):
        print(f"Not found: {TARBALL}", file=sys.stderr)
        sys.exit(1)
    if os.path.getsize(TARBALL) < 1000:
        print(f"File too small ({os.path.getsize(TARBALL)} bytes). Replace with the real model tarball.", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(GALLERY_JSON):
        print("Extracting tarball...")
        subprocess.run(["tar", "-xzvf", "model_snapshot.tar.gz"], check=True, capture_output=True)
    if not os.path.exists(GALLERY_JSON):
        print("gallery.json not in tarball.", file=sys.stderr)
        sys.exit(1)
    with open(GALLERY_JSON) as f:
        data = json.load(f)
    classes = list(data.get("classes", {}).keys())
    print("Sign classes in gallery:")
    for c in sorted(classes):
        print(f"  {c}")
    print(f"\nTotal: {len(classes)} classes")

if __name__ == "__main__":
    main()
