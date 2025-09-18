#!/usr/bin/env python3
"""Extract a deterministic subset of N samples from a processed .pt file.

Usage:
    python scripts/extract_pubchem_subset.py /path/to/processed_pubchem.pt /path/to/output_subset.pt --n 1000
"""
import argparse
import torch
import random
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str, help='Path to source .pt file')
    parser.add_argument('out', type=str, help='Path to output subset .pt file')
    parser.add_argument('--n', type=int, default=1000, help='Number of samples to extract')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for deterministic sampling')
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    assert src.exists(), f"Source file not found: {src}"

    print(f"Loading data from: {src}")
    data = torch.load(src)

    if isinstance(data, dict) and 'samples' in data:
        # Some .pt files store a dict with a 'samples' key
        data_list = data['samples']
    else:
        data_list = list(data)

    total = len(data_list)
    print(f"Total samples in source: {total}")

    n = min(args.n, total)
    random.seed(args.seed)
    indices = list(range(total))
    random.shuffle(indices)
    sel = indices[:n]
    sel.sort()

    subset = [data_list[i] for i in sel]

    # Keep original structure if it was a dict
    if isinstance(data, dict) and 'samples' in data:
        out_obj = dict(data)
        out_obj['samples'] = subset
    else:
        out_obj = subset

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving subset ({n}) to: {out}")
    torch.save(out_obj, out)
    print("Done.")

if __name__ == '__main__':
    main()
