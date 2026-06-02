"""Add uns/commit_hash to an h5ad in-place (DataModule requires it).
Usage: python bench/patch_uns.py <file.h5ad> <tag>
"""
import sys

import h5py
import numpy as np

fn, tag = sys.argv[1], sys.argv[2]
with h5py.File(fn, "r+") as f:
    g = f.require_group("uns")
    if "commit_hash" in g:
        del g["commit_hash"]
    g.create_dataset("commit_hash", data=np.bytes_(tag))
print(f"patched {fn} uns/commit_hash={tag}")
