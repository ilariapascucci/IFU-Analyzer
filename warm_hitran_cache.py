#!/usr/bin/env python
"""
HITRAN Cache Warm-Up

Build hitran_local/, the committed offline starter pack, from
cube_explorer.line_catalog.COMMON_HITRAN_QUERIES (one table per molecule and
isotopologue in _ALL_ISOTOPOLOGUES_BY_MOLECULE, over 1-30um). This needs internet.

    python warm_hitran_cache.py              # wipe hitran_local/, then refetch all
    python warm_hitran_cache.py --additive   # keep it, fetch only what is missing

Run this script again after you change _ALL_ISOTOPOLOGUES_BY_MOLECULE. On-demand
runtime fetches go to hitran_cache/. This script does not change them.
"""

import os
import shutil
import sys

from cube_explorer.line_catalog import (
    COMMON_HITRAN_QUERIES, fetch_hitran_data, compute_hitran_table_name,
    BUNDLED_HITRAN_DIR,
)

CACHE_DIR = BUNDLED_HITRAN_DIR
ADDITIVE = "--additive" in sys.argv[1:]

print("\n" + "=" * 70)
print(f" HITRAN CACHE WARM-UP{'  (additive)' if ADDITIVE else ''}")
print("=" * 70)

if not ADDITIVE and os.path.isdir(CACHE_DIR):
    print(f"[INFO] Clearing {CACHE_DIR}/ ...")
    shutil.rmtree(CACHE_DIR)
os.makedirs(CACHE_DIR, exist_ok=True)

failures = []
skipped = 0
for query in COMMON_HITRAN_QUERIES:
    table_name = compute_hitran_table_name(
        query["wavelength_min"], query["wavelength_max"],
        query["molecule"], query["isotopologue"],
    )
    data_path = os.path.join(CACHE_DIR, f"{table_name}.data")

    if ADDITIVE and os.path.exists(data_path):
        print(f"\n--- {query['name']} --- already cached, skipping")
        skipped += 1
        continue

    print(f"\n--- {query['name']} ---")
    rows, meta = fetch_hitran_data(
        query["wavelength_min"], query["wavelength_max"],
        molecule=query["molecule"], isotopologue=query["isotopologue"],
        refresh_cache=not ADDITIVE,
        write_dir=CACHE_DIR,
    )
    if rows and os.path.exists(data_path):
        print(f"[OK] Cached as {table_name} ({len(rows)} lines)")
    else:
        msg = meta.get("message", "no lines returned and no error reported")
        print(f"[ERROR] Failed to cache {query['name']}: {msg}")
        failures.append(query["name"])

print("\n" + "=" * 70)
if failures:
    n = len(failures)
    print(f"[ERROR] {n} quer{'y' if n == 1 else 'ies'} failed: {', '.join(failures)}")
    sys.exit(1)
cached = len(COMMON_HITRAN_QUERIES) - skipped
print(f"[OK] {cached} table(s) cached"
      + (f", {skipped} already present" if skipped else "")
      + f" in {CACHE_DIR}/")
