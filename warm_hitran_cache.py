#!/usr/bin/env python
"""
HITRAN Cache Warm-Up

Empties hitran_local/ and refetches only cube_explorer.line_catalog's
COMMON_HITRAN_QUERIES, so the cache holds one clean, known table per query
instead of accumulated fragments from ad hoc past sessions. Needs internet;
run it before going offline (a meeting, a demo) so those exact queries then
work with no network at all -- fetch_hitran_data() caches by exact
(molecule, isotopologue, wavelength range), so re-run this whenever
COMMON_HITRAN_QUERIES changes.
"""

import os
import shutil
import sys

from cube_explorer.line_catalog import (
    COMMON_HITRAN_QUERIES, fetch_hitran_data, compute_hitran_table_name,
)

CACHE_DIR = "hitran_local"

print("\n" + "=" * 70)
print(" HITRAN CACHE WARM-UP")
print("=" * 70)

if os.path.isdir(CACHE_DIR):
    print(f"[INFO] Clearing {CACHE_DIR}/ ...")
    shutil.rmtree(CACHE_DIR)
os.makedirs(CACHE_DIR, exist_ok=True)

failures = []
for query in COMMON_HITRAN_QUERIES:
    print(f"\n--- {query['name']} ---")
    rows, _ = fetch_hitran_data(
        query["wavelength_min"], query["wavelength_max"],
        molecule=query["molecule"], isotopologue=query["isotopologue"],
        refresh_cache=True,
    )
    table_name = compute_hitran_table_name(
        query["wavelength_min"], query["wavelength_max"],
        query["molecule"], query["isotopologue"],
    )
    if rows and os.path.exists(os.path.join(CACHE_DIR, f"{table_name}.data")):
        print(f"[OK] Cached as {table_name} ({len(rows)} lines)")
    else:
        print(f"[ERROR] Failed to cache {query['name']}")
        failures.append(query["name"])

print("\n" + "=" * 70)
if failures:
    print(f"[ERROR] {len(failures)}/{len(COMMON_HITRAN_QUERIES)} quer{'y' if len(failures)==1 else 'ies'} failed: {', '.join(failures)}")
    sys.exit(1)
print(f"[OK] All {len(COMMON_HITRAN_QUERIES)} queries cached in {CACHE_DIR}/")
