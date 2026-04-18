"""
rerun_uniform_full_fit.py
-------------------------
Reruns uniform_full_fit for all 36 cross-sections to regenerate
full_fit_detail_k{k}.csv with portfolio weight columns included.

Run once, then delete this file.

Usage
-----
    python rerun_uniform_full_fit.py
"""

from itertools import combinations
from pathlib import Path
import traceback

from part_3_metrics_collection.uniform_full_fit import uniform_full_fit

CHARACTERISTICS = [
    'BEME', 'r12_2', 'OP', 'Investment',
    'ST_Rev', 'LT_Rev', 'AC', 'LTurnover', 'IdioVol',
]

PAIRS  = list(combinations(CHARACTERISTICS, 2))
PORT_N = 10

done, skipped, failed = 0, 0, 0

for feat1, feat2 in PAIRS:
    cs = f'LME_{feat1}_{feat2}'
    try:
        result = uniform_full_fit(feat1, feat2, k=PORT_N)
        print(f"  [OK]     {cs:<35}  SR={result['test_SR']:+.4f}")
        done += 1
    except FileNotFoundError:
        print(f"  [SKIP]   {cs:<35}  Selected_Ports not found — run pipeline first")
        skipped += 1
    except Exception as e:
        print(f"  [ERROR]  {cs:<35}  {e}")
        traceback.print_exc()
        failed += 1

print(f"\nDone: {done}  Skipped: {skipped}  Failed: {failed}")