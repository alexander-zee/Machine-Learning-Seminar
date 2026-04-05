"""
Cross-sections for AP-trees: (LME, feat1, feat2) with feat1 < feat2 in FEATS_LIST order.

The folder name is ``LME_<feat1>_<feat2>`` (same convention as ``step2_tree_portfolios``).

``FEATS_LIST`` is imported by ``run_part2`` and ``pick_best_lambda`` (single source of truth).

With **nine** non-LME characteristics in the panel (including ``IdioVol`` from ``svar``),
there are C(9,2) = **36** distinct triplets, matching Table B.1 style enumeration.
"""

from __future__ import annotations

from itertools import combinations

FEATS_LIST = [
    "LME",
    "BEME",
    "r12_2",
    "OP",
    "Investment",
    "ST_Rev",
    "LT_Rev",
    "AC",
    "LTurnover",
    "IdioVol",
]


def all_triplet_pairs() -> list[tuple[str, str]]:
    """
    All unordered pairs of secondary characteristics (excluding LME).

    Indices (i, j) satisfy 1 <= i < j < len(FEATS_LIST), matching AP_Pruning sub_dir names.
    """
    feats = FEATS_LIST
    return [(feats[i], feats[j]) for i, j in combinations(range(1, len(feats)), 2)]


def triplet_subdir_name(feat1: str, feat2: str) -> str:
    """``LME_feat1_feat2`` with indices resolved like Part 2 (order follows FEATS_LIST)."""
    i1 = FEATS_LIST.index(feat1)
    i2 = FEATS_LIST.index(feat2)
    if i1 == 0 or i2 == 0:
        raise ValueError("feat1 and feat2 must not be LME")
    if i1 > i2:
        i1, i2 = i2, i1
    return "_".join(["LME", FEATS_LIST[i1], FEATS_LIST[i2]])


def n_cross_sections() -> int:
    return len(all_triplet_pairs())
