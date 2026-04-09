"""
Cross-sections for AP/RP trees: (LME, feat1, feat2) with feat1 < feat2 in FEATS_LIST order.

Folder names match ``step2_tree_portfolios`` / ``step2_RP_tree_portfolios``:
``LME_<feat1>_<feat2>``.

``FEATS_LIST`` matches ``CHARACTERISTICS`` in ``step1_prepare_data`` on ``main``
(nine columns including LME; **no** IdioVol in the default main panel).
That yields C(8,2) = **28** triplets.
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
]


def all_triplet_pairs() -> list[tuple[str, str]]:
    """All unordered pairs of secondary characteristics (excluding LME)."""
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
