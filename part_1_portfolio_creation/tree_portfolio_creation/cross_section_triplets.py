"""
Cross-sections for AP/RP trees: (LME, feat1, feat2) with feat1 < feat2 in FEATS_LIST order.

Folder names match ``step2_tree_portfolios`` / ``step2_RP_tree_portfolios``:
``LME_<feat1>_<feat2>``.

``FEATS_LIST`` matches ``CHARACTERISTICS`` in ``step1_prepare_data`` (ten columns
including LME and IdioVol). Secondary features excluding LME give C(9,2) = **36** triplet pairs.
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
    """All unordered pairs of secondary characteristics (excluding LME)."""
    feats = FEATS_LIST
    return [(feats[i], feats[j]) for i, j in combinations(range(1, len(feats)), 2)]


def all_triplet_pairs_excluding_secondary(exclude: str) -> list[tuple[str, str]]:
    """
    Secondary-feature pairs that **do not** include ``exclude``.

    Example: ``exclude='IdioVol'`` → C(8,2) = **28** pairs (matches thesis table
    ``--rows no-idiovol``). The full grid without filtering is **36** pairs.
    """
    if exclude not in FEATS_LIST:
        raise ValueError(f"exclude must be one of {FEATS_LIST}, got {exclude!r}")
    if exclude == "LME":
        raise ValueError("exclude must be a secondary characteristic, not LME")
    feats = [f for f in FEATS_LIST[1:] if f != exclude]
    return [(feats[i], feats[j]) for i, j in combinations(range(len(feats)), 2)]


def canonical_feat_pair(feat1: str, feat2: str) -> tuple[str, str]:
    """Return (feat1, feat2) sorted by FEATS_LIST index (excl. LME); matches ``LME_*`` folder order."""
    i1, i2 = FEATS_LIST.index(feat1), FEATS_LIST.index(feat2)
    if i1 == 0 or i2 == 0:
        raise ValueError("feat1 and feat2 must not be LME")
    if i1 > i2:
        return FEATS_LIST[i2], FEATS_LIST[i1]
    return feat1, feat2


def triplet_subdir_name(feat1: str, feat2: str) -> str:
    """``LME_feat1_feat2`` with indices resolved like Part 2 (order follows FEATS_LIST)."""
    a, b = canonical_feat_pair(feat1, feat2)
    return "_".join(["LME", a, b])


def n_cross_sections() -> int:
    return len(all_triplet_pairs())
