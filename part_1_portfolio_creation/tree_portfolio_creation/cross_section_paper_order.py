"""
Fixed **36-row** cross-section order for thesis tables (IDs 1–36).

Matches the usual BPZ-style enumeration: Size (LME) plus every unordered pair among
nine non-LME characteristics, nested in characteristic order (all ``Value`` pairs first,
then ``Mom`` with those below it in the list, …).

Must use the same **set** of nine names as ``FEATS_LIST[1:]``.

**Order** matches BPZ-style / common thesis tables: among Value’s partners, **IdioVol**
appears **before** Turnover (``FEATS_LIST`` ends with ``LTurnover`` then ``IdioVol``).
Thesis row IDs 1–36 then align with captions like “Table 5.x” in the seminar.
"""

from __future__ import annotations

from part_1_portfolio_creation.tree_portfolio_creation.cross_section_triplets import (
    FEATS_LIST,
    triplet_subdir_name,
)

# Same multiset as FEATS_LIST[1:], but IdioVol before Turnover for paper row order
_sec = list(FEATS_LIST[1:])
_io, _to = _sec.index("IdioVol"), _sec.index("LTurnover")
_sec[_io], _sec[_to] = _sec[_to], _sec[_io]
THESIS_SECONDARIES_ORDER: list[str] = _sec
assert len(THESIS_SECONDARIES_ORDER) == 9
assert set(THESIS_SECONDARIES_ORDER) == set(FEATS_LIST[1:])

# LaTeX / thesis display names (column "1" is always Size in tables)
THESIS_CHAR_LABELS: dict[str, str] = {
    "LME": "Size",
    "BEME": "Value",
    "r12_2": "Mom",
    "OP": "OP",
    "Investment": "Inv",
    "ST_Rev": "SRev",
    "LT_Rev": "LRev",
    "AC": "Acc",
    "IdioVol": "IdioVol",
    "LTurnover": "Turnover",
}


def paper_table36_feature_pairs() -> list[tuple[str, str]]:
    """36 tuples (feat_a, feat_b) with indices increasing; folder = LME_feat_a_feat_b."""
    o = THESIS_SECONDARIES_ORDER
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(o):
        for b in o[i + 1 :]:
            pairs.append((a, b))
    assert len(pairs) == 36
    return pairs


def ap_subdir_for_pair(fa: str, fb: str) -> str:
    return triplet_subdir_name(fa, fb)


def rp_ap_subdir_for_pair(fa: str, fb: str) -> str:
    return "RP_" + triplet_subdir_name(fa, fb)


def tv_ap_subdir_for_pair(fa: str, fb: str) -> str:
    """
    Time-varying (kernel-weighted moments) pruning: same tree CSVs as AP, outputs ``TV_LME_*_*``.

    Run ``python run_all_tv_cross_sections.py`` (or ``TV_Pruning`` from ``run_part2``).
    """
    return "TV_" + triplet_subdir_name(fa, fb)
