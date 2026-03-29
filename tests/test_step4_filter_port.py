# tests/test_step4_filter.py
"""
Test Step 4 filter against R's output.

Input  : R's step3 output files (level_all_*.csv)
Expected: R's step4 output files (level_all_*_filtered.csv)

We feed R's step3 files into our step4, then check our filtered output
matches R's filtered output. Column names from R have a leading 'X' on
digit-starting names (R's read.table quirk), so we strip it before comparing.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TEST_DIR = Path(__file__).parent
ROOT_DIR = TEST_DIR.parent
R_DIR    = ROOT_DIR / "paper_data" / "tree_portfolio_quantile" / "LME_OP_Investment"

sys.path.insert(0, str(ROOT_DIR))

FEAT1 = "OP"
FEAT2 = "Investment"
FEATS = ["LME", FEAT1, FEAT2]

STEMS = (
    ["excess_combined"]
    + [f"{f}_{s}" for f in FEATS for s in ("min", "max")]
)


def strip_x(cols):
    """Strip R's leading 'X' from column names that start with a digit."""
    return [c[1:] if c.startswith("X") and c[1:2].isdigit() else c for c in cols]


@pytest.fixture(scope="module")
def filtered_dir():
    """Copy R's step3 CSVs into a temp dir, run our step4, yield results dir."""
    from part_1_portfolio_creation.tree_portfolio_creation.step4_filter_portfolios import filter_tree_ports

    tmpdir      = Path(tempfile.mkdtemp())
    triplet_dir = tmpdir / "LME_OP_Investment"
    triplet_dir.mkdir()

    for stem in STEMS:
        src = R_DIR / f"level_all_{stem}.csv"
        if not src.exists():
            pytest.skip(f"R step3 file not found: {src}")
        shutil.copy(src, triplet_dir / f"level_all_{stem}.csv")

    filter_tree_ports(feat1=FEAT1, feat2=FEAT2, tree_out=tmpdir)

    yield triplet_dir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.parametrize("stem", STEMS)
def test_filtered_matches_r(filtered_dir, stem):
    our_file = filtered_dir / f"level_all_{stem}_filtered.csv"
    r_file   = R_DIR        / f"level_all_{stem}_filtered.csv"

    assert our_file.exists(), f"Our step4 did not produce: {our_file.name}"
    assert r_file.exists(),   f"R filtered file not found: {r_file}"

    our = pd.read_csv(our_file)
    r   = pd.read_csv(r_file)

    assert list(our.columns) == strip_x(r.columns), f"{stem}: column names differ"
    np.testing.assert_allclose(
        our.values, r.values, rtol=1e-5, atol=1e-8,
        err_msg=f"{stem}: values do not match R"
    )