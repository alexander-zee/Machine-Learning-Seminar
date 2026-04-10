"""
Parse AP / RP tree portfolio column names for split depth and tree id.

Handles names like ``00.11111`` (Python) or ``X1111.11111`` (R export).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ParsedColumn:
    raw: str
    tree_id: str
    node_path: str
    depth: int
    """AP convention: depth = len(node_path) - 1 for inner nodes."""


def parse_portfolio_column(name: str) -> ParsedColumn:
    s = str(name).strip().strip('"')
    s = re.sub(r"^X(?=[0-9])", "", s)  # R-style X prefix before digits
    if "." not in s:
        return ParsedColumn(raw=name, tree_id=s, node_path="", depth=-1)
    tree_id, node_path = s.split(".", 1)
    depth = max(0, len(node_path) - 1) if node_path else 0
    return ParsedColumn(raw=name, tree_id=tree_id, node_path=node_path, depth=depth)


def first_split_direction(node_path: str) -> str | None:
    """First branching digit after the root character, if any."""
    if len(node_path) < 2:
        return None
    return node_path[1] if node_path[1] in ("1", "2") else None


def splits_summary(portfolio_columns: list[str]) -> list[dict]:
    rows = []
    for c in portfolio_columns:
        p = parse_portfolio_column(c)
        rows.append(
            {
                "column": c,
                "tree_id": p.tree_id,
                "node_path": p.node_path,
                "depth": p.depth,
                "first_split": first_split_direction(p.node_path),
            }
        )
    return rows
