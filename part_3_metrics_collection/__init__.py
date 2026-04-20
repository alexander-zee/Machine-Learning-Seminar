"""
`part_3_metrics_collection` package marker.

This repository historically relied on implicit namespace packages (no `__init__.py`),
which works for `import part_3_metrics_collection.foo`, but breaks patterns that
traverse attributes on the parent package (e.g. some `unittest.mock.patch` targets).

Keeping this file makes submodule access like::

    import part_3_metrics_collection.tv_extension_summary_table as tv

and patching via ``part_3_metrics_collection.tv_extension_summary_table`` reliable.
"""

from __future__ import annotations

__all__: list[str] = []
