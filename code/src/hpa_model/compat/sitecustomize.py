"""Local runtime compatibility fixes for the lab environment."""

from __future__ import annotations

import os
import sys
import types


# Old Streamlit builds in the existing environment need the Python protobuf backend.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def _patch_altair_v4_alias() -> None:
    try:
        import altair as alt
    except Exception:
        return

    if "altair.vegalite.v4.api" in sys.modules:
        return

    vegalite_module = sys.modules.setdefault("altair.vegalite", types.ModuleType("altair.vegalite"))
    v4_module = sys.modules.setdefault("altair.vegalite.v4", types.ModuleType("altair.vegalite.v4"))
    api_module = types.ModuleType("altair.vegalite.v4.api")
    api_module.Chart = alt.Chart

    setattr(vegalite_module, "v4", v4_module)
    setattr(v4_module, "api", api_module)
    sys.modules["altair.vegalite.v4.api"] = api_module


_patch_altair_v4_alias()
