"""
Stub out heavy ML dependencies so unit tests run without the full install.
Only needed in environments where the optional heavy deps aren't installed.
"""
import sys
import types
from unittest.mock import MagicMock


def _ensure_stub(name: str) -> types.ModuleType:
    """Create a stub module (and all parent packages) in sys.modules."""
    parts = name.split(".")
    parent: types.ModuleType | None = None
    for i, part in enumerate(parts):
        full = ".".join(parts[: i + 1])
        if full not in sys.modules:
            mod = types.ModuleType(full)
            mod.__path__ = []  # mark as package so submodule imports work
            sys.modules[full] = mod
            if parent is not None:
                setattr(parent, part, mod)
        parent = sys.modules[full]
    return sys.modules[name]


def _force_stub(name: str) -> types.ModuleType:
    """Always (re)create the stub, ignoring any partial install."""
    mod = _ensure_stub(name)
    return mod


# ── transformers ─────────────────────────────────────────────────────────────
try:
    from transformers import pipeline as _tp  # noqa: F401
except (ImportError, Exception):
    _ensure_stub("transformers")
    sys.modules["transformers"].pipeline = MagicMock()  # type: ignore[attr-defined]

# ── sklearn ───────────────────────────────────────────────────────────────────
try:
    from sklearn.pipeline import Pipeline as _SP  # noqa: F401
except (ImportError, Exception):
    _ensure_stub("sklearn")
    _ensure_stub("sklearn.pipeline")
    sys.modules["sklearn.pipeline"].Pipeline = MagicMock  # type: ignore[attr-defined]

# ── sentence_transformers ─────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer as _ST  # noqa: F401
except (ImportError, Exception):
    _ensure_stub("sentence_transformers")
    sys.modules["sentence_transformers"].SentenceTransformer = MagicMock  # type: ignore[attr-defined]

# ── datasets ──────────────────────────────────────────────────────────────────
try:
    from datasets import DatasetDict as _DD  # noqa: F401
except (ImportError, Exception):
    _ensure_stub("datasets")
    for _attr in ("DatasetDict", "Dataset", "Split", "IterableDataset", "IterableDatasetDict", "load_dataset"):
        setattr(sys.modules["datasets"], _attr, MagicMock)

# ── joblib ────────────────────────────────────────────────────────────────────
try:
    import joblib as _jl  # noqa: F401
except ImportError:
    _ensure_stub("joblib")
    sys.modules["joblib"].load = MagicMock()  # type: ignore[attr-defined]
