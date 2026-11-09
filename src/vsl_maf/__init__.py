import vsl_core as _vsl_core

from .adapter import MAFAdapter
from .integration import VSLFunctionMiddleware, VSLGateDenied, tool_arguments

__version__ = "0.2.0"

# Gate-decision records (VerbaLedger.write_gate_decision) and the 2026.10 Gamma
# rule arrived in vsl-core 0.2.0; this package's docs and examples rely on both.
_MIN_VSL_CORE = (0, 2, 0)
_found = tuple(int(part) for part in _vsl_core.__version__.split(".")[:3])
if _found < _MIN_VSL_CORE:
    raise ImportError(
        f"vsl-maf {__version__} needs vsl-core >= 0.2.0, found {_vsl_core.__version__}. "
        "Upgrade: pip install --upgrade git+https://github.com/supersemantics/VSL-Core.git"
    )

__all__ = [
    "MAFAdapter",
    "VSLFunctionMiddleware",
    "VSLGateDenied",
    "tool_arguments",
]
