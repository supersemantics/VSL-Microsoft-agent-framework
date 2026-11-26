"""MAFAdapter: a VSLAdapter conformant with vsl-core's conformance contract,
intended for use with the Microsoft Agent Framework (agent-framework).

MAF's FunctionMiddleware provides a natural pre-commitment hook: a middleware's
process() is called before any tool side effect fires. VSLFunctionMiddleware
(see integration.py) uses that hook to call a compiled gate, preserving F1
pre-commitment.

What this package adds on top (see integration.py) is VSLFunctionMiddleware,
a FunctionMiddleware subclass that calls a compiled gate before the tool runs
and raises AutomationDeniedException on denial, so the caller can intercept it
via a standard try/except around agent.run().

This module is framework-agnostic -- it never imports agent_framework.
The compiled gate it returns is the same async callable vsl-core's own
PlainPythonReferenceAdapter produces. The only MAF-specific logic lives in
integration.py (VSLFunctionMiddleware), not here.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

from vsl_core.conformance.protocol import CompiledGate
from vsl_core.constructs import Fallback, Invariant, PreNode
from vsl_core.exceptions import AutomationDeniedException, InvariantViolation

_DEFAULT_FALLBACK = Fallback()

class MAFAdapter:
    """VSLAdapter for the Microsoft Agent Framework (agent-framework).

    Compiles PreNode/Invariant into callable gates for use inside a
    VSLFunctionMiddleware registered on an Agent instance (see integration.py),
    or for direct inline calls before any guarded action runs.

    This class never imports agent_framework -- the compiled gate is the same
    framework-agnostic async callable vsl-core's own PlainPythonReferenceAdapter
    produces. VSLFunctionMiddleware (integration.py) provides the MAF-specific
    middleware hook.
    """

    def __init__(self, *, identity_key: str = "vsl-maf") -> None:
        """`identity_key` is recorded on every denial this adapter's gates raise;
        pass the agent's own identity so a denial says which agent it stopped."""
        if not identity_key:
            raise ValueError("identity_key must be a non-empty string.")
        self.identity_key = identity_key

    def compile_pre_node(self, pre_node: PreNode) -> CompiledGate:
        if pre_node.fallback != _DEFAULT_FALLBACK:
            warnings.warn(
                f"PreNode '{pre_node.name}' sets a non-default Fallback "
                f"({pre_node.fallback!r}), but VSLFunctionMiddleware does not consume "
                "Fallback -- a denial raises once, immediately, with no retry loop. "
                "Implement retry behavior yourself around the compiled gate if needed.",
                stacklevel=2,
            )
        identity_key = self.identity_key

        async def gate(candidate_input: Any) -> None:
            estimate = await pre_node.monitor(candidate_input)
            if not estimate.sufficient(pre_node.gamma_threshold):
                raise AutomationDeniedException(
                    reason=f"Pre-node '{pre_node.name}' insufficient Gamma",
                    identity_key=identity_key,
                )

        return gate

    def compile_invariant(self, invariant: Invariant) -> CompiledGate:
        identity_key = self.identity_key

        async def gate(candidate_input: Any) -> None:
            holds = await invariant.rule(candidate_input)
            if not holds:
                reason = f"Invariant '{invariant.name}' violated"
                terminal_state_name = (
                    invariant.on_violation.name if invariant.on_violation is not None else None
                )
                if terminal_state_name is not None:
                    reason += f" -- entering terminal state '{terminal_state_name}'"
                raise InvariantViolation(
                    reason=reason,
                    identity_key=identity_key,
                    invariant_name=invariant.name,
                    terminal_state_name=terminal_state_name,
                )

        return gate
