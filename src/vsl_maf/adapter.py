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

import inspect
import warnings
from typing import Any, Callable

from vsl_core.conformance.protocol import CompiledGate
from vsl_core.constructs import Fallback, Invariant, PreNode
from vsl_core.exceptions import AutomationDeniedException, InvariantViolation
from vsl_core.metrics import F2Modification

_DEFAULT_FALLBACK = Fallback()

# Source-text markers for hosted, black-box model APIs. A gate whose
# monitor/rule reaches one of these cannot honestly claim F2Modification.FULL
# -- FULL requires direct access to the energy function itself (logits,
# weights, activations), which a hosted API call structurally cannot expose.
# The honest ceiling for a hosted-API-backed check is PARTIAL (MEDIUM), per
# vsl-core's own README example. This is a heuristic, not a proof: it scans
# source text -- the monitor/rule's OWN body (inspect.getsource(fn), not its
# containing module's entire source), plus one level of indirection through
# whatever names it directly calls -- catching the common "the actual API
# call lives in a helper module" shape. It can still miss deeper indirection
# and cannot confirm a genuine FULL claim is correct either.
_HOSTED_API_MARKERS = (
    "openai",
    "anthropic",
    "google.genai",
    "google.generativeai",
    "cohere",
    "mistralai",
    # MAF's own hosted-model clients (FoundryChatClient, OpenAIChatClient,
    # AzureOpenAIChatClient, ...) and Azure AI inference: a monitor that asks
    # another agent or a chat client to judge is a hosted-model call too.
    "chatclient",
    "azure.ai.inference",
)


def _reaches_hosted_api(fn: Callable[..., Any], _seen: set | None = None) -> bool:
    """Scoped to fn's OWN source (inspect.getsource(fn)), not its containing
    module's entire source. One level of indirection through directly-called
    helpers is also checked (same logic as vsl-langgraph's adapter.py).
    """
    if _seen is None:
        _seen = set()
    if fn in _seen:
        return False
    _seen.add(fn)

    if inspect.getmodule(fn) is None:
        return False
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return False
    # Case-insensitive: real code capitalises SDK class names (AsyncAnthropic,
    # AsyncOpenAI), which a case-sensitive check against lowercase markers misses.
    source_lower = source.lower()
    if any(marker in source_lower for marker in _HOSTED_API_MARKERS):
        return True

    code = getattr(fn, "__code__", None)
    globalns = getattr(fn, "__globals__", None)
    if code is None or globalns is None:
        return False
    for referenced_name in code.co_names:
        candidate = globalns.get(referenced_name)
        if callable(candidate) and candidate is not fn:
            if _reaches_hosted_api(candidate, _seen):
                return True
    return False


def _warn_if_full_claim_reaches_hosted_api(
    name: str, assurance_basis: Any, fn: Callable[..., Any]
) -> None:
    if assurance_basis.f2_modification != F2Modification.FULL:
        return
    if not _reaches_hosted_api(fn):
        return
    warnings.warn(
        f"'{name}' claims F2Modification.FULL, but its monitor/rule appears to "
        "call a hosted model API (openai/anthropic/google.genai/cohere/mistralai "
        "detected in its source). A hosted, black-box API call cannot directly "
        "modify the energy function -- FULL is not achievable this way. "
        "The honest ceiling is F2Modification.PARTIAL (MEDIUM assurance), per "
        "vsl-core's own README F1/F2 example. This is a heuristic check, not "
        "proof -- it can miss indirect calls, and passing it does not confirm "
        "FULL is correct either.",
        stacklevel=3,
    )


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
        _warn_if_full_claim_reaches_hosted_api(
            f"PreNode '{pre_node.name}'", pre_node.assurance_basis, pre_node.monitor
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
        _warn_if_full_claim_reaches_hosted_api(
            f"Invariant '{invariant.name}'", invariant.assurance_basis, invariant.rule
        )
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
