"""VSLFunctionMiddleware: MAF FunctionMiddleware that applies a compiled VSL gate
before a tool (function) call runs.

vsl-core does not intercept anything on its own: a compiled gate only ever
raises or returns, on demand, when called. VSLFunctionMiddleware is that call
site, made reusable for MAF's middleware mechanism -- it calls the gate
immediately before the tool would run (preserving F1 pre-commitment: the gate
completes strictly before the guarded tool's side effect fires), and on denial
raises AutomationDeniedException so the caller can intercept it via a standard
try/except around agent.run() / agent.run_stream().

MAF FunctionMiddleware semantics:
  - context.function.name  -- the tool being called
  - context.arguments      -- the validated arguments dict
  - call_next()            -- continues to the next middleware or the actual tool
                              NOTE: call_next() takes NO arguments in MAF;
                              middleware mutates the shared context in-place.

Registration (pass as element of the `middleware` list when creating an Agent):

    async with Agent(
        client=FoundryChatClient(...),
        tools=approve_purchase,
        middleware=[
            VSLFunctionMiddleware(confidence_gate, candidate_input_fn=lambda ctx: dict(ctx.arguments)),
            VSLFunctionMiddleware(cap_gate,        candidate_input_fn=lambda ctx: dict(ctx.arguments)),
        ],
    ) as agent:
        result = await agent.run("Approve $250, confidence=5.0")

Chaining: MAF runs middleware in registration order for each tool call.
Confidence gate runs first; if it raises, the cap gate never fires and the
tool never runs -- F1 pre-commitment holds for the entire chain.

Scoping to a specific tool name:

    VSLFunctionMiddleware(gate, function_name="approve_purchase")

`agent_framework` must be importable at runtime. If it is not installed, this
module raises ImportError on import, matching the package's declared dependency.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from vsl_core.conformance.protocol import CompiledGate
from vsl_core.exceptions import AutomationDeniedException  # noqa: F401 -- re-exported for callers

# MAF FunctionMiddleware is the base class for all function-level middleware.
# It defines the abstract process(context, call_next) method that MAF calls
# before each tool invocation.  Importing it here means agent-framework must
# be installed, which matches the declared pyproject.toml dependency.
try:
    from agent_framework import FunctionMiddleware as _FunctionMiddleware
    _MIDDLEWARE_BASE: type = _FunctionMiddleware
except ImportError:  # allow tests to import this module without agent-framework
    _MIDDLEWARE_BASE = object  # type: ignore[assignment]


class VSLFunctionMiddleware(_MIDDLEWARE_BASE):  # type: ignore[misc]
    """MAF FunctionMiddleware that applies a VSL gate before a tool runs.

    On denial, AutomationDeniedException (or the more specific InvariantViolation)
    is raised, call_next() is never called, and the guarded tool never runs.
    F1 pre-commitment holds because process() raises strictly before call_next().

    `candidate_input_fn` extracts what the gate should evaluate from the
    FunctionInvocationContext; defaults to passing the whole context unchanged.
    For argument-based gates the typical extractor is:

        lambda ctx: dict(ctx.arguments)

    `function_name` optionally scopes the gate to a specific tool -- calls to
    other tools pass through unchanged.  None means the gate applies to every
    tool call this middleware instance receives.
    """

    def __init__(
        self,
        gate: CompiledGate,
        *,
        candidate_input_fn: Callable[[Any], Any] | None = None,
        function_name: str | None = None,
    ) -> None:
        super().__init__()
        self._gate = gate
        self._extract = candidate_input_fn or (lambda ctx: ctx)
        self._function_name = function_name

    async def process(
        self,
        context: Any,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        """Gate the tool call. MAF invokes this before the tool runs."""
        if self._function_name is not None:
            fn_name = getattr(getattr(context, "function", None), "name", None)
            if fn_name != self._function_name:
                await call_next()
                return

        # Gate runs here -- raises on denial (F1 pre-commitment: call_next is
        # only reached if the gate returns normally).
        await self._gate(self._extract(context))
        await call_next()
