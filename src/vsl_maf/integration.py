"""VSLFunctionMiddleware: MAF FunctionMiddleware that applies a compiled VSL gate
before a tool (function) call runs.

vsl-core does not intercept anything on its own: a compiled gate only ever
raises or returns, on demand, when called. VSLFunctionMiddleware is that call
site, made reusable for MAF's middleware mechanism. It calls the gate
immediately before the tool would run, so the gate completes strictly before the
guarded tool's side effect (F1 pre-commitment).

How a denial reaches your code (agent-framework 1.x)
----------------------------------------------------
MAF's function-invocation loop converts an ordinary exception raised by function
middleware into a tool-error result and keeps running: the tool does not run, but
the model only sees "Error: Function failed.", may retry or carry on, and
``agent.run()`` returns normally. For an enforcement layer that is fail-open at
the level of the run. MAF's fail-closed signal is ``MiddlewareFailure``: never
turned into a tool result, it cancels the current batch of tool calls and
propagates to the caller of ``agent.run()`` (for streaming runs, when the stream
is consumed).

So by default (``on_denial="abort"``) a denial raises ``VSLGateDenied``, a
``MiddlewareFailure`` that carries the original ``AutomationDeniedException`` (or
``InvariantViolation``) as ``.denial`` and as ``__cause__``:

    try:
        result = await agent.run("Approve $5000")
    except VSLGateDenied as denied:
        print(denied.denial.reason)
        if isinstance(denied.denial, InvariantViolation): ...

``on_denial="return_to_model"`` keeps MAF's default instead: the denial becomes a
tool-error result, the model is told the call failed and the run continues. The
tool still never runs. Use it only when you want the model to explain or try
another route, and remember that your code is then not told about the denial.

If the gate itself fails (a monitor that raises something other than a denial),
the middleware fails closed in "abort" mode too: ``MiddlewareFailure`` chained to
the error, so the tool never runs unguarded.

MAF FunctionMiddleware semantics used here:
  - context.function.name  -- the tool being called
  - context.arguments      -- the arguments, a Mapping or a pydantic model
  - call_next()            -- continues to the next middleware or the tool;
                              takes NO arguments in MAF.

Ordering: register VSL middleware after any argument-repair middleware. MAF fails
closed if arguments change after a security or policy middleware has seen them.

`agent_framework` must be importable at runtime for registration on an Agent. The
module still imports without it, so the gate logic can be unit tested alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Awaitable, Callable, Literal

from vsl_core.conformance.protocol import CompiledGate
from vsl_core.exceptions import AutomationDeniedException

try:
    from agent_framework import FunctionMiddleware as _FunctionMiddleware
    from agent_framework import MiddlewareFailure as _MiddlewareFailure

    _MIDDLEWARE_BASE: type = _FunctionMiddleware
    _FAILURE_BASE: type = _MiddlewareFailure
except ImportError:  # allow tests to import this module without agent-framework
    _MIDDLEWARE_BASE = object  # type: ignore[assignment]

    class _FallbackFailure(Exception):
        """Stand-in for agent_framework.MiddlewareFailure when MAF is absent."""

    _FAILURE_BASE = _FallbackFailure

OnDenial = Literal["abort", "return_to_model"]
_ON_DENIAL_VALUES = ("abort", "return_to_model")


class VSLGateDenied(_FAILURE_BASE):  # type: ignore[misc, valid-type]
    """A VSL gate denied a tool call and the run was aborted, fail-closed.

    A ``MiddlewareFailure``, so MAF propagates it to the caller of ``agent.run()``
    instead of handing the model a tool error. ``denial`` is the gate's own
    ``AutomationDeniedException`` (or the more specific ``InvariantViolation``);
    it is also the exception's ``__cause__``. ``function_name`` is the tool that
    was not run.
    """

    def __init__(self, denial: AutomationDeniedException, function_name: str | None = None) -> None:
        tool = f" '{function_name}'" if function_name else ""
        super().__init__(f"VSL gate denied tool call{tool}: {denial}")
        self.denial = denial
        self.function_name = function_name


def tool_arguments(context: Any) -> dict[str, Any]:
    """The tool call's arguments as a plain dict, whether MAF passed a Mapping or
    a pydantic model. The usual ``candidate_input_fn`` for argument-based gates."""
    arguments = getattr(context, "arguments", None)
    if arguments is None:
        return {}
    if isinstance(arguments, Mapping):
        return dict(arguments)
    model_dump = getattr(arguments, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump())
    raise TypeError(f"Unsupported tool-argument type {type(arguments).__name__}.")


class VSLFunctionMiddleware(_MIDDLEWARE_BASE):  # type: ignore[misc]
    """MAF FunctionMiddleware that applies a VSL gate before a tool runs.

    On denial, ``call_next()`` is never called and the guarded tool never runs.
    What the caller sees depends on ``on_denial`` (see the module docstring):
    ``"abort"`` (default) raises ``VSLGateDenied`` out of ``agent.run()``;
    ``"return_to_model"`` lets MAF hand the model a tool error.

    ``candidate_input_fn`` extracts what the gate evaluates from the
    ``FunctionInvocationContext``; it defaults to the whole context. For
    argument-based gates use ``tool_arguments``.

    ``function_name`` scopes the gate to one tool; other tools pass through.
    None means every tool call this middleware receives is gated.
    """

    def __init__(
        self,
        gate: CompiledGate,
        *,
        candidate_input_fn: Callable[[Any], Any] | None = None,
        function_name: str | None = None,
        on_denial: OnDenial = "abort",
    ) -> None:
        if on_denial not in _ON_DENIAL_VALUES:
            raise ValueError(f"on_denial must be one of {_ON_DENIAL_VALUES}, got {on_denial!r}.")
        super().__init__()
        self._gate = gate
        self._extract = candidate_input_fn or (lambda ctx: ctx)
        self._function_name = function_name
        self._on_denial = on_denial

    async def process(
        self,
        context: Any,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        """Gate the tool call. MAF invokes this before the tool runs."""
        tool_name = getattr(getattr(context, "function", None), "name", None)
        if self._function_name is not None and tool_name != self._function_name:
            await call_next()
            return

        try:
            await self._gate(self._extract(context))
        except AutomationDeniedException as denial:
            if self._on_denial == "abort":
                raise VSLGateDenied(denial, tool_name) from denial
            raise
        except _FAILURE_BASE:
            raise
        except Exception as error:
            if self._on_denial == "abort":
                raise _FAILURE_BASE(f"VSL gate for '{tool_name}' failed; not running the tool unguarded") from error
            raise

        # Only reached if the gate returned normally (F1 pre-commitment).
        await call_next()
