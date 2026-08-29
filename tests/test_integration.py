"""Integration tests for VSLFunctionMiddleware.

Uses a minimal mock of MAF's FunctionInvocationContext so these tests run
without agent_framework installed.  VSLFunctionMiddleware.process() uses only
duck-typed attribute access (context.function.name) and calls call_next() with
no arguments -- both are verified here through the mock.
"""

from __future__ import annotations

import pytest

from vsl_core.constructs import Invariant, PreNode
from vsl_core.exceptions import AutomationDeniedException, InvariantViolation
from vsl_core.metrics import AssuranceBasis, F2Modification, GammaEstimate

from vsl_maf import MAFAdapter, VSLFunctionMiddleware


# ---------------------------------------------------------------------------
# Minimal mock of MAF's FunctionInvocationContext
# ---------------------------------------------------------------------------

class _MockFunction:
    def __init__(self, name: str):
        self.name = name


class _MockContext:
    def __init__(self, function_name: str, arguments: dict | None = None):
        self.function = _MockFunction(function_name)
        self.arguments = arguments or {}
        self.result = None
        self.metadata: dict = {}


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_BASIS = AssuranceBasis(f1_pre_commitment=True, f2_modification=F2Modification.FULL)
_adapter = MAFAdapter()


async def _sufficient_monitor(_: object) -> GammaEstimate:
    return GammaEstimate(gamma_hat=5.0)


async def _insufficient_monitor(_: object) -> GammaEstimate:
    return GammaEstimate(gamma_hat=0.1)


async def _passing_rule(_: object) -> bool:
    return True


async def _failing_rule(_: object) -> bool:
    return False


def _make_pre_node(monitor) -> PreNode:
    return PreNode(
        name="test-pre-node",
        description="test",
        monitor=monitor,
        assurance_basis=_BASIS,
        gamma_threshold=1.1,
    )


def _make_invariant(rule) -> Invariant:
    return Invariant(
        name="test-invariant",
        description="test",
        rule=rule,
        assurance_basis=_BASIS,
    )


# ---------------------------------------------------------------------------
# Tests: PreNode gate via VSLFunctionMiddleware
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_middleware_allows_when_pre_node_passes():
    gate = _adapter.compile_pre_node(_make_pre_node(_sufficient_monitor))
    middleware = VSLFunctionMiddleware(gate)

    side_effect = [0]

    async def call_next():
        side_effect[0] += 1

    await middleware.process(_MockContext("my_tool"), call_next)
    assert side_effect[0] == 1


@pytest.mark.asyncio
async def test_middleware_blocks_when_pre_node_denies():
    gate = _adapter.compile_pre_node(_make_pre_node(_insufficient_monitor))
    middleware = VSLFunctionMiddleware(gate)

    side_effect = [0]

    async def call_next():
        side_effect[0] += 1

    with pytest.raises(AutomationDeniedException):
        await middleware.process(_MockContext("my_tool"), call_next)

    # call_next must never have been called -- F1 pre-commitment
    assert side_effect[0] == 0


# ---------------------------------------------------------------------------
# Tests: Invariant gate via VSLFunctionMiddleware
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_middleware_allows_when_invariant_holds():
    gate = _adapter.compile_invariant(_make_invariant(_passing_rule))
    middleware = VSLFunctionMiddleware(gate)
    side_effect = [0]

    async def call_next():
        side_effect[0] += 1

    await middleware.process(_MockContext("tool"), call_next)
    assert side_effect[0] == 1


@pytest.mark.asyncio
async def test_middleware_raises_invariant_violation_specifically():
    gate = _adapter.compile_invariant(_make_invariant(_failing_rule))
    middleware = VSLFunctionMiddleware(gate)
    side_effect = [0]

    async def call_next():
        side_effect[0] += 1

    with pytest.raises(InvariantViolation):
        await middleware.process(_MockContext("tool"), call_next)
    assert side_effect[0] == 0


# ---------------------------------------------------------------------------
# Tests: function_name scoping
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_middleware_bypasses_when_function_name_does_not_match():
    gate = _adapter.compile_pre_node(_make_pre_node(_insufficient_monitor))
    middleware = VSLFunctionMiddleware(gate, function_name="guarded_tool")

    side_effect = [0]

    async def call_next():
        side_effect[0] += 1

    # Different tool name -- middleware should pass through without gating
    await middleware.process(_MockContext("other_tool"), call_next)
    assert side_effect[0] == 1


@pytest.mark.asyncio
async def test_middleware_gates_when_function_name_matches():
    gate = _adapter.compile_pre_node(_make_pre_node(_insufficient_monitor))
    middleware = VSLFunctionMiddleware(gate, function_name="guarded_tool")

    side_effect = [0]

    async def call_next():
        side_effect[0] += 1

    with pytest.raises(AutomationDeniedException):
        await middleware.process(_MockContext("guarded_tool"), call_next)
    assert side_effect[0] == 0


# ---------------------------------------------------------------------------
# Tests: candidate_input_fn extraction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_candidate_input_fn_receives_extracted_value():
    received = []

    async def monitor(candidate):
        received.append(candidate)
        return GammaEstimate(gamma_hat=5.0)

    pre_node = PreNode(
        name="capture-monitor",
        description="test",
        monitor=monitor,
        assurance_basis=_BASIS,
        gamma_threshold=1.1,
    )
    gate = _adapter.compile_pre_node(pre_node)
    middleware = VSLFunctionMiddleware(gate, candidate_input_fn=lambda ctx: dict(ctx.arguments))

    async def call_next():
        pass

    ctx = _MockContext("tool", arguments={"amount": 42.0})
    await middleware.process(ctx, call_next)
    assert received == [{"amount": 42.0}]


# ---------------------------------------------------------------------------
# Tests: MAF call_next takes no arguments
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_next_receives_no_arguments():
    """MAF's call_next() contract: no args -- verify middleware never passes args."""
    gate = _adapter.compile_pre_node(_make_pre_node(_sufficient_monitor))
    middleware = VSLFunctionMiddleware(gate)

    call_next_args = []

    async def call_next(*args, **kwargs):
        call_next_args.append((args, kwargs))

    await middleware.process(_MockContext("tool"), call_next)
    # call_next must have been called with no positional or keyword arguments
    assert call_next_args == [((), {})]


# ---------------------------------------------------------------------------
# Tests: gate reusability across multiple middleware calls
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gate_is_reusable_across_process_calls():
    """Same compiled gate called multiple times via process must not leak state."""

    async def conditional_monitor(candidate):
        return GammaEstimate(gamma_hat=5.0 if candidate == "good" else 0.1)

    pre_node = PreNode(
        name="reusable",
        description="test",
        monitor=conditional_monitor,
        assurance_basis=_BASIS,
        gamma_threshold=1.1,
    )
    gate = _adapter.compile_pre_node(pre_node)
    middleware = VSLFunctionMiddleware(
        gate, candidate_input_fn=lambda ctx: ctx.arguments.get("input")
    )

    async def call_next():
        pass

    await middleware.process(_MockContext("tool", {"input": "good"}), call_next)

    with pytest.raises(AutomationDeniedException):
        await middleware.process(_MockContext("tool", {"input": "bad"}), call_next)

    # Should pass again -- no state leak from the failing call
    await middleware.process(_MockContext("tool", {"input": "good"}), call_next)
