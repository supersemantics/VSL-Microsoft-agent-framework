"""Integration tests for VSLFunctionMiddleware against a mock context.

A minimal mock of MAF's FunctionInvocationContext, so these run with or without
agent_framework installed. VSLFunctionMiddleware.process() uses only duck-typed
attribute access (context.function.name, context.arguments) and calls
call_next() with no arguments. tests/test_real_agent.py covers a real MAF Agent.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from vsl_core.constructs import Invariant, PreNode
from vsl_core.exceptions import AutomationDeniedException, InvariantViolation
from vsl_core.metrics import AssuranceBasis, F2Modification, GammaEstimate

from vsl_maf import MAFAdapter, VSLFunctionMiddleware, VSLGateDenied, tool_arguments
from vsl_maf.integration import _FAILURE_BASE


class _MockFunction:
    def __init__(self, name: str):
        self.name = name


class _MockContext:
    def __init__(self, function_name: str, arguments=None):
        self.function = _MockFunction(function_name)
        self.arguments = arguments if arguments is not None else {}
        self.result = None
        self.metadata: dict = {}


# No model at the guarded call in these synthetic gates: F1 = True, F2 = NONE.
_BASIS = AssuranceBasis(f1_pre_commitment=True, f2_modification=F2Modification.NONE)
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
    return PreNode(name="test-pre-node", description="test", monitor=monitor, assurance_basis=_BASIS, gamma_threshold=1.1)


def _make_invariant(rule) -> Invariant:
    return Invariant(name="test-invariant", description="test", rule=rule, assurance_basis=_BASIS)


class _Counter:
    def __init__(self):
        self.calls = 0

    async def __call__(self):
        self.calls += 1


# --- PreNode ---------------------------------------------------------------

async def test_middleware_allows_when_pre_node_passes():
    call_next = _Counter()
    await VSLFunctionMiddleware(_adapter.compile_pre_node(_make_pre_node(_sufficient_monitor))).process(_MockContext("my_tool"), call_next)
    assert call_next.calls == 1


async def test_denial_aborts_the_run_by_default():
    call_next = _Counter()
    middleware = VSLFunctionMiddleware(_adapter.compile_pre_node(_make_pre_node(_insufficient_monitor)))
    with pytest.raises(VSLGateDenied) as info:
        await middleware.process(_MockContext("my_tool"), call_next)
    assert call_next.calls == 0  # F1: the tool never ran
    assert isinstance(info.value.denial, AutomationDeniedException)
    assert info.value.__cause__ is info.value.denial
    assert info.value.function_name == "my_tool"
    assert isinstance(info.value, _FAILURE_BASE)  # MiddlewareFailure, so MAF propagates it


async def test_return_to_model_raises_the_plain_denial():
    call_next = _Counter()
    middleware = VSLFunctionMiddleware(_adapter.compile_pre_node(_make_pre_node(_insufficient_monitor)), on_denial="return_to_model")
    with pytest.raises(AutomationDeniedException):
        await middleware.process(_MockContext("my_tool"), call_next)
    assert call_next.calls == 0


def test_on_denial_is_validated():
    with pytest.raises(ValueError):
        VSLFunctionMiddleware(_adapter.compile_pre_node(_make_pre_node(_sufficient_monitor)), on_denial="ignore")


async def test_a_failing_gate_fails_closed():
    async def broken_monitor(_):
        raise RuntimeError("judge service down")

    call_next = _Counter()
    middleware = VSLFunctionMiddleware(_adapter.compile_pre_node(_make_pre_node(broken_monitor)))
    with pytest.raises(_FAILURE_BASE) as info:
        await middleware.process(_MockContext("my_tool"), call_next)
    assert call_next.calls == 0
    assert isinstance(info.value.__cause__, RuntimeError)
    assert not isinstance(info.value, VSLGateDenied)


# --- Invariant -------------------------------------------------------------

async def test_middleware_allows_when_invariant_holds():
    call_next = _Counter()
    await VSLFunctionMiddleware(_adapter.compile_invariant(_make_invariant(_passing_rule))).process(_MockContext("tool"), call_next)
    assert call_next.calls == 1


async def test_invariant_violation_is_carried_specifically():
    call_next = _Counter()
    middleware = VSLFunctionMiddleware(_adapter.compile_invariant(_make_invariant(_failing_rule)))
    with pytest.raises(VSLGateDenied) as info:
        await middleware.process(_MockContext("tool"), call_next)
    assert isinstance(info.value.denial, InvariantViolation)
    assert call_next.calls == 0


async def test_return_to_model_keeps_the_invariant_type():
    middleware = VSLFunctionMiddleware(_adapter.compile_invariant(_make_invariant(_failing_rule)), on_denial="return_to_model")
    with pytest.raises(InvariantViolation):
        await middleware.process(_MockContext("tool"), _Counter())


# --- scoping and extraction ------------------------------------------------

async def test_middleware_bypasses_when_function_name_does_not_match():
    call_next = _Counter()
    middleware = VSLFunctionMiddleware(_adapter.compile_pre_node(_make_pre_node(_insufficient_monitor)), function_name="guarded_tool")
    await middleware.process(_MockContext("other_tool"), call_next)
    assert call_next.calls == 1


async def test_middleware_gates_when_function_name_matches():
    call_next = _Counter()
    middleware = VSLFunctionMiddleware(_adapter.compile_pre_node(_make_pre_node(_insufficient_monitor)), function_name="guarded_tool")
    with pytest.raises(VSLGateDenied):
        await middleware.process(_MockContext("guarded_tool"), call_next)
    assert call_next.calls == 0


async def test_tool_arguments_handles_mappings_and_pydantic_models():
    class Args(BaseModel):
        amount: float

    received = []

    async def monitor(candidate):
        received.append(candidate)
        return GammaEstimate(gamma_hat=5.0)

    middleware = VSLFunctionMiddleware(_adapter.compile_pre_node(_make_pre_node(monitor)), candidate_input_fn=tool_arguments)
    await middleware.process(_MockContext("tool", arguments={"amount": 42.0}), _Counter())
    await middleware.process(_MockContext("tool", arguments=Args(amount=7.0)), _Counter())
    assert received == [{"amount": 42.0}, {"amount": 7.0}]


async def test_call_next_receives_no_arguments():
    call_next_args = []

    async def call_next(*args, **kwargs):
        call_next_args.append((args, kwargs))

    await VSLFunctionMiddleware(_adapter.compile_pre_node(_make_pre_node(_sufficient_monitor))).process(_MockContext("tool"), call_next)
    assert call_next_args == [((), {})]


async def test_gate_is_reusable_across_process_calls():
    async def conditional_monitor(candidate):
        return GammaEstimate(gamma_hat=5.0 if candidate == "good" else 0.1)

    gate = _adapter.compile_pre_node(_make_pre_node(conditional_monitor))
    middleware = VSLFunctionMiddleware(gate, candidate_input_fn=lambda ctx: ctx.arguments.get("input"))
    await middleware.process(_MockContext("tool", {"input": "good"}), _Counter())
    with pytest.raises(VSLGateDenied):
        await middleware.process(_MockContext("tool", {"input": "bad"}), _Counter())
    await middleware.process(_MockContext("tool", {"input": "good"}), _Counter())


# --- adapter ---------------------------------------------------------------

async def test_denials_carry_the_configured_identity():
    adapter = MAFAdapter(identity_key="purchase-agent")
    with pytest.raises(AutomationDeniedException) as pre:
        await adapter.compile_pre_node(_make_pre_node(_insufficient_monitor))("x")
    with pytest.raises(InvariantViolation) as inv:
        await adapter.compile_invariant(_make_invariant(_failing_rule))("x")
    assert pre.value.identity_key == inv.value.identity_key == "purchase-agent"
    with pytest.raises(ValueError):
        MAFAdapter(identity_key="")


async def _judge_with_a_maf_chat_client(candidate):  # never awaited: only its source is inspected
    from agent_framework.foundry import FoundryChatClient  # noqa: F401

    return GammaEstimate(gamma_hat=0.0)


def test_full_claim_on_a_maf_chat_client_judge_warns():
    pre_node = PreNode(
        name="judge",
        description="LLM judge",
        monitor=_judge_with_a_maf_chat_client,
        assurance_basis=AssuranceBasis(f1_pre_commitment=True, f2_modification=F2Modification.FULL),
    )
    with pytest.warns(UserWarning, match="hosted model API"):
        MAFAdapter().compile_pre_node(pre_node)
