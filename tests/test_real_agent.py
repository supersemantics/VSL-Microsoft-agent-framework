"""End-to-end through a real MAF Agent, no credentials: a scripted chat client
asks for a tool call, MAF's own function-invocation loop runs the middleware, and
the tests check what the tool, the model and the caller each see.

Skipped when agent_framework is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("agent_framework")

from agent_framework import (  # noqa: E402
    Agent,
    BaseChatClient,
    ChatMiddlewareLayer,
    ChatResponse,
    Content,
    FunctionInvocationLayer,
    Message,
    tool,
)

from vsl_core.constructs import Invariant, PreNode  # noqa: E402
from vsl_core.exceptions import InvariantViolation  # noqa: E402
from vsl_core.ledger import VerbaLedger  # noqa: E402
from vsl_core.metrics import ROBUST_GAMMA_DEFAULT_THRESHOLD, AssuranceBasis, F2Modification, GammaEstimate  # noqa: E402

from vsl_maf import MAFAdapter, VSLFunctionMiddleware, VSLGateDenied, tool_arguments  # noqa: E402

PURCHASE_LIMIT = 1000.0


class ScriptedClient(FunctionInvocationLayer, ChatMiddlewareLayer, BaseChatClient):
    """Asks for approve_purchase(amount) once, then reports what came back. The
    layer order is the one MAF's own clients use."""

    def __init__(self, amount: float):
        super().__init__()
        self.amount = amount
        self.tool_results: list = []
        self.model_calls = 0

    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        self.model_calls += 1
        results = [c for c in messages[-1].contents if c.type == "function_result"]
        if results:
            self.tool_results.extend(results)
            return ChatResponse(messages=[Message(role="assistant", contents=["done"])])
        call = Content.from_function_call(call_id="c1", name="approve_purchase", arguments={"amount": self.amount})
        return ChatResponse(messages=[Message(role="assistant", contents=[call])])


def _agent(amount: float, on_denial="abort", ledger: VerbaLedger | None = None):
    executed: list[float] = []

    @tool(approval_mode="never_require")
    def approve_purchase(amount: float) -> str:
        executed.append(amount)
        return f"APPROVED {amount}"

    async def headroom(candidate) -> GammaEstimate:
        amount = float(candidate.get("amount", 0.0))
        estimate = GammaEstimate(gamma_hat=PURCHASE_LIMIT / amount if amount > 0 else 0.0)
        if ledger is not None:
            ledger.write_gate_decision(
                identity_key="purchase-agent",
                primitive_name="purchase-within-limit",
                estimate=estimate,
                gamma_threshold=ROBUST_GAMMA_DEFAULT_THRESHOLD,
            )
        return estimate

    gate = MAFAdapter(identity_key="purchase-agent").compile_pre_node(
        PreNode(
            name="purchase-within-limit",
            description="Purchases stay comfortably inside the limit.",
            monitor=headroom,
            assurance_basis=AssuranceBasis(f1_pre_commitment=True, f2_modification=F2Modification.NONE),
        )
    )
    client = ScriptedClient(amount)
    agent = Agent(
        client=client,
        tools=approve_purchase,
        middleware=[VSLFunctionMiddleware(gate, candidate_input_fn=tool_arguments, function_name="approve_purchase", on_denial=on_denial)],
    )
    return agent, client, executed


async def test_allowed_call_runs_the_tool():
    agent, client, executed = _agent(250.0)
    result = await agent.run("buy")
    assert executed == [250.0]
    assert result.text == "done"
    assert client.tool_results[0].result == "APPROVED 250.0"


async def test_denied_call_aborts_the_run_and_never_runs_the_tool():
    agent, client, executed = _agent(5000.0)
    with pytest.raises(VSLGateDenied) as info:
        await agent.run("buy")
    assert executed == []
    assert client.model_calls == 1  # the model was never handed a result to carry on from
    assert info.value.denial.identity_key == "purchase-agent"
    assert info.value.function_name == "approve_purchase"


async def test_return_to_model_hands_the_model_an_error_and_the_run_continues():
    agent, client, executed = _agent(5000.0, on_denial="return_to_model")
    result = await agent.run("buy")
    assert executed == []
    assert result.text == "done"
    assert client.tool_results and client.tool_results[0].exception is not None


async def test_gate_decisions_are_recorded_by_the_application():
    ledger = VerbaLedger()
    agent, _, _ = _agent(250.0, ledger=ledger)
    await agent.run("buy")
    agent, _, _ = _agent(5000.0, ledger=ledger)
    with pytest.raises(VSLGateDenied):
        await agent.run("buy")
    payloads = [e.payload for e in ledger.store.all_entries()]
    assert [p["allowed"] for p in payloads] == [True, False]
    assert all(p["primitive_name"] == "purchase-within-limit" for p in payloads)
    assert ledger.verify_integrity()


async def test_middleware_list_order_is_gate_order_and_the_first_denial_wins():
    """Pins MAF Python's behaviour the README describes (registration order); if a
    release changes it, this test says so. (MAF for .NET runs them the other way.)"""
    order: list[str] = []

    def recording(name, inner):
        async def gate(candidate):
            order.append(name)
            await inner(candidate)
        return gate

    async def under_cap(candidate) -> bool:
        return float(candidate.get("amount", 0.0)) <= PURCHASE_LIMIT

    basis = AssuranceBasis(f1_pre_commitment=True, f2_modification=F2Modification.NONE)
    cap = MAFAdapter().compile_invariant(Invariant(name="cap", description="d", rule=under_cap, assurance_basis=basis))

    async def always_passes(_candidate) -> bool:
        return True

    other = MAFAdapter().compile_invariant(Invariant(name="other", description="d", rule=always_passes, assurance_basis=basis))

    @tool(approval_mode="never_require")
    def approve_purchase(amount: float) -> str:
        return "APPROVED"

    agent = Agent(client=ScriptedClient(5000.0), tools=approve_purchase, middleware=[
        VSLFunctionMiddleware(recording("cap", cap), candidate_input_fn=tool_arguments),
        VSLFunctionMiddleware(recording("other", other), candidate_input_fn=tool_arguments),
    ])
    with pytest.raises(VSLGateDenied):
        await agent.run("buy")
    assert order == ["cap"]


async def test_invariant_denial_reaches_the_caller_as_its_specific_type():
    executed: list = []

    @tool(approval_mode="never_require")
    def approve_purchase(amount: float) -> str:
        executed.append(amount)
        return "APPROVED"

    async def under_cap(candidate) -> bool:
        return float(candidate.get("amount", 0.0)) <= PURCHASE_LIMIT

    gate = MAFAdapter().compile_invariant(
        Invariant(name="purchase-hard-cap", description="Never above the cap.", rule=under_cap,
                  assurance_basis=AssuranceBasis(f1_pre_commitment=True, f2_modification=F2Modification.NONE))
    )
    agent = Agent(client=ScriptedClient(5000.0), tools=approve_purchase,
                  middleware=[VSLFunctionMiddleware(gate, candidate_input_fn=tool_arguments)])
    with pytest.raises(VSLGateDenied) as info:
        await agent.run("buy")
    assert isinstance(info.value.denial, InvariantViolation)
    assert executed == []
