# vsl-maf

A [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/overview/) (`agent-framework`) adapter for [vsl-core](https://github.com/supersemantics/VSL-Core): implements `vsl_core.conformance.protocol.VSLAdapter` and provides `VSLFunctionMiddleware`, a `FunctionMiddleware` subclass that gates every tool call an `Agent` makes before it fires.

Requires vsl-core 0.2.0 or later and agent-framework-core 1.25 or later. Handover records (below) need vsl-core 0.3.0.

Is:

- A `VSLAdapter` implementation (`MAFAdapter`) that passes `vsl_core.conformance.suite.run_conformance_suite` with an empty result.
- `VSLFunctionMiddleware`, which calls a compiled gate inside `process()` before `call_next()`, so the gate completes strictly before the guarded tool's side effect (F1 pre-commitment), and makes a denial abort the run fail-closed.

Is not:

- A replacement for `vsl-core`: the PreNode/Invariant/Fallback/TerminalState constructs, the ledger and the governance vocabulary all live there. This package only compiles and wires them into MAF.
- Automatic interception. `VSLFunctionMiddleware` must be passed in the `middleware=[...]` list when creating an `Agent`.

## Install

```bash
pip install "agent-framework-core>=1.25"
pip install git+https://github.com/supersemantics/VSL-Core.git
pip install -e .   # from a local clone of this repo
```

Add the MAF connector for your model service as well, for example `agent-framework-foundry`.

## Usage

```python
from typing import Annotated

from agent_framework import Agent, tool
from agent_framework.foundry import FoundryChatClient
from azure.identity.aio import AzureCliCredential
from pydantic import Field

from vsl_core.constructs import PreNode
from vsl_core.ledger import VerbaLedger
from vsl_core.metrics import ROBUST_GAMMA_DEFAULT_THRESHOLD, AssuranceBasis, F2Modification, GammaEstimate
from vsl_maf import MAFAdapter, VSLFunctionMiddleware, VSLGateDenied, tool_arguments

REFUND_LIMIT = 500.0
REFUND_GATE = "refund-within-policy"
ledger = VerbaLedger()

@tool(approval_mode="never_require")
def issue_refund(amount: Annotated[float, Field(description="Refund amount")]) -> str:
    return f"Refunded {amount}"

async def refund_headroom(candidate) -> GammaEstimate:
    amount = float(candidate.get("amount", 0.0))
    # A ratio where 1.0 is break-even; 0.0 for a missing amount, so the gate fails closed.
    estimate = GammaEstimate(gamma_hat=REFUND_LIMIT / amount if amount > 0 else 0.0)
    ledger.write_gate_decision(
        identity_key="refund-agent",
        primitive_name=REFUND_GATE,
        estimate=estimate,
        gamma_threshold=ROBUST_GAMMA_DEFAULT_THRESHOLD,
    )
    return estimate

gate = MAFAdapter(identity_key="refund-agent").compile_pre_node(PreNode(
    name=REFUND_GATE,
    description="Refunds must stay comfortably inside the policy limit.",
    monitor=refund_headroom,
    # No model at the guarded call: the gate only allows or blocks.
    assurance_basis=AssuranceBasis(f1_pre_commitment=True, f2_modification=F2Modification.NONE),
    gamma_threshold=ROBUST_GAMMA_DEFAULT_THRESHOLD,
))

async with Agent(
    client=FoundryChatClient(project_endpoint="...", model="gpt-4o-mini", credential=AzureCliCredential()),
    name="RefundAgent",
    instructions="Use issue_refund to process refund requests.",
    tools=issue_refund,
    middleware=[VSLFunctionMiddleware(gate, candidate_input_fn=tool_arguments, function_name="issue_refund")],
) as agent:
    try:
        result = await agent.run("Refund 900 to order 1234")
    except VSLGateDenied as denied:
        print(f"Refund denied before it ran: {denied.denial.reason}")
```

`compile_invariant` works the same way with an `Invariant`. Its denial carries the more specific `InvariantViolation` as `denied.denial`.

## How a denial reaches your code

MAF's function-invocation loop turns an ordinary exception from function middleware into a tool-error result and keeps running. The tool does not run, but the model is only told "Error: Function failed.", may retry or carry on, and `agent.run()` returns normally, so your code never learns about the denial. For a governance gate that is fail-open at the level of the run.

So by default (`on_denial="abort"`) a denial raises `VSLGateDenied`, a subclass of MAF's `MiddlewareFailure`. MAF's documented fail-closed signal is never turned into a tool result: it cancels the current batch of tool calls and propagates to the caller of `agent.run()` (for streaming runs, when the stream is consumed). `denied.denial` is the gate's `AutomationDeniedException` or `InvariantViolation` (also the exception's `__cause__`), and `denied.function_name` is the tool that did not run.

`on_denial="return_to_model"` keeps MAF's default: the model gets a tool error and the run continues. The tool still never runs. Use it only when you want the model to explain the refusal or take another route, knowing your code is not told.

A gate that fails for another reason (a monitor raising `RuntimeError`, say) also aborts with `MiddlewareFailure` in `"abort"` mode, so the tool never runs unguarded.

A caveat from MAF itself: when the model asks for several tools at once, an abort cancels the rest of the batch cooperatively. A synchronous tool body already running in a worker thread cannot be interrupted, though its result is discarded. Each gated tool's own gate still runs before that tool.

## How it works in MAF

MAF calls `process(context, call_next)` before every tool invocation:

- `context.function.name`: the tool being called.
- `context.arguments`: the arguments, as a mapping or a pydantic model. `tool_arguments(context)` returns a plain dict either way.
- `call_next()`: continues to the next middleware or the tool. It takes no arguments.

If the gate raises, `call_next()` is never reached and the tool never runs.

Register VSL middleware after any argument-repair middleware: MAF fails closed if arguments change after a policy middleware has seen them.

## Chaining multiple gates

```python
middleware=[
    VSLFunctionMiddleware(cap_gate,    candidate_input_fn=tool_arguments, function_name="approve_purchase"),
    VSLFunctionMiddleware(budget_gate, candidate_input_fn=tool_arguments, function_name="approve_purchase"),
]
```

MAF runs middleware in registration order. If the first gate denies, the second gate and the tool never run.

## Writing gate values honestly

vsl-core never measures Gamma: your monitor computes `gamma_hat` and the gate only compares it, passing when it is strictly above `max(threshold, 1 + delta_estimation_error)` (vsl-core 0.2.0 rules, version `2026.10`).

- Build it as a ratio where 1.0 is break-even: a limit divided by the requested amount, a judge's score divided by its break-even score.
- Return `0.0` when the input is missing or unreadable, so the gate fails closed.
- Never pass a raw 0–1 confidence, or a confidence the model supplies in the tool's own arguments.
- Keep thresholds above 1.0 (`ROBUST_GAMMA_HIGH_STAKES_THRESHOLD`, 1.2, for payments, medical and legal actions). For a rule that must never be broken, use an `Invariant`.

## `VSLFunctionMiddleware` doesn't consume `Fallback`

A `PreNode`'s `Fallback` is policy data only here: on denial the middleware raises once, immediately, with no retry loop. `compile_pre_node` warns (`UserWarning`) if a `Fallback` is non-default, so this can't be silently assumed. vsl-core's own reference adapter doesn't consume `Fallback` either.

## Recording decisions: the ledger doesn't write itself

Nothing in this package writes to a `VerbaLedger`. That is the application's choice, and vsl-core's conformance contract forbids an adapter from making it. Record each gate's decision from inside its monitor or rule with `write_gate_decision()` (vsl-core 0.2.0), as in the usage example above:

- For a PreNode, pass the monitor's `GammaEstimate` and the PreNode's `gamma_threshold`. `allowed` is then computed with the same check the gate makes, and a contradiction is refused.
- For an Invariant, pass `allowed=holds`.
- Pass a `decision_id` to group a decision's records.

A denied gate's record flags drift, so follow it with the decision's next entry (`PRE_NODE`, then `VERIFICATION`, or `TERMINAL`) from your `except VSLGateDenied` block, as `examples/financial-agents/purchase_approval_agent.py` does. An agent with no ledger calls produces working gates and no audit trail, and this package won't warn you.

`VerbaLedger` accepts any `LedgerStore`: `JsonlLedgerStore` for a local append-only file, or the optional immudb store in vsl-core.

## Multi-agent workflows: handovers and Cluster Gamma

MAF composes agents: one agent can call another as a tool (`agent.as_tool()`), or a workflow can hand a conversation from one agent to the next. Each agent's own gates can pass while the hand-off between them goes unchecked.

With vsl-core 0.3.0, record the hand-off before the receiving agent acts, and have the receiver name it in its first gate decision:

```python
handover = ledger.write_handover(
    identity_key="intake-agent",
    to_identity_key="refund-agent",
    decision_id=decision_id,
    reason="customer asked for a refund",
)
# ...inside the refund agent's monitor:
ledger.write_gate_decision(
    identity_key="refund-agent",
    primitive_name=REFUND_GATE,
    estimate=estimate,
    gamma_threshold=ROBUST_GAMMA_DEFAULT_THRESHOLD,
    decision_id=decision_id,
    caused_by=handover.entry_id,   # shows the receiver checked what it was handed
)
```

When the sub-agent is called as a tool, the calling agent's `VSLFunctionMiddleware` sees that call like any other, so the same gating applies to the hand-off itself.

Gating every tool call does not make the composed system safe. vsl-core's motivating example is the 2010 Flash Crash: every participant's Gamma was above 1, and the Cluster-level Gamma was still 0. If several agents coordinate on one outcome, declare them as a `vsl_core.cluster.Cluster` and pass its `cluster_key` on their records. The Cluster's `invariants` are plain `Invariant` objects, compiled and wired through `MAFAdapter`/`VSLFunctionMiddleware` like any other.

## The F1/F2 assurance distinction

Every `PreNode`/`Invariant` requires an `AssuranceBasis`. F2 is judged at the call the gate guards and is never earned by the gate itself:

| Gate | F1 | F2 | Level |
|---|---|---|---|
| Amount cap or budget check before a tool (no model at the call) | True | `NONE` | LOW |
| Gate before a hosted-model call whose system prompt the application writes | True | `PARTIAL` | MEDIUM |
| Gate before a self-hosted model where the application steers logits or activations | True | `FULL` | HIGH |

The level measures influence on the model, not enforcement: a LOW gate still blocks every tool call it denies. Report "blocks before the action: yes" separately rather than raising F2.

The basis is self-declared, not verified. `MAFAdapter` carries one heuristic check: `compile_pre_node`/`compile_invariant` warn (`UserWarning`) when a monitor or rule claims `F2Modification.FULL` but its source appears to call a hosted model API (`openai`, `anthropic`, `google.genai`, `cohere`, `mistralai`, Azure AI inference, or a MAF chat client such as `FoundryChatClient`). A hosted, black-box call cannot modify the energy function, so the honest ceiling is `PARTIAL`. The check can miss indirect calls, and passing it doesn't confirm a `FULL` claim.

## Sample agents

`examples/financial-agents/purchase_approval_agent.py` chains an Invariant (hard cap) and a PreNode (budget headroom) on a real MAF `Agent` with `FoundryChatClient`, records every decision, and handles denials. It needs Azure/Foundry credentials; see `examples/README.md`.

## Layout

```
src/vsl_maf/
├── adapter.py      MAFAdapter (VSLAdapter conformance contract)
└── integration.py  VSLFunctionMiddleware, VSLGateDenied, tool_arguments
examples/
└── financial-agents/purchase_approval_agent.py
tests/
├── test_conformance.py    run_conformance_suite(MAFAdapter()) == []
├── test_integration.py    VSLFunctionMiddleware.process() against a mock context
└── test_real_agent.py     a real MAF Agent with a scripted chat client (no credentials)
```

## Status

Alpha. Conformance-suite green. Tested against agent-framework-core 1.25.0, including end-to-end runs through a real `Agent` and MAF's own function-invocation loop with a scripted chat client: allowed calls, denials aborting the run, `return_to_model`, Invariant denials and recorded gate decisions. The Foundry example has not been run against a live Foundry endpoint.

See [CHANGELOG.md](CHANGELOG.md).

## License

MIT -- see LICENSE. Copyright Super Semantics.
