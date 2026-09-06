# vsl-maf

A [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/overview/) (`agent-framework`) adapter for [vsl-core](https://github.com/supersemantics/VSL-Core): implements `vsl_core.conformance.protocol.VSLAdapter` and provides `VSLFunctionMiddleware` — a `FunctionMiddleware` subclass that gates every tool call an `Agent` makes before it fires.

Is:

- A `VSLAdapter` implementation (`MAFAdapter`) that passes `vsl_core.conformance.suite.run_conformance_suite` with an empty result.
- A `VSLFunctionMiddleware` wiring class that inherits from MAF's `FunctionMiddleware`, calling a compiled gate inside `process()` before `call_next()` runs, preserving F1 pre-commitment: the gate completes strictly before the guarded tool's side effect.

Is not:

- A replacement for `vsl-core` itself -- the PreNode/Invariant/Fallback/TerminalState constructs, the ledger, and the governance vocabulary all live in `vsl-core`. This package only compiles and wires those constructs into MAF.
- Automatic interception. `VSLFunctionMiddleware` must be explicitly passed in the `middleware=[...]` list when creating an `Agent`.

## Install

```bash
pip install agent-framework
pip install git+https://github.com/supersemantics/VSL-Core.git
pip install -e .   # from a local clone of this repo
```

## Usage

```python
from agent_framework import Agent, tool
from agent_framework.foundry import FoundryChatClient
from azure.identity.aio import AzureCliCredential
from pydantic import Field
from typing import Annotated

from vsl_core.constructs import PreNode
from vsl_core.metrics import AssuranceBasis, F2Modification, GammaEstimate
from vsl_maf import MAFAdapter, VSLFunctionMiddleware

@tool(approval_mode="never_require")
def my_tool(value: Annotated[float, Field(description="Input value")]) -> str:
    return f"Processed: {value}"

async def monitor(candidate):
    return GammaEstimate(gamma_hat=candidate["confidence"])

pre_node = PreNode(
    name="my-pre-node",
    description="...",
    monitor=monitor,
    assurance_basis=AssuranceBasis(f1_pre_commitment=True, f2_modification=F2Modification.FULL),
    gamma_threshold=1.0,
)
gate = MAFAdapter().compile_pre_node(pre_node)

async with Agent(
    client=FoundryChatClient(project_endpoint="...", model="gpt-4o-mini", credential=AzureCliCredential()),
    name="MyAgent",
    instructions="Use my_tool to process requests.",
    tools=my_tool,
    middleware=[
        VSLFunctionMiddleware(
            gate,
            candidate_input_fn=lambda ctx: dict(ctx.arguments),
            function_name="my_tool",
        ),
    ],
) as agent:
    try:
        result = await agent.run("Process value=0.5 with confidence=0.1")
    except AutomationDeniedException as exc:
        print(f"Tool call denied: {exc.reason}")
```

`compile_invariant` works the same way -- pass an `Invariant` instead of a `PreNode`. An Invariant violation raises the more specific `InvariantViolation` (a subclass of `AutomationDeniedException`).

## How it works in MAF

`VSLFunctionMiddleware` is a `FunctionMiddleware` subclass registered on the `Agent`. MAF calls `process(context, call_next)` before every tool invocation where:

- `context.function.name` — the tool being called
- `context.arguments` — the validated arguments dict
- `call_next()` — continues to the next middleware or the actual tool (**no arguments** — MAF's contract)

If the compiled gate raises (Gamma insufficient / Invariant violated), `call_next()` is never reached and the tool never runs — F1 pre-commitment holds.

## Chaining multiple gates

```python
middleware=[
    VSLFunctionMiddleware(confidence_gate, candidate_input_fn=..., function_name="approve_purchase"),
    VSLFunctionMiddleware(cap_gate,        candidate_input_fn=..., function_name="approve_purchase"),
]
```

MAF runs middleware in registration order. If the confidence gate raises, the cap gate and the tool never fire.

## `VSLFunctionMiddleware` doesn't consume `Fallback`

A `PreNode`'s `Fallback` (`on_failure`, `delta_factor`, `max_retries`, `on_max_retries`) is policy data only here -- `MAFAdapter.compile_pre_node`/`compile_invariant` never act on `pre_node.fallback`. On denial, `VSLFunctionMiddleware.process` raises once, immediately; there is no retry loop. `compile_pre_node` warns (`UserWarning`) if a `PreNode`'s `Fallback` is non-default, so this can't be silently assumed -- but the field itself stays inert here, same as `vsl-langgraph`'s `LangGraphAdapter` and `vsl-langchain`'s `LangChainAdapter`. `vsl-core`'s own `PlainPythonReferenceAdapter` doesn't consume `Fallback` either.

## `VSLFunctionMiddleware` doesn't write to the ledger either

Nothing in `process()` calls `VerbaLedger.write*` -- a gate denial becomes a raised exception, and that's it. If you want an audit trail of gate decisions, your own code (e.g. the `try`/`except` around `agent.run()`) has to call `ledger.write_monitor`/`ledger.write(LedgerEntryType.PRE_NODE, ...)`/`write_verification` explicitly, causally linked via `caused_by` -- the same three entry types `vsl-core`'s own [building guide](https://github.com/supersemantics/VSL-Core/blob/main/docs/building-with-vsl-core.md#step-3-the-five-things-youll-actually-write) says to write around every gated call. Skip `PRE_NODE` and `VerbaLedger.audit()`'s `drift_flagged_monitor_has_pre_node` check fails on any denial -- not a bug in `audit()`, just an incomplete write. An agent with no ledger calls anywhere produces real, working gating and zero audit trail -- this package won't warn you, because from its point of view every gate did exactly what it was compiled to do.

For persisting those writes to a durable, hosted ledger rather than a local file, `VerbaLedger` accepts any `LedgerStore` -- see [`vsl-core-ledger-client`](https://github.com/supersemantics/VSL-Core-ledger-client).

## The F1/F2 assurance distinction -- same caveat as `vsl-core`, `vsl-langgraph`, and `vsl-langchain`

Every `PreNode`/`Invariant` here still requires an `AssuranceBasis`, and it's still **self-declared, not independently verified** -- nothing in this package checks that a stated `F2Modification.FULL` actually matches what a `monitor`/`rule` does. See `vsl-core`'s README ("The F1/F2 assurance distinction") for the full explanation.

`MAFAdapter` does carry one detectable-case check, the same heuristic `vsl-langgraph`'s `LangGraphAdapter` has: `compile_pre_node`/`compile_invariant` warn (`UserWarning`) if a `monitor`/`rule` claims `F2Modification.FULL` but its source appears to call a hosted model API (`openai`/`anthropic`/`google.genai`/`cohere`/`mistralai`) -- a black-box API call structurally cannot modify the energy function directly, so `FULL` isn't achievable that way; the honest ceiling is `PARTIAL`. This is a heuristic, not proof: it can miss indirect calls, and passing it doesn't confirm a genuine `FULL` claim is correct either.

## Node-level Gamma is not Cluster-level safety

This matters as soon as more than one `Agent` is coordinating, which MAF (like any multi-agent framework) supports: **gating `gamma_threshold` at every tool call does not mean the composed, multi-agent system is safe.** Node-level Gamma is necessary but not sufficient.

`vsl-core`'s own motivating example is the 2010 Flash Crash: every individual participant's Gamma was above 1 -- each one, in isolation, looked sufficiently governed -- and the Cluster-level Gamma was still 0. A `Cluster`'s Gamma is a separate quantity, checked separately, and it can fail even when every constituent `PreNode`/`Invariant` gate is passing cleanly. `vsl_core.cluster.Cluster` isn't MAF-specific -- its `invariants` field holds plain `Invariant` objects, compiled and wired through `MAFAdapter`/`VSLFunctionMiddleware` exactly like any other. If more than one `Agent` is coordinating on a shared outcome, `Cluster`/`ClusterKey` is the construct that addresses this -- not a denser sprinkling of `VSLFunctionMiddleware` instances on individual tool calls.

## Sample agents

`examples/financial-agents/purchase_approval_agent.py` chains both governance primitives on a real MAF `Agent`:

- **PreNode** (`purchase-confidence`): soft, threshold-based confidence gate.
- **Invariant** (`purchase-hard-cap`): non-bypassable hard cap -- blocked even when confidence is high.

It creates a real `Agent` with `FoundryChatClient`, so it needs real Azure/Foundry credentials to run. Set env vars and run:

```bash
export FOUNDRY_PROJECT_ENDPOINT=https://<your-service>.services.ai.azure.com/api/projects/<your-project>
export FOUNDRY_MODEL=gpt-4o-mini
az login
python examples/financial-agents/purchase_approval_agent.py
```

For the credential-free path -- `VSLFunctionMiddleware.process()` exercised directly against a mock context, proving gate behavior without any real agent or API call -- see `tests/test_integration.py`.

## Layout

```
src/vsl_maf/
├── adapter.py      MAFAdapter (VSLAdapter conformance contract)
└── integration.py  VSLFunctionMiddleware (FunctionMiddleware subclass)
examples/
├── README.md
└── financial-agents/
    └── purchase_approval_agent.py   PreNode + Invariant, Part 1 synthetic / Part 2 real MAF Agent
tests/
├── test_conformance.py    run_conformance_suite(MAFAdapter()) == []
└── test_integration.py    exercises VSLFunctionMiddleware.process() with a mock MAF context
```

## Status

Alpha. Conformance-suite green; integration tested against a mock MAF `FunctionInvocationContext` for both the `PreNode` and `Invariant` paths (pass-through and denial), `function_name` scoping, `candidate_input_fn` extraction, MAF's no-argument `call_next()` contract, and gate reusability. The real `Agent` integration in Part 2 of the example has not yet been end-to-end tested against a live Foundry endpoint.

## License

MIT -- see LICENSE. Copyright Super Semantics.
