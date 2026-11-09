"""Purchase auto-approval agent gated by vsl-core through vsl-maf.

A Microsoft Agent Framework agent with two VSL governance primitives wired as
FunctionMiddleware. Every call the model makes to approve_purchase is gated
before the tool runs:

- Invariant ("purchase-hard-cap"): the amount must never exceed $1000.
- PreNode ("purchase-within-budget"): the purchase must leave headroom in the
  remaining daily budget. Its monitor returns budget / amount, a ratio where
  1.0 is break-even, and records every decision with write_gate_decision().

Neither gate involves a model at the guarded call, so both are F1 = True,
F2 = NONE (assurance level LOW): they block every purchase they deny, but they
do not shape what the model produces. Report the two separately.

A denial aborts the run (VSLGateDenied), so this code sees it, and the model is
never handed a tool error to talk its way around.

Setup:
    pip install "agent-framework-core>=1.25" agent-framework-foundry azure-identity
    pip install git+https://github.com/supersemantics/VSL-Core.git
    export FOUNDRY_PROJECT_ENDPOINT=https://<service>.services.ai.azure.com/api/projects/<project>
    export FOUNDRY_MODEL=gpt-4o-mini
    az login

Run:
    python examples/financial-agents/purchase_approval_agent.py
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Annotated, Any

from azure.identity.aio import AzureCliCredential
from pydantic import Field

from agent_framework import Agent, tool
from agent_framework.foundry import FoundryChatClient

from vsl_core.constructs import Invariant, PreNode
from vsl_core.exceptions import InvariantViolation
from vsl_core.ledger import JsonlLedgerStore, LedgerEntryType, VerbaLedger
from vsl_core.metrics import ROBUST_GAMMA_DEFAULT_THRESHOLD, AssuranceBasis, F2Modification, GammaEstimate

from vsl_maf import MAFAdapter, VSLFunctionMiddleware, VSLGateDenied, tool_arguments

AGENT_ID = "purchase-agent"
HARD_CAP = 1000.0
DAILY_BUDGET_REMAINING = 500.0
BUDGET_GATE = "purchase-within-budget"
CAP_GATE = "purchase-hard-cap"

# No model at either guarded call: the gates only allow or block.
NO_MODEL_SHAPING = AssuranceBasis(f1_pre_commitment=True, f2_modification=F2Modification.NONE)

ledger = VerbaLedger(JsonlLedgerStore("purchase-ledger.jsonl", fsync=True))
current_decision: dict[str, str] = {}


# approval_mode="never_require" for sample brevity; "always_require" adds a human step.
@tool(approval_mode="never_require")
def approve_purchase(
    amount: Annotated[float, Field(description="Purchase amount in USD")],
    vendor: Annotated[str, Field(description="Vendor name")],
) -> str:
    """Approve a purchase transaction."""
    return f"APPROVED: ${amount:.2f} from {vendor}"


async def budget_headroom(candidate: dict[str, Any]) -> GammaEstimate:
    amount = float(candidate.get("amount", 0.0))
    # Ratio where 1.0 is break-even; 0.0 for a missing or bad amount, so the gate fails closed.
    estimate = GammaEstimate(gamma_hat=DAILY_BUDGET_REMAINING / amount if amount > 0 else 0.0)
    ledger.write_gate_decision(
        identity_key=AGENT_ID,
        primitive_name=BUDGET_GATE,                     # must equal the PreNode's name
        estimate=estimate,
        gamma_threshold=ROBUST_GAMMA_DEFAULT_THRESHOLD,  # must equal the PreNode's threshold
        decision_id=current_decision["id"],
    )
    return estimate


async def under_hard_cap(candidate: dict[str, Any]) -> bool:
    holds = 0.0 < float(candidate.get("amount", 0.0)) <= HARD_CAP
    ledger.write_gate_decision(
        identity_key=AGENT_ID, primitive_name=CAP_GATE, allowed=holds, decision_id=current_decision["id"]
    )
    return holds


async def main() -> None:
    adapter = MAFAdapter(identity_key=AGENT_ID)
    cap_gate = adapter.compile_invariant(Invariant(
        name=CAP_GATE,
        description=f"Amount must never exceed ${HARD_CAP:.0f}.",
        rule=under_hard_cap,
        assurance_basis=NO_MODEL_SHAPING,
    ))
    budget_gate = adapter.compile_pre_node(PreNode(
        name=BUDGET_GATE,
        description="A purchase must leave headroom in the remaining daily budget.",
        monitor=budget_headroom,
        assurance_basis=NO_MODEL_SHAPING,
        gamma_threshold=ROBUST_GAMMA_DEFAULT_THRESHOLD,
    ))

    # MAF runs function middleware in registration order before every
    # approve_purchase call: hard cap -> budget -> tool (only if both pass).
    async with (
        AzureCliCredential() as credential,
        Agent(
            client=FoundryChatClient(
                project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
                model=os.getenv("FOUNDRY_MODEL", "gpt-4o-mini"),
                credential=credential,
            ),
            name="PurchaseAgent",
            instructions="You are a purchase approval agent. Always call approve_purchase to process requests.",
            tools=approve_purchase,
            middleware=[
                VSLFunctionMiddleware(cap_gate, candidate_input_fn=tool_arguments, function_name="approve_purchase"),
                VSLFunctionMiddleware(budget_gate, candidate_input_fn=tool_arguments, function_name="approve_purchase"),
            ],
        ) as agent,
    ):
        for request in (
            "Approve a $250 purchase from Contoso.",     # within budget: approved
            "Approve a $480 purchase from Fabrikam.",    # 500/480 = 1.04, not above 1.1: denied
            "Approve a $5000 purchase from Northwind.",  # above the hard cap: blocked
        ):
            current_decision["id"] = str(uuid.uuid4())
            print(f"\nUser: {request}")
            try:
                result = await agent.run(request)
                print(f"Agent: {result.text}")
            except VSLGateDenied as denied:
                kind = "BLOCKED (hard cap)" if isinstance(denied.denial, InvariantViolation) else "DENIED (budget)"
                print(f"  {kind}: {denied.denial.reason}")
                # A denial is drift as far as audit() is concerned: record what happened next.
                ledger.write(
                    LedgerEntryType.TERMINAL if isinstance(denied.denial, InvariantViolation) else LedgerEntryType.PRE_NODE,
                    identity_key=AGENT_ID,
                    decision_id=current_decision["id"],
                    payload={"escalated_to": "finance-approvers", "reason": denied.denial.reason},
                )

    print("\nLedger intact:", ledger.verify_integrity())


if __name__ == "__main__":
    asyncio.run(main())
