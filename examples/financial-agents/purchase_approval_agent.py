"""Purchase auto-approval agent gated by vsl-core through vsl-maf.

A Microsoft Agent Framework agent with two VSL governance primitives wired as
FunctionMiddleware -- every call the LLM makes to approve_purchase is
automatically gated before the tool runs:

- PreNode  ("purchase-confidence"): soft gate -- monitor confidence must
  clear the robust Gamma threshold.
- Invariant ("purchase-hard-cap"): hard gate -- amount must never exceed $1000,
  regardless of confidence.

Setup:
    pip install agent-framework azure-identity pydantic
    export FOUNDRY_PROJECT_ENDPOINT=https://<service>.services.ai.azure.com/api/projects/<project>
    export FOUNDRY_MODEL=gpt-4o-mini
    az login

Run:
    python examples/financial-agents/purchase_approval_agent.py
"""

from __future__ import annotations

import asyncio
import os
from typing import Annotated, Any

from azure.identity.aio import AzureCliCredential
from pydantic import Field

from agent_framework import Agent, tool
from agent_framework.foundry import FoundryChatClient

from vsl_core.constructs import Invariant, PreNode
from vsl_core.exceptions import AutomationDeniedException, InvariantViolation
from vsl_core.metrics import ROBUST_GAMMA_DEFAULT_THRESHOLD, AssuranceBasis, F2Modification, GammaEstimate

from vsl_maf import MAFAdapter, VSLFunctionMiddleware


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

# approval_mode="never_require" for sample brevity; use "always_require" in prod.
@tool(approval_mode="never_require")
def approve_purchase(
    amount: Annotated[float, Field(description="Purchase amount in USD")],
    confidence: Annotated[float, Field(description="Approval confidence score (0-10)")],
) -> str:
    """Approve a purchase transaction."""
    return f"APPROVED: ${amount:.2f} (confidence={confidence:.1f})"


# ---------------------------------------------------------------------------
# VSL monitors and rules
# ---------------------------------------------------------------------------

async def _confidence_monitor(candidate: dict[str, Any]) -> GammaEstimate:
    return GammaEstimate(gamma_hat=float(candidate.get("confidence", 0.0)))


async def _under_hard_cap(candidate: dict[str, Any]) -> bool:
    return float(candidate.get("amount", 0.0)) <= 1000.0


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

async def main() -> None:
    adapter = MAFAdapter()

    # Compile both gates once.
    confidence_gate = adapter.compile_pre_node(PreNode(
        name="purchase-confidence",
        description="Confidence must clear the robust Gamma threshold.",
        monitor=_confidence_monitor,
        assurance_basis=AssuranceBasis(f1_pre_commitment=True, f2_modification=F2Modification.FULL),
        gamma_threshold=ROBUST_GAMMA_DEFAULT_THRESHOLD,
    ))
    cap_gate = adapter.compile_invariant(Invariant(
        name="purchase-hard-cap",
        description="Amount must never exceed $1000, regardless of confidence.",
        rule=_under_hard_cap,
        assurance_basis=AssuranceBasis(f1_pre_commitment=True, f2_modification=F2Modification.FULL),
    ))

    # Both middleware instances are passed to the Agent at creation time.
    # MAF calls them in order before every approve_purchase invocation:
    #   confidence gate -> cap gate -> approve_purchase tool (only if both pass)
    async with (
        AzureCliCredential() as credential,
        Agent(
            client=FoundryChatClient(
                project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
                model=os.getenv("FOUNDRY_MODEL", "gpt-4o-mini"),
                credential=credential,
            ),
            name="PurchaseAgent",
            instructions=(
                "You are a purchase approval agent. "
                "Always call approve_purchase to process requests."
            ),
            tools=approve_purchase,
            middleware=[
                VSLFunctionMiddleware(
                    confidence_gate,
                    candidate_input_fn=lambda ctx: dict(ctx.arguments),
                    function_name="approve_purchase",
                ),
                VSLFunctionMiddleware(
                    cap_gate,
                    candidate_input_fn=lambda ctx: dict(ctx.arguments),
                    function_name="approve_purchase",
                ),
            ],
        ) as agent,
    ):
        scenarios = [
            "Approve a $250 purchase. Confidence score is 5.0.",
            "Approve a $250 purchase. Confidence score is 0.2.",
            "Approve a $5000 purchase. Confidence score is 5.0.",
        ]

        for request in scenarios:
            print(f"\nUser: {request}")
            try:
                result = await agent.run(request)
                print(f"Agent: {result.text}")
            except InvariantViolation as exc:
                print(f"  BLOCKED (hard cap): {exc.reason}")
            except AutomationDeniedException as exc:
                print(f"  DENIED (low confidence): {exc.reason}")


if __name__ == "__main__":
    asyncio.run(main())
