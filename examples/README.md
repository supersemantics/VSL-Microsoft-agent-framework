# examples

Sample agents organized by domain.

## financial-agents

### `purchase_approval_agent.py`

Chains both VSL governance primitives on a purchase approval flow, wired as
`VSLFunctionMiddleware` on a Microsoft Agent Framework `Agent`:

- **Invariant** (`purchase-hard-cap`): the amount must never exceed $1000.
- **PreNode** (`purchase-within-budget`): the purchase must leave headroom in the
  remaining daily budget. The monitor returns budget ÷ amount, a ratio where 1.0
  is break-even, and 0.0 for a missing amount so the gate fails closed.

Both gates run in registration order before `approve_purchase`. A denial aborts
the run with `VSLGateDenied`, so the example's own code sees it. Every gate
decision is recorded with `write_gate_decision()` in a JSONL ledger.

Neither gate involves a model at the guarded call, so both are F1 = True,
F2 = `NONE` (level LOW): they block what they deny, but they do not shape what
the model produces.

Needs real Azure/Foundry credentials, because the example creates a real `Agent`
backed by `FoundryChatClient`:

```bash
pip install "agent-framework-core>=1.25" agent-framework-foundry azure-identity
export FOUNDRY_PROJECT_ENDPOINT=https://<your-service>.services.ai.azure.com/api/projects/<your-project>
export FOUNDRY_MODEL=gpt-4o-mini
az login
python examples/financial-agents/purchase_approval_agent.py
```

Without credentials, `tests/test_real_agent.py` runs the same shape of gate
through a real MAF `Agent` with a scripted chat client, and
`tests/test_integration.py` drives `VSLFunctionMiddleware.process()` directly.
