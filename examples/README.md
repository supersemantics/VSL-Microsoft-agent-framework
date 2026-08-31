# examples

Sample agents organized by domain.

## financial-agents

### `purchase_approval_agent.py`

Chains both VSL governance primitives against a purchase approval flow, wired
as `VSLFunctionMiddleware` on a Microsoft Agent Framework `Agent`:

- **PreNode** (`purchase-confidence`): soft, threshold-based -- the monitor's
  confidence signal must clear the robust Gamma threshold before a purchase is
  even considered.
- **Invariant** (`purchase-hard-cap`): non-bypassable -- the amount must never
  exceed $1000, regardless of how confident the monitor is.

Both gates run in registration order before `approve_purchase` fires; either
one raising means the tool never runs.

Needs real Azure/Foundry credentials -- this example creates a real `Agent`
backed by `FoundryChatClient`, not a mocked one:

```bash
pip install agent-framework azure-identity pydantic
export FOUNDRY_PROJECT_ENDPOINT=https://<your-service>.services.ai.azure.com/api/projects/<your-project>
export FOUNDRY_MODEL=gpt-4o-mini
az login
python examples/financial-agents/purchase_approval_agent.py
```

For gate behavior exercised without any real agent or API call, see
`tests/test_integration.py` -- it drives `VSLFunctionMiddleware.process()`
directly against a mock context.
