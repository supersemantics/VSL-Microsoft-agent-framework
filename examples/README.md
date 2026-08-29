# examples

Sample agents organized by domain.

## financial-agents

### `purchase_approval_agent.py`

Chains both VSL governance primitives against a purchase approval flow:

- **PreNode** (`purchase-confidence`): soft, threshold-based — the monitor's
  confidence signal must clear the robust Gamma threshold before a purchase is
  even considered.
- **Invariant** (`purchase-hard-cap`): non-bypassable — the amount must never
  exceed a fixed cap, regardless of how confident the monitor is.

**Part 1** is LLM-free (no API keys needed) and exercises three scenarios
against `kernel.invoke()` with both gates wired as `VSLFunctionFilter`s:

| scenario | amount | confidence | outcome |
|---|---|---|---|
| approved | $250 | 5.0 | `APPROVED` |
| blocked by low confidence | $250 | 0.2 | `AutomationDeniedException` |
| blocked by hard cap | $5,000 | 5.0 | `InvariantViolation` |

**Part 2** shows the identical gate wiring used with a `ChatCompletionAgent`.
A `ScriptedChatCompletion` mock service drives the agent offline; swap it for
`AzureChatCompletion()` or `OpenAIChatCompletion()` to use a real service.

```
python examples/financial-agents/purchase_approval_agent.py
```
