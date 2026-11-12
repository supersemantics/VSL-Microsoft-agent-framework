# Changelog

What changed in each release, and why. Newest first.

## 0.2.1 (2026-11-12, tag `v0.2.1`): vsl-core pinned to v0.3.0

**Before:** vsl-core was installed from `git+https://github.com/supersemantics/VSL-Core.git`
with no version, so an install took whatever vsl-core's `main` was that day. An application could
not pin vsl-core itself either: pip refuses two different URLs for one package, so pinning
`@v0.3.0` next to this adapter failed to install.

**After:** vsl-core is pinned to its release tag, `...VSL-Core.git@v0.3.0`. An application that
also installs vsl-core names the same URL, tag included.

**Why:** reproducible installs; tested against vsl-core 0.3.0.

## 0.2.0 — denials abort the run; vsl-core 0.2.0 guidance

### Fixed: a denial now reaches the caller of `agent.run()`

**Before:** `VSLFunctionMiddleware` raised the gate's `AutomationDeniedException`. In agent-framework 1.x, MAF's function-invocation loop turns an ordinary exception from function middleware into a tool-error result and keeps running. The tool did not run (F1 held), but `agent.run()` returned normally: the model was told "Error: Function failed." and could retry or carry on, and the README's `try/except AutomationDeniedException` around `agent.run()` never fired. Checked against agent-framework-core 1.25.0 with a real `Agent` and a scripted chat client: a denied $5000 purchase returned a normal result, and the application never saw the denial.

**After:** by default (`on_denial="abort"`) a denial raises `VSLGateDenied`, a subclass of MAF's `MiddlewareFailure`, which MAF propagates to the caller instead of absorbing. `denied.denial` is the original `AutomationDeniedException` or `InvariantViolation`, and `denied.function_name` is the tool that did not run. A gate that fails for any other reason also aborts with `MiddlewareFailure`, so the tool never runs unguarded. `on_denial="return_to_model"` keeps MAF's tool-error behaviour for callers who want the model to explain the refusal.

**Why:** a governance gate the application cannot observe is fail-open at the level of the run, and the documented `except` never fired.

**Breaking:** code that caught `AutomationDeniedException` around `agent.run()` should catch `VSLGateDenied` and read `.denial`. That `except` never fired on agent-framework 1.x anyway.

### Changed: brought in line with vsl-core 0.2.0 and its guidance

- **F2 labels.** The example and the README labelled no-model gates (a confidence gate, a hard cap) `F2Modification.FULL`. F2 is judged at the guarded call, and a gate that only allows or blocks is `NONE` (LOW); they are now `NONE`. The README has the F1/F2 table and the "level is not enforcement" note.
- **Gamma values.** The README example passed a raw confidence with `gamma_threshold=1.0`, and the sample passed a model-supplied 0–10 confidence as `gamma_hat`. Both now use a ratio where 1.0 is break-even (limit ÷ amount, budget ÷ amount), fail closed with 0.0, and keep the threshold above 1.0. A "Writing gate values honestly" section states the rules.
- **Gate-decision records.** The ledger section told readers to hand-write MONITOR/PRE_NODE/VERIFICATION entries. It now uses `write_gate_decision()` from inside the monitor or rule, and the sample records every decision in a JSONL ledger.
- **Handovers.** A new multi-agent section shows `write_handover()` and `caused_by` for agent-to-agent hand-offs (`agent.as_tool()`, workflows). That part needs vsl-core 0.3.0.
- **Requirements.** On import, vsl-maf refuses vsl-core older than 0.2.0. The dependency is now `agent-framework-core>=1.25.0` rather than the full `agent-framework` metapackage.
- The README no longer links a separate hosted ledger package; it names vsl-core's own stores.

### Added

- `MAFAdapter(identity_key=...)`: denials carry the agent's identity instead of the fixed `"vsl-maf"`.
- `tool_arguments(context)`: the tool call's arguments as a dict, whether MAF passes a mapping or a pydantic model.
- The F2 heuristic also flags MAF chat clients (`FoundryChatClient`, `OpenAIChatClient`, ...) and Azure AI inference behind a `FULL` claim.
- `tests/test_real_agent.py`: end-to-end through a real MAF `Agent` and its function-invocation loop, no credentials.

**Tests:** 22 pass with agent-framework-core 1.25.0, against both vsl-core 0.2.0 (`main`) and 0.3.0: 1 conformance, 15 against a mock context, 6 through a real `Agent`. One of the real-Agent tests pins MAF's middleware order (list order). Three of the real-Agent tests fail against 0.1.0, because the denial never reached the caller: the bug above. Not tested: streaming runs, where MAF documents that `MiddlewareFailure` is raised when the stream is consumed.

## 0.1.0

Initial implementation: `MAFAdapter` and `VSLFunctionMiddleware`.
