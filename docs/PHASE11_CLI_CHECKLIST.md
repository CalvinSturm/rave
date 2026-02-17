# Phase 11 CLI Checklist

Scope guardrails:
- CLI UX only.
- No pipeline behavior changes.
- No API redesign.
- No new abstraction layers.

## Checklist

- [ ] Help text is concise, accurate, and includes copy-paste examples.
- [ ] Structured `--json` output is explicit and test-covered.
- [ ] Human-readable default output is clear for non-JSON mode.
- [ ] `probe`/`devices` ergonomics are practical for operator workflows.
- [ ] Progress output is visible and script-friendly.
- [ ] CLI docs match actual command behavior.
- [ ] CLI tests cover help, JSON schema shape, and non-JSON output.

## Proposed PR Slices

1. PR11-1: Help + Structured JSON Contract (Done)
- Improve top-level help examples.
- Add explicit `--json` output mode to command flows.
- Keep JSON file output behavior (`--json-out`) deterministic.
- Add/adjust CLI tests for JSON + non-JSON output.
- Update README CLI quickstart.

2. PR11-2: Devices + Probe Ergonomics (Done)
- Add `rave devices` command for quick device listing.
- Improve `probe` output for operator troubleshooting.
- Add tests for `devices`/`probe --help` and JSON shape.
- Update docs with probe/devices runbook.

3. PR11-3: Progress Output UX (Next)
- Add opt-in progress output for long-running commands.
- Keep output deterministic and non-noisy.
- Add tests for flag wiring/help and progress mode behavior.
- Update docs for interactive vs scripted usage.

4. PR11-4: Docs + Consistency Sweep
- Ensure all scripts/docs use current CLI shape.
- Tighten wording, examples, and JSON schema notes.
- Final CLI polish pass with tests.
