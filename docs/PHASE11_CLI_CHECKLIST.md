# Phase 11 CLI Checklist

Scope guardrails:
- CLI UX only.
- No pipeline behavior changes.
- No API redesign.
- No new abstraction layers.

## Checklist

- [x] Help text is concise, accurate, and includes copy-paste examples.
- [x] Structured `--json` output is explicit and test-covered.
- [x] Human-readable default output is clear for non-JSON mode.
- [x] `probe`/`devices` ergonomics are practical for operator workflows.
- [x] Progress output is visible and script-friendly.
- [x] CLI docs match actual command behavior.
- [x] CLI tests cover help, JSON schema shape, and non-JSON output.
- [x] JSON contract versioning is explicit for parsers.

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

3. PR11-3: Progress Output UX (Done)
- Add opt-in progress output for long-running commands.
- Keep output deterministic and non-noisy.
- Add tests for flag wiring/help and progress mode behavior.
- Update docs for interactive vs scripted usage.

4. PR11-4: Docs + Consistency Sweep (Done)
- Ensure all scripts/docs use current CLI shape.
- Tighten wording, examples, and JSON schema notes.
- Final CLI polish pass with tests.

5. PR11-5: JSON Contract Versioning (Done)
- Add `schema_version` to all CLI JSON payloads and progress JSONL.
- Add parser-focused tests for `probe --json` / `devices --json`.
- Document schema/version expectations in README.

## Phase 11 Closeout (PR11-6)

Status: Completed on 2026-02-18 (docs + release hygiene only).

PR slice links:
- PR11-1: https://github.com/CalvinSturm/rave/tree/phase11-cli-pr1-help-json (head: https://github.com/CalvinSturm/rave/commit/6fcb9fa)
- PR11-2: https://github.com/CalvinSturm/rave/tree/phase11-cli-pr2-devices-probe (head: https://github.com/CalvinSturm/rave/commit/4bfdda5)
- PR11-3: https://github.com/CalvinSturm/rave/tree/phase11-cli-pr3-progress-ux (head: https://github.com/CalvinSturm/rave/commit/f22fd52)
- PR11-4: https://github.com/CalvinSturm/rave/tree/phase11-cli-pr4-consistency-sweep (head: https://github.com/CalvinSturm/rave/commit/693d339)
- PR11-5: https://github.com/CalvinSturm/rave/tree/phase11-cli-pr5-json-schema-contract (head: https://github.com/CalvinSturm/rave/commit/44cf2d0)
