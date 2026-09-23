# Native Codex telemetry calibration

Date: 2026-09-11

## Final-controller integration calibration

After the independent review fixes, the controller ran a fresh native
`gpt-5.5` low session with the final restrictive profile, disabled unrelated
apps/plugins/hooks/skills, and one real shell call. It exited zero in 7.109 s
and reported `CALIBRATION_OK` with startup skill names `NONE`.

Final native usage: input 13,108; cached input 5,632; output 104; reasoning
output 0; inclusive total 13,212. Three native request-completion records gave
an observed input peak of 6,625. The extra unclassified input-only completion
was 5,586 tokens, so the request-sum cross-check failed by exactly that amount.
The corrected collector marked final totals complete, peak coverage unverified,
and compaction counts unavailable (`null`). No request sums replaced the
authoritative final counters.

Durable evidence is in `../results/e2e-live-calibration/`: raw native JSONL,
command, final text, execution timing, sanitized telemetry events and summary.
This calibration is excluded from the 150 planned scientific attempts.

## Result

`TelemetryCollector` uses invocation-only native OpenTelemetry log export while
retaining `codex exec --ignore-user-config`. It binds an ephemeral
`127.0.0.1` port, accepts JSON only at `/v1/logs`, and closes the listener and
accepted uploads before the context exits. The retained event list contains only token counts,
measurement source, and compaction markers. Raw OTLP records, prompts, tool
commands/results, authentication fields, environment values, conversation
identifiers, and model text are absent from the retained metric events and
summaries. Native event identifiers are used only in private, in-memory
deduplication keys.

This follows the official OpenAI [configuration reference](https://developers.openai.com/codex/config-reference),
which defines invocation-overridable `otel.exporter` and
`otel.log_user_prompt`, and the [advanced configuration telemetry guide](https://developers.openai.com/codex/config-advanced/#observability-and-telemetry),
which documents token counts on `codex.sse_event` `response.completed` logs
and states that raw prompt export is opt-in. The calibration explicitly set
`otel.log_user_prompt=false`, and disabled separate OTel metric and trace
exporters.

## Test-first evidence

Literal tests were written before `telemetry.py`. The initial command

```text
python3 -m unittest skills/cudaq-algorithms/evals/e2e/tests/test_telemetry.py
```

failed six tests because the implementation was absent. After the minimal
implementation, all six passed when loopback binding was allowed outside the
workspace sandbox. The live calibration then exposed an OTLP/cumulative
counter mismatch. A seventh literal regression test first failed because a
mismatched cross-check was labelled complete, then passed after completeness
was corrected. An eighth red-green test ensures distinct identifier-free exec
compaction events are not collapsed while identified app-server replays are
deduplicated. A ninth verifies that equal OTLP timestamps with different token
counts remain distinct. The final verification command and count are recorded
in the handoff, rather than inferred from these earlier runs.

The tests cover:

- cumulative snapshots replacing earlier snapshots instead of being summed;
- cached input remaining a subset of input and reasoning output remaining a
  subset of output (`total = input + output` only);
- absent fields remaining `null` with a field-specific reason;
- replayed OTLP records and compaction notifications being deduplicated;
- prompt/tool/environment-bearing records being discarded;
- loopback-only endpoint configuration, path rejection, and listener closure;
- cumulative totals remaining authoritative when the two native channels do
  not agree.

Independent review added regressions for interrupted streams, missing final
fields, larger intermediate counters, incomplete counter comparisons,
compaction aliases, unverified compaction transport, and an accepted HTTP
upload that otherwise completed after context exit. The accounting regressions
failed against the original implementation. The shutdown regression received
HTTP 200 after context exit before the fix and now verifies that late upload
cannot change the saved measurements. All 18 telemetry tests passed with:

```text
python3 -B -m unittest skills/cudaq-algorithms/evals/e2e/tests/test_telemetry.py
```

Loopback permission was allowed for the two local HTTP tests. This review
made no additional native model calls and read no authentication files.

## Authorized native calibration

One fresh `codex-cli 0.144.4` run used the existing native login, model
`gpt-5.5`, low reasoning effort, `--ephemeral`, `--ignore-user-config`, and the
previously preflighted restricted worker permission profile. The worker made
exactly one real shell tool call. Raw JSONL and OTLP text were held only in
memory; only the following measurement metadata was printed.

The native OTLP records actually observed all four installed request-level
token attributes:

| response completion | `input_token_count` | `cached_token_count` | `output_token_count` | `reasoning_token_count` |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 7,473 | 0 | 0 | 0 |
| 2 | 8,490 | 6,656 | 38 | 0 |
| 3 | 8,577 | 7,680 | 8 | 0 |

The normal final `turn.completed` JSONL usage event observed:

| field | value |
| --- | ---: |
| `input_tokens` | 17,067 |
| `cached_input_tokens` | 14,336 |
| `output_tokens` | 46 |
| `reasoning_output_tokens` | 0 |
| inclusive total derived as input + output | 17,113 |

The second and third OTLP completions sum exactly to every final JSONL field.
The first completion is an additional input-only native response and makes the
sum of all OTLP completions exceed JSONL by 7,473 input/total tokens. The OTLP
record does not expose a request-purpose or accounting field that safely
identifies this event, so the collector does not guess that it is a warm-up or
silently drop it. Instead:

- cumulative attempt totals come from the final native JSONL snapshot;
- request-level OTLP records provide an observed input peak of 8,577 tokens;
- the cross-check is marked incomplete/mismatched with the exact difference;
- the full JSONL stream contained zero observed compaction markers; the actual
  compaction count remains unavailable because this transport's notification
  coverage is not established.

The extra record did not change this calibration's peak because 7,473 is below
8,577. In another run an unclassified input-only native response could affect
the all-response observed peak. That transport limitation must remain visible
in benchmark reporting.

## Completeness and limitations

Available in this installation and observed during calibration:

- cumulative input, cached input, output, and reasoning output tokens;
- per-response input, cached input, output, and reasoning output tokens;
- inclusive totals computed without double-counting subsets;
- an observed (not inferred) per-response input-token peak;
- observed compaction markers when an event source supplies them;
- duplicate suppression and an explicit cumulative/request counter check.

Not exposed by this native OTLP log transport:

- a request-level total field (the collector derives it only when both input
  and output are present);
- a request-purpose/accounting marker for the extra input-only completion;
- a verified complete compaction-notification series from `exec --json`;
- billed dollars;
- a guaranteed semantic distinction between model context occupancy and raw
  API input tokens.

The collector therefore never substitutes cumulative input, model context
capacity, or zero for a missing request peak. Any missing token field remains
`null` and appears in `missing_reasons`.

`totals` now uses only the final native exec JSONL completion. Without that
event every final total is `null`; a missing final field stays `null` even if
OTLP emitted it. `observed_request_totals` retains the separate OTLP sums and
`observed_cumulative_totals` retains the latest native snapshot for diagnosis.
These observations are not substitutes for unavailable final attempt totals.
`native_turn_completed` explicitly records completion-event presence.

The numerical `observed_peak_request_input_tokens` is retained even when
coverage cannot be verified. Its completeness flag requires a completed turn,
an input count on every received response, matching cumulative input/output
counts, and no discrepancy among comparable counters. `peak_observation`
separately reports the number of received responses and inputs present. Even a
matching cross-check verifies accounting consistency, not the internal context
of unexposed model operations. Full cumulative-crosscheck completeness requires
all five counters to be comparable; matching only the available subset does
not establish full completeness.

Read-only inspection of the installed CLI's serialized exec event and item
variants did not establish a compaction event in that JSONL route. Internal and
app-server compaction variants do not prove exec emits them. Consequently,
`compactions` remains `null` and its completeness flag false by default, while
`observed_compactions` counts received markers. A caller may explicitly set
`compaction_notifications_supported=True` only after establishing that its
native source supplies every compaction; it must also reach turn completion
before a complete count is reported. Modern identified compaction items replace
deprecated same-turn aliases, and distinct item IDs remain distinct.
