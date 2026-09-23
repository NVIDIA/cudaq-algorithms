# Source Review

## Source ownership and freshness

Current public code and authoritative tests in the checked-out repository
control API behavior. [Source lookup](../../cudaq-algorithms/references/source-provenance.md) records common source/test/example
locations; [coverage history](../coverage/history.md) holds the historical review
anchor, an audit trail rather than an active contract or compatibility promise;
each record adds only contract-specific stable symbols and paths.

When a selected record differs from current source or tests:

1. compare the relevant public symbol and tests;
2. treat current public source as authoritative for generated code;
3. report the drift and evidence level;
4. update a maintained record only when that update is in scope;
5. never retain a stale line-number claim merely because the prose is familiar.

Do not copy historical repository hashes or dependency pins into primitive
records. Historical last-review values belong in coverage history. A validation or evaluation result instead records the exact revision and
dependencies actually used by that run.

Use line numbers only for a non-obvious invariant that benefits from a precise
anchor. Prefer stable symbol and test names for ordinary provenance.

## Historical review procedure

## Freshness check

Before implementing from a maintained record:

```bash
git rev-parse HEAD
git diff 61ac072d -- \
  python/cudaq_algorithms tests/python docs/sphinx \
  pyproject.toml .cudaq_version
```

Inspect only the selected symbols and their tests. If public signatures or
scientific assertions changed, follow current source for generated code, report
the drift, and update the maintained record only when that work is in scope. Do
not silently upgrade the lifecycle from `draft` or claim a newly verified
version range.
