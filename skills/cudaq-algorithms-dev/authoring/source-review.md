# Source Review

## Source ownership and freshness

Current public code and authoritative tests in the checked-out repository
control API behavior. [Source lookup](../../cudaq-algorithms/references/source-provenance.md)
records common source/test/example locations; each record adds only
contract-specific stable symbols and paths. Follow the
[coverage policy](../coverage/policy.md) for evidence claims. Archived reviews
are audit material, not current contracts or compatibility promises.

When a selected record differs from current source or tests:

1. compare the relevant public symbol and tests;
2. treat current public source as authoritative for generated code;
3. report the drift and evidence level;
4. update a maintained record only when that update is in scope;
5. never retain a stale line-number claim merely because the prose is familiar.

Do not copy historical repository hashes or dependency pins into primitive
records. Historical last-review values belong with the archived review. A validation or evaluation result instead records the exact revision and
dependencies actually used by that run.

Use line numbers only for a non-obvious invariant that benefits from a precise
anchor. Prefer stable symbol and test names for ordinary provenance.

## Freshness check

Before implementing from a maintained record:

```bash
git rev-parse HEAD
git status --short -- \
  python/cudaq_algorithms tests/python docs/sphinx \
  pyproject.toml .cudaq_version
```

Inspect only the selected symbols and their tests. If public signatures or
scientific assertions changed, follow current source for generated code, report
the drift, and update the maintained record only when that work is in scope. Do
not silently upgrade the lifecycle from `draft` or claim a newly verified
version range. If comparing against a previous review, obtain its actual
recorded revision from that review; do not assume a fixed historical hash is
the baseline for today's checkout.
