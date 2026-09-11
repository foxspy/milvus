# Strict group-by with per-group Search

For strict group-by requests, selecting the first topK group labels can be cheap while filling their remaining result slots through a global ANN iterator can scan many unrelated candidates. This change replaces the acceptance-probe and union-filter iterator optimization from #53306 with a configurable ordinary Search per selected label.

## Configuration and migration

| QueryNode setting | Default | Meaning |
| --- | --- | --- |
| `queryNode.groupBy.enable_search_path` | `false` | Enable per-group Search for eligible strict group-by requests. |
| `queryNode.groupBy.enable_search_path_k` | `1` | Positive minimum outer topK (number of groups), inclusive. |

Both settings are refreshable. QueryNode applies them after the query hook and freezes their values into each serialized plan. Caller/hook values cannot override the server settings. The C++ parser consumes these controls before sending search parameters to the index.

`strictGroupAcceptanceThreshold` and `strictGroupProbeCandidates` are retired. Remove them from deployments and use the settings above. The old union-filter iterator path is removed. With the new switch disabled, searches use the original group-by iterator through completion. Old serialized JSON control keys are discarded. No protobuf field or persisted index format changes.

The new path applies to strict group-by with one query vector, group_size > 1, supported scalar labels and row-level vector search. JSON group-by and other requests retain iterator execution. The threshold compares topK, not group_size.

## Execution

1. Consume the original iterator until it selects topK labels. Preserve its accepted candidates and lock the selected labels.
2. Prepare membership for these labels. Raw scalar fields are scanned once into per-label logical row lists; scalar indexes are pinned and queried with In(1) or IsNull for each label. Preserve user filters and timestamp/deletion visibility.
3. Search each selected label sequentially for the complete group_size, including labels already full in phase one. Phase-one rows remain eligible. Reuse the existing sealed index, sealed raw or growing provider, including nullable vector mappings.
4. Merge both phases by metric, deduplicate logical offsets and keep the best group_size hits per label. If approximate Search underfills, continue the original iterator to fill missing quotas. Errors and cancellation propagate; they are not converted into a successful fallback.

## Search parameters

Each internal Search sets topk=group_size and round_decimal=-1. It clears grouping, iterative-filter and iterator-v2 state and stale materialized-view filter metadata. It removes ef, search_list_size and search_list, so the backend chooses its default search budget for the new k. Growing/temp index parameter construction must preserve this policy after rebuilding its own search parameters. Metric, trace and necessary metric-specific context such as BM25 avgdl are retained. Backend algorithm selection remains automatic unless explicitly provided in the retained ordinary search parameters.

## Costs and validation

Raw membership needs O(M) extra row-offset storage, where M is the number of visible rows in the selected labels. Sequential execution keeps bitmap memory O(N), rather than O(topK*N); filling a bitmap per label still entails O(topK*N) bit work. No latency or recall improvement is claimed without measurements on the target workload. The feature defaults off.

Regression coverage includes configuration snapshots and validation, actual plan parsing and threshold boundaries, full-group replacement, metric direction, deduplication, underfilled fallback, cancellation/error propagation, nullable scalar membership, nullable vector offset mapping, HNSW/IVF searches, growing indexes and deletion/reinsert across multiple raw chunks. Tests use the branch's Knowhere backend; Cardinal deployment performance remains to be measured separately.
