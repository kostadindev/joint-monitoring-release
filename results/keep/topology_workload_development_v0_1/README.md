# Anonymous-topology workload monitoring: local development release candidate

This is a local complete-run candidate, not a publication or held-out result. The
project code is MIT; dataset redistribution licensing remains undeclared. It contains
self-authored simulated workloads and recorded judge calls, not MESA or external raw
incident data, credentials or provider caches.

## Strict access boundary

`allocator_inputs.jsonl.gz` is the ONLY per-example allocator data. Every row is
exactly `{graph: {nodes: [integers], edges: [[integer, integer], ...]}, k, s}`. No
roles, trust, names, message content/lengths, case IDs, mappings, targets, labels or
judge scores are present. Row order is bookkeeping, not a feature. Fixed algorithm
parameters and seeds come from the declared benchmark configuration. The optional
runtime learned model in allocator_parameters/ contains only topology features and
fixed coefficients; its full supervision audit is evaluation-only.

Both selection and grouping must finish from this input. A trusted acquisition
service then maps the anonymous selections/groups to recorded channels and provides
their full message histories to the judge. `acquisition_prompts` in the verifier
shows this transition and enforces the acquired-edge budget and disjoint group cap.
Do not pass its named records, private mappings or returned text back to an allocator.

`judge_data/observations.jsonl.gz` is the acquisition service's full message store,
NOT free allocator input. It contains recorded names/workflow/graph and public
messages (including shown protected-reference markings), without hidden labels.
The service must expose only the chosen channels. `evaluation/` contains private
mappings/indexes, labels, conditions, original records, plans and allocation outputs.
`analysis/` and `audit/` are also evaluation-only: raw calls and scores are not policy
features. Original work_<20 hex digits> IDs are opaque bookkeeping and preserved
in these private/evaluator files to reproduce source hashes exactly.

## Scientific scope

All 288 records are development variants of twelve original target cases, crossing
4/16 cases per workload, clean/deliberate-false/honest-error conditions, audit-copy
alignment and history. These are message interventions on one shared skeleton,
not 288 independent attacks or distinct topology families. Error/attack twins share
public views and judgments. Labels identify factual origins, not malicious intent.
The original five-case training library is separate development; the 512-token
training judge interface differs from this 1,024-token/citation interface. Training
IDs, source hashes, weights and frozen model are retained, not relabeled as held out.

Ordinary policies partition acquired edges once within k and cap s. Exact matched
random rows describe expectations over uniform partitions of the same selection
and size multiset. Candidate groups are a shared research library, not all executed
by one allocation. Their rational inclusion weights preserve edge and call costs.
24 KB is a declared input operating limit; 60 KB full-context references are separate
allowance classes. Infeasible assignments and judge abstentions remain explicit.

Protected references, content-based rule checking and pooled joins are available to
detectors AFTER the graph-only allocation. They may not feed keys into grouping.
Privileged location/allocation oracles are evaluator diagnostics, never prospective
policies. Full-context and rule baselines remain even when they outperform the LLM.

## Reproduction and cost

`python tooling/verify_topology_release.py RELEASE_DIRECTORY` checks hashes, strict
allocator schemas and mappings, public-label separation, exact prompts, budgets,
expected weights, source metrics/baselines, source/model hashes and reused-call
provenance. Add `--replay` to regenerate frozen allocations and complete analysis
with bundled execution/analysis sources and ledger snapshot. Replay needs the
project dependencies and makes no model calls. It uses temporary files; original
plan bytes remain unchanged and the ledger is supplied as an explicit override.

Reused calls retain their historical usage as standalone replay cost, with exact
source-file/line/record hashes and raw source records. Their incremental cost is
zero. New-run research spend comes from its filtered actual ledger, including
transport losses; never sum per-policy replay costs or bill reused views again.
Whole source-file hashes are archival provenance; selected raw-line hashes are
independently verifiable here without redistributing the entire superseded library.

Run sources, post-run analysis sources and training sources are separated. The
execution bundle also includes explicitly hashed support Python modules needed by
historical imports. Four transitive script imports and support package modules are
bound to the earlier local v0.1 manifest, retained as audit/support_source_manifest.json;
these dependencies were not all listed in the new execution plan's source_hashes.
Verification establishes stated integrity/reproduction checks,
not universal benchmark realism, rationale correctness or deployment security.
