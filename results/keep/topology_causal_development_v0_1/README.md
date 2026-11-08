# Local causal topology-monitoring development candidate

This private local candidate contains self-authored synthetic tasks, native actor
outputs, controlled private intervention instructions, and actual judge responses.
It is not a publication, held-out benchmark, or evidence of a universal winning
strategy. Every predeclared world and unknown is retained; no activation filter.
The frozen plans retain exact actor and judge model/settings; this extension uses
a 512-token actor cap and a uniform 2048-token judge cap across arms, separate from
the earlier 1024-token workload judge. No cross-configuration judge reuse is claimed.
The four underlying cases are crossed with three causal topologies. Each suite has
clean/A/B/AB native worlds and an observational honest-error AB twin; twins add no
execution or independent sample. Case-order counterbalancing does not estimate a
within-case order effect. A joint action flip does not imply individually invisible
false fragments; inspect clean/single-intervention correctness and source masks.

`allocator_inputs.jsonl` is the ONLY allocator input. Each row contains exactly an
anonymous integer-node graph plus k and s. No names, roles, text, case/record IDs,
labels, trust, private instructions or outcomes are provided. Row positions are
bookkeeping, never features. Seeds/algorithm settings are declared separately.
Both selection and grouping finish before acquisition. `judge_data/observations.jsonl`
is a trusted acquisition service's full store, NOT free allocator input; expose only
acquired channel histories. `evaluation/` holds mappings and labels, including the
positive/known-benign/unknown source partition. Unknown channels are not benign.
`private/generation/` contains full native authorities and private actor prompts;
`audit/monitor/` contains frozen judge plans, exact prompts, retries, raw responses
and benchmark metrics. Neither directory is an allocator feature source.

Full-network and deterministic matching references remain. Diagnostic views use
privileged potential locations and are labeled separately from topology-only
policies. Exact-random rows are expectations over disjoint allocations with matched
sizes, not policies reading all candidate groups. Native malformed/empty outputs,
noncompliance, unknown provenance, model errors, judge abstentions and infeasible
allocations remain explicit. Source truth covers structured facts and declared
computations; prose is unverified, and source origin does not establish intent.
Original native records and annotations remain immutable in `private/generation/`.
The judge evaluation uses a separately frozen strict-domain annotation correction:
noncanonical decision strings are UNKNOWN, even when a human recognizes a synonym.
`annotation_correction.json` binds the original artifacts, adapter and corrected
records. `secondary_alias_actions.json` is a separately labeled posthoc symmetric
action-only sensitivity; it never repairs malformed JSON or changes source labels.
Ordinary string `reason` metadata remains unverified prose and does not invalidate
the deterministic checker's supported structured facts.

Run `python tooling/build_topology_causal_release.py --verify CANDIDATE` to verify
file hashes, the access boundary, frozen generation replay, exact allocations and
judge prompts/configuration/raw-response lineage, metrics and actual ledger cost.
This needs the project's Python dependencies (recorded in manifest.environment, not credentials) and makes no model
calls. Frozen sources are overlaid into temporary runtime directories; original
plan bytes stay intact. Only the generation directory's physical location is
relocated during replay. Original directory strings are retained as provenance.

Actual spending is the filtered shared-ledger snapshot, with confirmed charges,
retained uncertainty and reservations separate. Native predecessor reuse and
judge-view reuse do not create new charges; never sum policy replay usage as spend.
The snapshot includes unreferenced durable attempts/errors as well as scored views.
Finalized durable artifacts are included; a lost transport response cannot be
reconstructed from an uncertain reservation. This candidate is local only and has
not been published. Dataset redistribution licensing remains undeclared.
