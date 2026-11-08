# Supplemental workload contrast audit after independent review

This addendum was specified after workload dispatch began and after the small
format/cost pilot was inspected, but before full-library aggregate contrasts were
unblinded. It responds to Claude's frozen-workload review. It is an exploratory
post-launch supplement, not a retroactively preregistered primary analysis. The
frozen judge, planner, assignments, core analyzer and original reported metrics are
unchanged. No model calls, policy tuning, threshold changes, resampling, outcome-based
exclusions or held-out data are involved.

The implementation is `scripts/audit_workload_contrasts.py`, with independent tests.
It writes only `supplemental_contrast_audit.json` inside the completed run directory.
Its output records hashes of its own source, this addendum, the frozen plan, records,
assignments, exact-random plans and completed calls. The release should retain this
source/protocol separately from the original frozen-run source snapshots.

## Full coverage before reporting

Require exact coverage of every planned feasible ordinary and exact-random view,
with no duplicate or unplanned completed keys, before any supplemental result is
written. For this frozen workload that means 9,647 unique views. Check assignment
hashes, counts, view-key hashes and frozen source hashes. Recompute only the primary
operational rows using the unchanged frozen implementation, writing its temporary
files to a temporary directory so that original metrics and `primary_contrast.json`
remain untouched.

For the primary k=7, s=2, workload-16, 24 KB comparison, require all twelve target
clusters, both aligned/shuffled routes, clean/false-report conditions, stable/updated
histories, seeds 0/1/2, and both provenance and exact-matched-random groupers. These
are 576 assignment/expectation rows. `provenance` is the code name for adjacency
(or protected-adjacency) grouping. Require zero source-truth edges in clean worlds
and exactly one in false-report worlds. Assert matching source coverage and reference
availability for paired grouping arms. Honest-error twins add no independent judge
information and are excluded from these contrasts.

## Complete error accounting and abstention bounds

Within each target cluster and route, average histories/seeds within each condition.
Report each arm's operational true positives, **clean-world and attacked-world false
positives**, abstained truth edges, and abstained negative edges. Registry/audit
channels are negative source-attribution opportunities alongside benign reports.

Report FPR using all three denominators, preserving their distinctions:

- Selected-negative FPR: false flags divided by inspected non-origin channels.
- Network-wide benign FPR: the same false flags divided by all graph channels minus
  source-truth channels. Uninspected benign channels can dilute this rate, so it
  must not replace selected-negative FPR.
- Report-channel FPR: false flags on inspected benign specialist report channels
  divided by those channels, excluding registry/audit and downstream decision edges.

For a row with T detected truth edges, F false flags, U_T abstained truth edges and
U_F abstained selected-negative edges, use T in [T,T+U_T], F in [F,F+U_F]. FPR upper
bounds add abstained negative edges relevant to the respective denominator; network
bounds add only *inspected* negative abstentions. Unknown uninspected edges are not
invented judge decisions. The TP−2FP interval is
`[T−2(F+U_F), T+U_T−2F]`. Average clean and false-report utility with equal class
weight after averaging histories/seeds within target case.

For every adjacency-minus-random contrast, conservative bounds are
`[adjacency_lower−random_upper, adjacency_upper−random_lower]`. Apply the same
interval arithmetic to the unconditional alignment interaction. These are bounds
under unresolved judge outputs, **not sampling confidence intervals**. Report the
per-case values, mean bounds and case-sign counts; do not introduce a confirmatory
p-value or treat repeated seeds, histories or views as independent sample size.
Positive recall/utility differences favor adjacency; negative FP differences favor
adjacency. A utility winner must not be selected solely from operational scores
that reward abstention as though it were reliable no-alert behavior.

## Mechanism and stricter citation support

Alongside judge differences, report selected-source coverage, reference-pair
availability, co-observation, within-group rule TP/FP, and pooled record-join TP/FP.
Also show the difference between the LLM recall contrast and rule-grouping contrast
as a descriptive residual. Routing changes which reference is available under a
fixed graph-only selection; the interaction is not a pure judge effect. Do not
bootstrap a conditional interaction between different pair-available subsets.

The original `message_supported_tp` metric is retained unchanged. A supplementary
**strict relational citation** TP requires all of the following in the exact judge
view and that flagged channel's citation list:

1. A cited actual false-origin message on that flagged channel.
2. A cited trusted reference message.
3. Publicly parsed claims sharing the same case, field and revision, with different
   values. References from another case, field or revision cannot support this TP.

This is a known-grammar corroboration check, not certification of the entire free-
text rationale. It does not resample a valid primary score if citations fail. For
exact-random policies, apply the frozen inclusion weights to these additive counts;
scoring all candidate views for research does not turn them into one giant policy.
Count citation validity **among flag-bearing views only**, and show numerator,
denominator and rate. Missing empty citation objects on unflagged views do not
inflate this conditional denominator.

## Unique-view reliability by group size

Report judgments, abstentions, format retries, input/output tokens and logical replay
credits by the number of channels actually shown in each unique completed judge
view. This is not weighted by how many policies reuse the view. Record attempts at
or above their recorded output-token cap, distinguishing attempts from logical views.
This directly checks whether the workload's 1,024-token schema still disadvantages
larger groups; the earlier 512-token diagnostic is not proof either way. Output-at-cap
is a truncation diagnostic, not by itself proof of why a particular judgment failed.
Citation validity among flagged views is reported here too. Logical replay costs can
include cached responses and exclude uncertain transport failures; the authoritative
shared ledger remains the research-spend source.

## Independent background-overlap audit

The reviewer raised possible cross-cluster reuse of background cases. The actual
frozen generator uses `make_case(10000 + index*100 + j)` for background j, with target
indices 0–11 and j=0–14 at size16. The frozen records independently verify:

- 288 episode variants and twelve distinct target case IDs;
- sixteen distinct case IDs within each target cluster (including its target);
- 192 distinct case IDs across the twelve clusters;
- **no case-ID intersection between any pair of target clusters**.

The size4 background cases are a subset of the size16 cases within the same target
cluster, and conditions/routing/history intentionally share them. This establishes
identity separation across target clusters, not distributional independence from a
shared synthetic generator, grammar, model or judge. Small-N uncertainty and the
controlled-environment limitation remain.

Verified frozen episode file SHA-256:
`58f59c20a48543ce3265c3d985dcf9ea25a58255ae47bafe7967b6788c9ba4dd`.

Run only after the complete view library exists:

```sh
python scripts/audit_workload_contrasts.py results/record_workload_development_v1
python -m pytest tests/test_audit_workload_contrasts.py -q
```
