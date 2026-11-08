# Development policy search: results and limits

All 9,152 planned views on the 51 original record-workflow episodes completed.
There are 118 format retries and 47 explicit terminal schema abstentions (0.51% of
logical views). Their distribution by group size is: two channels 9, three 10,
four 22, seven 5, eight 1. These are counts, not size-adjusted failure rates.
The shared ledger records 0.677856091 confirmed additional credits, plus 0.10 retained
for two unknown-charge 502 requests, totaling 0.777856091 against authorization.
No active reservations remain at this checkpoint. Provider summaries from the last
resumed process alone do not represent the entire research spend.

The full grid comprises 46,512 policy/episode evaluations. Additional control
analysis has 22,410 evaluations and the supervised/structural pair search 1,428.
These are repeated uses of a common judge library, **not independent samples**.
There are only three distinct attacked tasks and two additional clean tasks.
All numbers below use the previously declared strict score >0.5 rule.

## Main comparison on jointly activated attacks

Total edge budgets are 7/13 (flat), 8/15 (hierarchical), and 7/13 (sequential), with
maximum group size 2. Recall includes unselected direct-false edges. Random/chunk
numbers average 100 allocation seeds; bundle expectations enumerate every allowed
bundle subset and every odd-budget padding choice. No confidence interval based on
seed count is appropriate.

| Policy | Flat source recall | Hierarchical source recall | Sequential source recall |
|---|---:|---:|---:|
| Random edges, random pairs | 4.7% | 1.3% | 7.5% |
| MESA-static selection, provenance groups | 0% | 0% | 0% |
| Uniform protected receipt/report bundles | 50.0% | 66.7% | 50.0% |
| Public eligibility-role-prioritized bundles | 100% | 100% | 100% |
| Location oracle, attacked reports only | 0% | 0% | 0% |
| Location oracle plus each report's protected reference | 100% | 100% | 100% |
| Full graph in one view (larger k and s; reference only) | 100% | 100% | 83.3% |

**False accusations and costs must accompany that table.** Uniform bundles have
source precision 100%, 100%, 85.7% on the combined attacks. Their inspected
non-registry false-positive rates are 0%, 0%, 6.9%; their clean-case non-registry
false-positive rates are 0%, 0%, 2.8%. Role-prioritized bundles have combined-attack
precision 100%, 100%, 96%, and clean non-registry FPR 0%, 0%, 1.4%.
The sequential errors include accusations on honest summaries with unshown inputs.
See the manual audit; do not silently count propagated false content as direct origin.

Uniform bundles use about 2,710 / 2,838 / 2,795 input-plus-output tokens per combined
episode. Full graph uses about 1,929 / 2,177 / 2,073, but inspects more edges and allows
a larger group. Thus the short logs do not establish a practical token-cost advantage
for partitioning. The full graph's source precision is 66.7%, 85.7%, and 71.4% here.
It is neither automatically infeasible nor uniformly dominated.

The deterministic sender-receipt matching baseline has the **same recall as uniform
and role-prioritized bundles**, with zero source false flags in these completed
development cases and zero LLM calls. It uses the same inspected edges. This is an
important limit: the structured truth check does not require an LLM. Source matching
would still flag an honest originating error; these cases do not test intent inference.

## Selection versus grouping

For MESA-static selection, attacked-edge coverage is 0%, 88.9%, and 100% in the three
workflows. Hierarchical and sequential failures therefore illustrate the difference
between observing attack-bearing edges and observing their corroborating references.
This is a comparison to the documented six-feature static adaptation, not the full
MESA algorithm with clean execution probes or its published oracle-detector numbers.

Holding greedy provenance-bundle selections fixed, provenance grouping versus
exactly matched-size random grouping yields source recalls:

| Workflow | Provenance grouping | Matched random membership | Calls in both |
|---|---:|---:|---:|
| Flat | 55.6% | 7.9% | 4.00 |
| Hierarchical | 50.0% | 6.5% | 4.22 |
| Sequential | 55.6% | 11.1% | 4.00 |

The random column now enumerates every matched-size partition at each fixed selection
(19,530 allocations across all conditions), then averages the three selection seeds
equally. The original three random grouping draws gave 11.1%, 27.8%, and 11.1%; they
were unusually favorable to random grouping in the first two workflows and remain
archived. These averages use three selection seeds and three tasks. Group-size vectors match within
each paired comparison; tokens differ slightly because judge outputs differ. In
this constructed setting the input/output relationship encodes useful evidence.
Receiver grouping has zero source recall at these selections; spectral grouping
yields 22.2%, 50.0%, 11.1%. Generic graph proximity is not enough by itself.

## Tuning result and its failure conditions

Eligibility-role prioritization works best for the combined attack **because that
attack always targets age and policy window**. On single-source attacks to priority,
delivery zone, or price, its recall falls to 25%, 50%, 25%, versus uniform bundles'
50%, 66.7%, 50%. This is a useful distribution-shift warning, not a general winner.
For the current objective of broad source coverage, uniform bundle sampling is the
cleaner reference; a risk-prioritized policy requires an explicit attack/harm prior.

The ridge pair scorer was tested with three regularization strengths and three false-
accusation penalties, in exact-k and at-most-k forms. Every held-out development case
keeps all its variants and environments outside that fold's training data. The folds
are nevertheless reused to compare hyperparameters, so this is tuning. Some settings
score 66.7%, 66.7%, 83.3% on combined attacks, while others miss everything or choose
no observations. In the flat graph the protected bundles have identical public
features, so a 66.7% realization versus the exact uniform 50% expectation is seed
luck, not evidence of learned localization. Three attacked tasks cannot substantiate
learned generalization. Report the full grid and train/probe costs, not just its best
setting.

The retrospective allocation oracle obtains complete recovery with perfect precision
on these combined cases by choosing among real recorded judge views. It optimizes
TP minus twice FP, and is an upper bound for that additive reward **only within this
finite single/pair library and budget**. It is not a deployable policy, a universal
judge ceiling, or the correct oracle for every other reward weight.

## Next decision

Additional strong baselines qualify the group-cap interpretation. On the same random
selected edges, a deterministic cross-group structured join obtains 31.3%, 23.3%,
31.3% source recall with no false source flags, versus the grouped LLM's 4.7%, 1.3%,
7.5%. It preserves extracted facts instead of fusing only suspicion scores; this
alternative architecture has no LLM context requirement in the structured setting.
Do not claim group co-location is necessary for every possible monitor.

Full-selected context at the **same k and same greedy bundle selections**, allowing
s=k, gives 55.6%, 50.0%, 61.1% recall at about 1,269 / 1,396 / 1,368 tokens, versus
2,730 / 2,965 / 2,814 for provenance groups. Source precision is 76.9%, 75.0%, 91.7%
for full-selected and 100%, 81.8%, 90.9% for grouped views. This is a measured cost
trade-off, not strict universal dominance. Full graph remains a separate larger-k
reference. None of the 9,152 scored views has a source score exactly 0.5, so changing
the source rule to >=0.5 leaves its results unchanged.

Abstention rates by group size are 0/697 (1), 9/4,437 (2), 10/833 (3), 22/2,434 (4),
0/130 (5), 0/94 (6), 5/272 (7), 1/187 (8), 0/17 (12), 0/34 (13), and 0/17 (15).
There is no monotonic size effect in this small library. The main combined-attack
bundle and exact matched-membership results have no abstained true-source edges;
excluding abstentions or taking the optimistic recall bound does not change those
recalls. Other configurations retain explicit abstention counts and bounds.

Keep these results as a positive-control benchmark and a manual judge-error study.
The next development stage should balance target roles, include missing-context and
honest-error negatives, vary whether graph structure reveals the relevant evidence,
and introduce a verified three-channel evidence family. A separate workload scaling
arm must make full context actually exceed the declared allowance before a practical
budget-limited claim. Fresh held-out cases remain unused. See
`manual_data_review_2026-11-06.md` for concrete examples and the tuning proposal.
