"""Cost-free development controls using the common all-pair judge library.

Enumerate uniform/role-prior bundle selection exactly, compare 100 random seeds,
and distinguish attack-location knowledge from allocation-oracle knowledge. No
new judge calls, no threshold tuning, no claims of independent validation.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from record_strategies import policies, select, group
from record_allocation import view_key, policy_scores, measure, summarize, rates
from record_pair_search import load_views
from record_policy_extensions import enumerated_bundle_policies, extended_metrics


def make_policy(selector,grouper,k,s,seed,groups):
    return {'selector':selector,'grouper':grouper,'k':k,'s':s,'seed':seed,'groups':groups,
            'selected':sorted(e for g in groups for e in g),'group_sizes':list(map(len,groups))}


def controls(rec):
    es=sorted(map(tuple,rec['graph']['edges']));n=len(es);k=math.ceil(n/2)
    # Main grid at the half-edge comparison point plus full-network reference
    # policies. Larger groups at other budgets remain in the full grid output.
    for p in policies(rec):
        if p['k']==k and p['s']==2:yield p
        elif p['k']==n and p['seed']==0 and p['selector']=='random' and (
            (p['s']==1 and p['grouper']=='singleton') or (p['s']==n and p['grouper']=='all_selected')):yield p
    yield from enumerated_bundle_policies(rec,k)
    yield from enumerated_bundle_policies(rec,k,True)
    truth=set(map(tuple,rec['labels']['direct_false_edges']))
    for seed in range(100):
        selected=select(rec,k,'random',seed)
        yield make_policy('random_100','random_chunks',k,2,seed,group(rec,selected,2,'random_chunks',seed))
        # Privileged baseline: knows direct attack locations, but not which
        # other channels or pairings supply useful context. On clean cases the
        # same rule reduces to random allocation; its FPR is not a deployable rate.
        rng=random.Random(seed);rest=[e for e in es if e not in truth];rng.shuffle(rest)
        selected=sorted(truth|set(rest[:k-len(truth)]))
        yield make_policy('location_oracle_random_context','random_chunks',k,2,seed,
                          group(rec,selected,2,'random_chunks',seed))
    if truth:
        yield make_policy('location_only_oracle','attacked_channels_only',k,2,0,[sorted(truth)])
        bundles=[[('registry',e[0]),e] for e in sorted(truth)]
        yield make_policy('location_reference_oracle','protected_pairs',k,2,0,bundles)


def summarize_extended(rows):
    summary=summarize(rows);buckets=defaultdict(list)
    keys=('environment','condition','selector','grouper','k','s')
    for r in rows:buckets[tuple(r[k] for k in keys)].append(r)
    for item in summary:
        rs=buckets[tuple(item[k] for k in keys)]
        for metric in ('source_nonregistry','rule_source','rule_false_origin','llm_false_origin','global_structured_join_source','global_structured_join_content'):
            item[metric]=rates({k:sum(r[metric][k] for r in rs) for k in rs[0][metric]})
        for key in ('incident_alert','uncertain_source','used_edges','reference_detected'):
            item['mean_'+key]=sum(r[key] for r in rs)/len(rs)
        item['mean_source_utility_p2']=sum(r['source']['tp']-2*r['source']['fp'] for r in rs)/len(rs)
        coobserved=sum(r['reference_coobserved'] for r in rs)
        item['detected_given_reference_coobserved']=sum(r['reference_detected'] for r in rs)/coobserved if coobserved else None
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--views',nargs='+',required=True)
    p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    recs=[json.loads(l) for l in Path(a.records).read_text().splitlines()]
    if any(r['split']!='development' for r in recs):raise ValueError('development only')
    indexed,hashes=load_views(a.views);rows=[]
    with (out/'assignments.jsonl').open('w') as stream:
        for rec in recs:
            for policy in controls(rec):
                missing=[g for g in policy['groups'] if view_key(g) not in indexed.get(rec['id'],{})]
                if missing:raise ValueError('incomplete view library; do not drop cases')
                scores=policy_scores(rec,policy,indexed[rec['id']]);row=measure(rec,policy,scores)
                row.update(extended_metrics(rec,policy,scores,indexed[rec['id']]))
                rows.append(row);stream.write(json.dumps({'episode_id':rec['id'],**policy})+'\n')
    (out/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (out/'summary.json').write_text(json.dumps({'scope':'Development only; bundle probabilities exactly enumerated, other random policies averaged over 100 seeds. No seed is an independent task.',
                                               'source_hashes':hashes,'rows':summarize_extended(rows)},indent=2))
    # Per-case means are retained so aggregate plots cannot conceal case effects.
    percase=[]
    for cid in sorted({r['case_id'] for r in rows}):
        percase.extend(dict(r,case_id=cid) for r in summarize_extended([x for x in rows if x['case_id']==cid]))
    (out/'per_case.json').write_text(json.dumps(percase,indent=2))
    print(json.dumps({'episodes':len(recs),'cases':len({r['case_id'] for r in recs}),'evaluations':len(rows)}))


if __name__=='__main__':main()
