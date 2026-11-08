"""Prospective acquisition/grouping and rule-baseline mechanism diagnostics.

No labels or unselected contents enter policy choices. The rule matcher is an
upper diagnostic for this controlled grammar, not a simulation of an LLM judge.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
from record_audit_analysis import parse_claim, rule_flags
from record_strategies import select, group, _partition, policy_id


def group_selected(rec,selected,s,method,seed):
    if method!='observed_keys':return group({'graph':rec['graph']},selected,s,method,seed)
    if s==1:return [[e] for e in selected]
    # All parsing occurs after paid acquisition; fields on unselected channels
    # cannot influence selection or grouping.
    keys={tuple(e):set() for e in selected}
    for m in rec['messages']:
        e=(m['sender'],m['recipient'])
        if e not in keys:continue
        parsed=parse_claim(m['content'])
        if parsed:
            cid,field,revision,_=parsed;keys[e].add((cid,field,revision))
    return _partition(selected,s,lambda a,b:len(keys[a]&keys[b]),seed)


def policies(rec,seeds=(0,1,2),budgets=(4,7,13)):
    graph_only={'graph':rec['graph']}
    for k in budgets:
        for seed in seeds:
            for selector in ('random','provenance_bundles','mesa_static','degree','pagerank','betweenness','downstream_coverage'):
                selected=select(graph_only,k,selector,seed)
                for s in sorted({1,2,4,k}):
                    if s>k:continue
                    methods=('singleton',) if s==1 else ('provenance','random_matched_provenance','observed_keys','spectral')
                    if s==k:methods+=('all_selected',)
                    for grouper in methods:
                        groups=group_selected(rec,selected,s,grouper,seed)
                        if sorted(e for g in groups for e in g)!=selected:raise ValueError('bad partition')
                        if any(not 1<=len(g)<=s for g in groups):raise ValueError('bad group cap')
                        yield {'selector':selector,'grouper':grouper,'k':k,'s':s,'seed':seed,
                               'selected':selected,'groups':groups}


def byte_profile(rec):
    from monitor_record_workflows import view_prompt,FORMAT_RETRY
    from run_record_audit_diagnostics import CLARIFIED
    edges=sorted(map(tuple,rec['graph']['edges']))
    count=lambda es:len((CLARIFIED+FORMAT_RETRY+view_prompt(rec,es)).encode())
    base=count([]);delta={e:count([e])-base for e in edges}
    if base+sum(delta.values())+2*(len(edges)-1)!=count(edges):raise ValueError('byte decomposition failed')
    return base,delta


def measure(rec,p,profile=None):
    selected=set(p['selected']);truth=set(map(tuple,rec['labels']['false_origin_edges']))
    grouped=set().union(*(rule_flags(rec,g) for g in p['groups']))
    joined=rule_flags(rec,selected)
    base,delta=profile if profile is not None else byte_profile(rec)
    group_bytes=[base+sum(delta[tuple(e)] for e in g)+2*(len(g)-1) for g in p['groups']]
    return {'id':rec['id'],'case_id':rec['case_id'],**{k:rec[k] for k in ('condition','environment','history','rendering')},
            'workload_size':rec.get('workload_size',1),'policy':policy_id(p),**{k:p[k] for k in ('selector','grouper','k','s','seed')},
            'n_truth':len(truth),'covered':len(selected&truth),'grouped_tp':len(grouped&truth),'grouped_fp':len(grouped-truth),
            'joined_tp':len(joined&truth),'joined_fp':len(joined-truth),'calls_if_llm':len(p['groups']),
            'max_group_input_bytes':max(group_bytes),'total_group_input_bytes':sum(group_bytes),
            'llm_byte_feasible':all(n<=24000 for n in group_bytes),'input_byte_allowance':24000}


def main():
    p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    rs=[json.loads(l) for l in Path(a.records).read_text().splitlines()];out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rows=[];buckets=defaultdict(list)
    with (out/'assignments.jsonl').open('w') as stream:
        for rec in rs:
            profile=byte_profile(rec)
            for policy in policies(rec):
                row=measure(rec,policy,profile);rows.append(row)
                stream.write(json.dumps({'episode_id':rec['id'],**policy})+'\n')
    (out/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    keys=('condition','environment','history','rendering','workload_size','selector','grouper','k','s')
    for r in rows:buckets[tuple(r[k] for k in keys)].append(r)
    summary=[]
    for key,items in sorted(buckets.items()):
        n=sum(r['n_truth'] for r in items)
        summary.append(dict(zip(keys,key))|{'cases':len({r['case_id'] for r in items}),'evaluations':len(items),
                      **{m:sum(r[m] for r in items)/n if n else None for m in ('covered','grouped_tp','joined_tp')},
                      'grouped_false_flags':sum(r['grouped_fp'] for r in items),
                      'joined_false_flags':sum(r['joined_fp'] for r in items),
                      'llm_feasible_evaluations':sum(r['llm_byte_feasible'] for r in items),
                      'max_group_input_bytes':max(r['max_group_input_bytes'] for r in items),
                      'mean_total_input_bytes':sum(r['total_group_input_bytes'] for r in items)/len(items)})
    (out/'summary.json').write_text(json.dumps({'scope':'Prospective allocation with grammar-aware deterministic matching; no new judge calls or independent evaluation.', 'rows':summary},indent=2))
    print(json.dumps({'records':len(rs),'evaluations':len(rows)}))

if __name__=='__main__':main()
