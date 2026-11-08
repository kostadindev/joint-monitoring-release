"""Exact matched-size randomization at fixed selected edges; offline development."""
import argparse
from itertools import combinations
import json
import math
from pathlib import Path
from record_strategies import select,group
from record_allocation import policy_scores,measure,summarize
from record_policy_extensions import extended_metrics
from record_pair_search import load_views


def partitions(edges,sizes):
    """Canonical unlabeled partitions with exactly this multiset of group sizes."""
    edges=tuple(sorted(edges));sizes=tuple(sorted(sizes))
    if sum(sizes)!=len(edges):raise ValueError('size mismatch')
    def recurse(remaining,shape):
        if not shape:
            yield ();return
        # Smallest remaining edge anchors one group. Distinct remaining sizes
        # avoid permutations of equal-size group slots.
        first=remaining[0]
        for size in sorted(set(shape)):
            restshape=list(shape);restshape.remove(size)
            for extra in combinations(remaining[1:],size-1):
                g=(first,)+extra;rest=tuple(e for e in remaining if e not in g)
                for tail in recurse(rest,tuple(restshape)):yield (g,)+tail
    return list(recurse(edges,sizes))


def main():
    p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--views',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    recs=[json.loads(l) for l in Path(a.records).read_text().splitlines()];indexed,hashes=load_views([a.views]);rows=[];total=0
    for rec in recs:
        if rec['split']!='development':raise ValueError('development only')
        k=math.ceil(len(rec['graph']['edges'])/2)
        for seed in (0,1,2):
            es=select(rec,k,'provenance_bundles',seed);original=group(rec,es,2,'provenance',seed)
            ps=partitions(es,list(map(len,original)));evaluated=[]
            for groups in ps:
                policy={'selector':'provenance_bundles','grouper':'exact_matched_random','k':k,'s':2,'seed':seed,'selected':es,'groups':groups}
                scores=policy_scores(rec,policy,indexed[rec['id']]);row=measure(rec,policy,scores)
                row.update(extended_metrics(rec,policy,scores,indexed[rec['id']]));evaluated.append(row)
            # Average within each seed first; otherwise seeds with more feasible
            # partitions would acquire more weight in the final comparison.
            result=dict(evaluated[0]);n=len(evaluated);total+=n
            for key,value in result.items():
                if key in ('seed','k','s','threshold'):continue
                if isinstance(value,(int,float)) and not isinstance(value,bool):result[key]=sum(r[key] for r in evaluated)/n
                elif isinstance(value,dict):result[key]={field:sum(r[key][field] for r in evaluated)/n for field in value}
            result['randomization_partitions']=n;result['expectation_not_a_single_policy_execution']=True;rows.append(result)
    (out/'per_case_seed_expectations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (out/'summary.json').write_text(json.dumps({'scope':'Exact random-group membership expectations at each fixed development selection, then equal weighting of three selector seeds. Not independent N.',
                                               'source_hashes':hashes,'enumerated_allocations':total,'rows':summarize(rows)},indent=2))
    print(json.dumps({'case_seed_expectations':len(rows),'enumerated_allocations':total}))


if __name__=='__main__':main()
