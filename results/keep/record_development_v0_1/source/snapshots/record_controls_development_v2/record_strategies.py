"""Candidate allocations for the fresh record-audit benchmark.

All selection uses public graph/roles only. Entity grouping uses only keys present
in already-selected messages; it does not inspect hidden labels or unselected text.
Provenance means declared sender-input/output dependency, not an attribution oracle.
"""
from __future__ import annotations
import math
import random
import networkx as nx
from network_strategies import select as graph_select, group as graph_group

SELECTORS=('random','random_nonregistry','mesa_static','betweenness','provenance_bundles','degree','pagerank','downstream_coverage','provenance_downstream')
GROUPERS=('singleton','random_chunks','provenance','random_matched_provenance','entity_keys','all_selected')

def graph_of(rec):
    g=nx.DiGraph();g.add_nodes_from(rec['graph']['nodes']);g.add_edges_from(rec['graph']['edges']);return g

def select(rec,k,method,seed):
    g=graph_of(rec);es=sorted(g.edges());k=min(k,len(es));rng=random.Random(seed)
    if method in ('random','mesa_static','betweenness','degree','pagerank','downstream_coverage'):return graph_select(g,k,method,seed)
    if method=='random_nonregistry':
        high=[e for e in es if e[0]!='registry'];low=[e for e in es if e[0]=='registry']
        rng.shuffle(high);rng.shuffle(low);return sorted((high+low)[:k])
    if method not in ('provenance_bundles','provenance_downstream'):raise ValueError(method)
    # A bundle pairs an output with one possible input to its sender. Protected
    # records provide stronger provenance, a public threat-model assumption.
    candidates=[]
    for output in es:
        for incoming in es:
            if incoming[1]==output[0]:
                weight=2.0 if incoming[0]=='registry' else 1.0
                if method=='provenance_downstream':weight*=1+len(nx.descendants(g,output[1]))/max(1,len(g))
                candidates.append((frozenset((incoming,output)),weight,rng.random()))
    chosen=set()
    while len(chosen)<k:
        feasible=[(bundle,weight,tie) for bundle,weight,tie in candidates if bundle-chosen and len(bundle|chosen)<=k]
        if not feasible:break
        bundle,_,_=max(feasible,key=lambda x:(x[1]/len(x[0]-chosen),x[2]))
        chosen.update(bundle)
    # Unused budget can inspect additional channels; no attack-dependent padding.
    rest=[e for e in es if e not in chosen];rng.shuffle(rest);chosen.update(rest[:k-len(chosen)])
    return sorted(chosen)

def _partition(es,s,affinity,seed):
    pool=list(es);rng=random.Random(seed);tie={e:rng.random() for e in pool};groups=[]
    while pool:
        if len(pool)==1:groups.append([pool.pop()]);break
        pairs=[(a,b) for i,a in enumerate(pool) for b in pool[:i]]
        a,b=max(pairs,key=lambda pair:(affinity(*pair),tie[pair[0]]+tie[pair[1]]))
        if affinity(a,b)<=0:groups.extend([[e] for e in pool]);break
        group=[a,b];pool.remove(a);pool.remove(b)
        while len(group)<s and pool:
            nxt=max(pool,key=lambda e:(sum(affinity(e,x) for x in group),tie[e]))
            if sum(affinity(nxt,x) for x in group)<=0:break
            group.append(nxt);pool.remove(nxt)
        groups.append(sorted(group))
    return sorted(groups)

def group(rec,es,s,method,seed):
    es=sorted(es)
    if s<1:raise ValueError('invalid group cap')
    if method=='all_selected':return [es] if len(es)<=s else None
    if s==1 or method=='singleton':return [[e] for e in es]
    if method in ('receiver','line_community','spectral'):return graph_group(graph_of(rec),es,s,method,seed)
    rng=random.Random(seed)
    if method=='random_chunks':
        shuffled=list(es);rng.shuffle(shuffled);return [sorted(shuffled[i:i+s]) for i in range(0,len(es),s)]
    def provenance(a,b):
        if a[1]==b[0]:return 2 if a[0]=='registry' else 1
        if b[1]==a[0]:return 2 if b[0]=='registry' else 1
        return 0
    if method in ('provenance','random_matched_provenance'):
        groups=_partition(es,s,provenance,seed)
        if method=='provenance':return groups
        shuffled=list(es);rng.shuffle(shuffled);out=[];offset=0
        for g in groups:out.append(sorted(shuffled[offset:offset+len(g)]));offset+=len(g)
        return out
    if method=='entity_keys':
        keys={e:set() for e in es}
        for m in rec['messages']:
            e=(m['sender'],m['recipient'])
            if e in keys:keys[e].update(m.get('payload',{}).get('facts',{}))
        def similarity(a,b):return len(keys[a]&keys[b])/max(1,len(keys[a]|keys[b]))
        return _partition(es,s,similarity,seed)
    raise ValueError(method)

def policies(rec,seeds=(0,1,2),fractions=(.25,.5,1.0)):
    n=len(rec['graph']['edges']);budgets=sorted({min(n,max(1,math.ceil(n*f))) for f in fractions});rows=[]
    for seed in seeds:
        for k in budgets:
            for selector in SELECTORS[:5]:
                es=select(rec,k,selector,seed)
                for s in sorted({1,min(2,k),min(4,k),k}):
                    for grouper in GROUPERS:
                        groups=group(rec,es,s,grouper,seed)
                        if groups is None:continue
                        if sorted(e for g in groups for e in g)!=es or any(len(g)>s for g in groups):raise AssertionError('invalid partition')
                        rows.append({'selector':selector,'grouper':grouper,'k':k,'s':s,'seed':seed,
                                     'selected':es,'groups':groups,'group_sizes':list(map(len,groups))})
    # Additional graph baselines and a downstream-weighted provenance heuristic
    # at the development comparison point. Keep the broader sweep tractable.
    k=math.ceil(n/2)
    for seed in seeds:
        for selector in SELECTORS:
            es=select(rec,k,selector,seed)
            methods=('provenance','random_matched_provenance') if selector not in SELECTORS[:5] else ()
            if selector in ('random','provenance_bundles'):methods+=('receiver','line_community','spectral')
            for grouper in methods:
                groups=group(rec,es,2,grouper,seed)
                rows.append({'selector':selector,'grouper':grouper,'k':k,'s':2,'seed':seed,
                             'selected':es,'groups':groups,'group_sizes':list(map(len,groups))})
    return rows

def policy_id(row):return f'{row["selector"]}/{row["grouper"]}/k{row["k"]}/s{row["s"]}/seed{row["seed"]}'

def rule_judge(rec,edges):
    """Cheap reference-join baseline for this explicitly structured environment.
    Detects false factual claims when a matching protected record is visible.
    It does NOT infer direct attack origin or malicious intent from disagreement.
    """
    selected=set(edges);observed=[m for m in rec['messages'] if (m['sender'],m['recipient']) in selected]
    references={}
    for m in observed:
        if m['trusted']:
            for field,value in m['payload'].get('facts',{}).items():references[(m['payload']['case_id'],field)]=value
    scores={e:0.0 for e in edges}
    for m in observed:
        if m['trusted']:continue
        for field,value in m['payload'].get('facts',{}).items():
            key=(m['payload']['case_id'],field)
            if key in references and value!=references[key]:scores[(m['sender'],m['recipient'])]=1.0
    return scores
