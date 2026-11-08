"""Graph-only edge selection and variable-size grouping; never reads attack labels.

These are baselines, not proposed novel algorithms. Spectral grouping is an
unsupervised graph embedding/clustering baseline, not a trained neural detector.
"""
from __future__ import annotations
import math
import networkx as nx
import numpy as np
from jmon.allocation.mesa_rank import mesa_scores

SELECTORS=('random','degree','pagerank','betweenness','mesa_static','downstream_coverage')
GROUPERS=('random_chunks','receiver','line_community','spectral','all_selected')

def downstream(g,node,hops=2):
    seen={node};front={node}
    for _ in range(hops):
        front={v for u in front for v in g.successors(u)}-seen;seen|=front
    return seen

def select(g,k,method,seed=0):
    if k<0: raise ValueError('negative budget')
    es=sorted(g.edges());k=min(k,len(es));rng=np.random.default_rng(seed)
    jitter={e:float(rng.random()) for e in es}
    if method=='random': score=jitter
    elif method=='degree': score={e:g.degree(e[0])+g.degree(e[1]) for e in es}
    elif method=='pagerank':
        pr=nx.pagerank(g);score={e:(pr[e[0]]+pr[e[1]])/2 for e in es}
    elif method=='betweenness': score=nx.edge_betweenness_centrality(g)
    elif method=='mesa_static': score=mesa_scores(g,combined=False)
    elif method=='downstream_coverage':
        # Bounded reachability avoids treating all members of a strongly connected
        # component as identical by definition. It can still tie on symmetric graphs.
        sets={e:downstream(g,e[1]) for e in es};covered=set();out=[]
        for _ in range(k):
            e=max((e for e in es if e not in out),key=lambda e:(len(sets[e]-covered),jitter[e]))
            out.append(e);covered|=sets[e]
        return sorted(out)
    else: raise ValueError(method)
    return sorted(sorted(es,key=lambda e:(score[e],jitter[e]),reverse=True)[:k])

def edge_affinity(g,es):
    # Endpoints and a bounded downstream neighborhood only; no observed content.
    neighborhoods={e:downstream(g,e[1]) for e in es}
    a=np.zeros((len(es),len(es)))
    for i,x in enumerate(es):
        for j,y in enumerate(es[:i]):
            overlap=len(neighborhoods[x]&neighborhoods[y])/max(1,len(neighborhoods[x]|neighborhoods[y]))
            weight=float(x[1]==y[1])+float(x[1]==y[0] or y[1]==x[0])+overlap
            a[i,j]=a[j,i]=weight
    return a

def _chunks(es,s):return [es[i:i+s] for i in range(0,len(es),s)]

def group(g,es,s,method,seed=0):
    if s<1: raise ValueError('maximum group size must be positive')
    es=sorted(es)
    if not es:return []
    rng=np.random.default_rng(seed)
    if method=='all_selected':
        return [es] if len(es)<=s else None # explicitly infeasible, never silent truncation
    if s==1:return [[e] for e in es]
    if method=='random_chunks':
        order=[es[i] for i in rng.permutation(len(es))];groups=_chunks(order,s)
    elif method=='receiver':
        groups=[]
        for receiver in sorted({e[1] for e in es}):
            incoming=[e for e in es if e[1]==receiver]
            order=[incoming[i] for i in rng.permutation(len(incoming))]
            groups.extend(_chunks(order,s))
    elif method=='line_community':
        line=nx.Graph();line.add_nodes_from(es)
        for i,x in enumerate(es):
            for y in es[:i]:
                if set(x)&set(y):line.add_edge(x,y)
        communities=list(nx.algorithms.community.greedy_modularity_communities(line)) if line.number_of_edges() else [{e} for e in es]
        groups=[c for community in sorted(communities,key=lambda x:sorted(x)) for c in _chunks(sorted(community),s)]
    elif method=='spectral':
        # Spectral embedding followed by deterministic-seeded Lloyd clustering.
        # Split oversized clusters to obey the cap; unequal groups are retained.
        a=edge_affinity(g,es);deg=a.sum(axis=1)
        inv=np.zeros_like(deg);inv[deg>0]=1/np.sqrt(deg[deg>0])
        lap=np.eye(len(es))-inv[:,None]*a*inv[None,:]
        _,vec=np.linalg.eigh(lap);q=max(1,math.ceil(len(es)/s));features=vec[:,:q]
        centers=features[rng.choice(len(es),size=q,replace=False)].copy()
        labels=np.zeros(len(es),dtype=int)
        for _ in range(30):
            nxt=((features[:,None,:]-centers[None,:,:])**2).sum(axis=2).argmin(axis=1)
            stable=np.array_equal(nxt,labels);labels=nxt
            for j in range(q):
                if np.any(labels==j):centers[j]=features[labels==j].mean(axis=0)
            if stable:break
        groups=[chunk for j in range(q) for chunk in _chunks([e for i,e in enumerate(es) if labels[i]==j],s)]
    else:raise ValueError(method)
    assert sorted(e for c in groups for e in c)==es
    assert all(1<=len(c)<=s for c in groups)
    return sorted([sorted(c) for c in groups])

def allocations(g,budgets=(2,4,8),caps=(1,2,4,8),seed=0):
    output=[]
    for k in budgets:
        if k>g.number_of_edges():continue
        for selector in SELECTORS:
            selected=select(g,k,selector,seed)
            for s in caps:
                if s>k:continue
                for grouper in GROUPERS:
                    groups=group(g,selected,s,grouper,seed)
                    if groups is None:continue
                    output.append({'selector':selector,'grouper':grouper,'k':k,'s':s,'seed':seed,
                                   'selected':selected,'groups':groups})
    return output
