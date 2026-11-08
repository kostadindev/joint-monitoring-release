"""New development allocation candidates using anonymous directed connectivity only.

An output-parent neighborhood is (u,v) plus ALL edges entering u. A depth-d
upstream cone recursively includes incoming edges for d predecessor layers.
These are topology-derived hypotheses, NOT certificates that the named channels
contain actual causal inputs or sufficient semantic evidence. Cycles terminate by
bounded expansion; temporal causality needs the separate execution adapter.

Greedy disjoint motif packing is a heuristic, not an optimizer. Motifs too large
for k/s are excluded whole, never trimmed and called parent-complete. Source
edges with no predecessors are not relational motifs. Remaining acquisition slots
are uniformly filled; uncovered grouped edges are singletons. Structural scoring
uses number of complete parent neighborhoods inside a motif divided by its size.
No observation, role, trust, identity magnitude, label, or length enters a score.

Independent-purpose deterministic PRNG streams define reproducible randomness.
ID sorting only canonicalizes storage and enumeration. Individual fixed-seed
choices need not commute with relabeling on symmetric graphs; the random law does.
Rotation is a fixed-seed coverage schedule, not independent Monte Carlo sampling.
"""
from __future__ import annotations
from collections import Counter
from fractions import Fraction
from itertools import combinations
import hashlib
import json
import math
import random
from topology_only_policies import validate_graph,select as legacy_select,group as legacy_group

VERSION='anonymous-neighborhood-v1'
SELECTORS=('parent_complete','bounded_cone','random','degree','structural_path_pairs','rotation','path_coverage')
GROUPERS=('parent_complete','bounded_cone','directed_path','singleton','full_selected','random_chunks','random_matched_sizes')


def integer(x,name,minimum=0):
    if type(x) is not int or x<minimum:raise ValueError(f'{name} must be an integer >= {minimum}')
    return x

def seed_for(seed,domain):
    if type(seed) is not int:raise ValueError('seed must be an integer')
    return int.from_bytes(hashlib.sha256(json.dumps([VERSION,seed,domain],separators=(',',':')).encode()).digest(),'big')

def edges_checked(g,selected):
    if not isinstance(selected,(list,tuple)):raise ValueError('selected must be an edge sequence')
    result=[]
    for edge in selected:
        if not isinstance(edge,(list,tuple)) or len(edge)!=2 or any(type(n) is not int for n in edge) or not g.has_edge(*edge):raise ValueError('invalid selected edge')
        result.append(tuple(edge))
    if len(result)!=len(set(result)):raise ValueError('duplicate selected edge')
    return sorted(result)

def serial_edges(edges):return [list(e) for e in sorted(edges)]
def serial_groups(groups):return [serial_edges(g) for g in sorted(tuple(sorted(p)) for p in groups)]

def _motifs(g,depth):
    """One exact bounded predecessor closure per output, deduplicated by edge set."""
    integer(depth,'depth',1);bundles={}
    for output in sorted(g.edges()):
        if not g.in_degree(output[0]):continue
        members={output};frontier={output[0]};expanded=set()
        for _ in range(depth):
            following=set()
            for node in frontier-expanded:
                incoming=set(g.in_edges(node));members.update(incoming)
                following.update(u for u,_ in incoming)
            expanded.update(frontier);frontier=following-expanded
            if not frontier:break
        bundle=tuple(sorted(members));bundles.setdefault(bundle,[]).append(output)
    return [{'edges':serial_edges(bundle),'anchor_outputs':serial_edges(anchors),'depth':depth}
            for bundle,anchors in sorted(bundles.items())]

def parent_neighborhoods(graph):return _motifs(validate_graph(graph),1)
def bounded_cones(graph,depth=2):return _motifs(validate_graph(graph),depth)

def _parent_sets(g):
    return {edge:frozenset([edge,*g.in_edges(edge[0])]) for edge in g.edges() if g.in_degree(edge[0])}

def _pack(g,available,limit,cap,depth,rng):
    parents=_parent_sets(g);remaining=set(available);groups=[];used=0
    motifs=[frozenset(map(tuple,m['edges'])) for m in _motifs(g,depth)]
    while True:
        possible=[m for m in motifs if len(m)<=min(cap,limit-used) and m<=remaining]
        if not possible:break
        def score(m):
            completed=sum(p<=m for p in parents.values())
            return (Fraction(completed,len(m)),completed)
        best=max(map(score,possible));ties=sorted(tuple(sorted(m)) for m in possible if score(m)==best)
        chosen=ties[rng.randrange(len(ties))];groups.append(chosen);used+=len(chosen);remaining.difference_update(chosen)
    return groups,remaining

def bounded_path_universe(graph,max_path_edges=3,max_paths=20000):
    """ALL simple directed paths of lengths1..L; cap overflow is an error.

    Uniform graph scenarios, not exploit paths, privileged critical assets or
    observed attacks. The path family is fixed before greedy sensor placement.
    """
    g=validate_graph(graph);integer(max_path_edges,'max_path_edges',1);integer(max_paths,'max_paths',1)
    paths=[]
    for start in sorted(g.nodes()):
        stack=[((start,),())]
        while stack:
            nodes,edges=stack.pop()
            if len(edges)>=max_path_edges:continue
            for target in sorted(g.successors(nodes[-1]),reverse=True):
                if target in nodes:continue
                extended=edges+((nodes[-1],target),);paths.append(extended)
                if len(paths)>max_paths:raise ValueError('path universe exceeds explicit max_paths cap; no truncated scenario objective returned')
                stack.append((nodes+(target,),extended))
    paths=sorted(paths)
    return {'kind':'all_bounded_simple_directed_paths','max_path_edges':max_path_edges,'max_paths':max_paths,
            'count':len(paths),'paths':[[list(e) for e in path] for path in paths],
            'scenario_sha256':hashlib.sha256(json.dumps(paths,separators=(',',':')).encode()).hexdigest(),
            'weights':'one per path; hit-any-edge OR coverage; independent of messages, attacks and outcomes'}

def _coverage_pick(g,k,universe,rng):
    memberships={e:set() for e in g.edges()}
    for index,path in enumerate(universe['paths']):
        for edge in path:memberships[tuple(edge)].add(index)
    chosen=[];covered=set();remaining=set(g.edges())
    for _ in range(k):
        gains={e:len(memberships[e]-covered) for e in remaining};gain=max(gains.values())
        ties=sorted(e for e in remaining if gains[e]==gain);edge=ties[rng.randrange(len(ties))]
        chosen.append(edge);covered.update(memberships[edge]);remaining.remove(edge)
    return chosen,len(covered)

def path_coverage_plan(graph,k,seed=0,max_path_edges=3,max_paths=20000):
    """Greedy cardinality-budget maximum OR coverage of a fixed graph path family.

    Inspired by IDS/sensor-placement set coverage. This is not Noel/Jajodia's
    vulnerability/critical-asset attack-graph model or a guarantee for joint
    detection. Any set-coverage guarantees concern ONLY this surrogate objective.
    """
    g=validate_graph(graph);integer(k,'k');seed_for(seed,'validate')
    if k>g.number_of_edges():raise ValueError('k exceeds available edges')
    universe=bounded_path_universe(graph,max_path_edges,max_paths)
    selected,covered=_coverage_pick(g,k,universe,random.Random(seed_for(seed,'selection:path_coverage')))
    return {'selected':serial_edges(selected),'scenario_count':universe['count'],'covered_scenarios':covered,
        'coverage_fraction':covered/universe['count'] if universe['count'] else None,
        'scenario_sha256':universe['scenario_sha256'],'max_path_edges':max_path_edges,'max_paths':max_paths,
        'objective':'uniform bounded-simple-path hit-any-edge OR coverage; structural surrogate, not joint judge reward'}

def select(graph,k,s,method='parent_complete',seed=0,depth=2,rotation_step=0,*,path_hops=3,path_cap=20000):
    """Acquire exactly k distinct channels. s is a legitimate allocation constraint."""
    g=validate_graph(graph);integer(k,'k');integer(s,'s',1);integer(depth,'depth',1);integer(rotation_step,'rotation_step');seed_for(seed,'validate')
    if k>g.number_of_edges():raise ValueError('k exceeds available edges')
    if method not in SELECTORS:raise ValueError('unknown selector')
    universe=sorted(g.edges());rng=random.Random(seed_for(seed,'selection:'+method))
    if method=='path_coverage':return path_coverage_plan(graph,k,seed,path_hops,path_cap)['selected']
    if method=='random':selected=rng.sample(universe,k)
    elif method=='rotation':
        rng.shuffle(universe)
        selected=[universe[(rotation_step*k+i)%len(universe)] for i in range(k)] if k else []
    elif method in ('degree','structural_path_pairs'):
        selected=legacy_select(graph,k,method,seed_for(seed,'legacy-selection:'+method))
    else:
        groups,remaining=_pack(g,universe,k,s,1 if method=='parent_complete' else depth,rng)
        selected=[e for p in groups for e in p]
        selected.extend(rng.sample(sorted(remaining),k-len(selected)))
    return serial_edges(selected)

def group(graph,selected,s,method='parent_complete',seed=0,depth=2):
    """Partition only already acquired edges; no graph edge can be acquired here."""
    g=validate_graph(graph);selected=edges_checked(g,selected);integer(s,'s',1);integer(depth,'depth',1);seed_for(seed,'validate')
    if method not in GROUPERS:raise ValueError('unknown grouper')
    rng=random.Random(seed_for(seed,'group:'+method))
    if method in ('directed_path','singleton','full_selected','random_matched_sizes'):
        result=legacy_group(graph,selected,s,method,seed_for(seed,'legacy-group'))
        if result is None:return None
    elif method=='random_chunks':
        order=list(selected);rng.shuffle(order);result=[order[i:i+s] for i in range(0,len(order),s)]
    else:
        result,remaining=_pack(g,selected,len(selected),s,1 if method=='parent_complete' else depth,rng)
        result.extend((e,) for e in sorted(remaining))
    result=serial_groups(result)
    flat=[tuple(e) for p in result for e in p]
    if sorted(flat)!=selected or len(flat)!=len(set(flat)) or any(not 1<=len(p)<=s for p in result):raise AssertionError('invalid disjoint partition')
    return result

def packing_diagnostics(graph,selected,groups,s,depth=2):
    """Graph-only structural opportunity counters; not semantic detection claims."""
    g=validate_graph(graph);selected=set(edges_checked(g,selected));integer(s,'s',1)
    parts=[] if groups is None else [set(edges_checked(g,p)) for p in groups]
    parents=_parent_sets(g);cones=_motifs(g,depth)
    covered=[e for p in parents.values() if any(p<=q for q in parts) for e in p]
    return {'parent_anchor_outputs':len(parents),'parent_candidates_fitting_group_cap':sum(len(p)<=s for p in parents.values()),
        'parent_anchors_fully_acquired':sum(p<=selected for p in parents.values()),
        'parent_anchors_coobserved':sum(any(p<=q for q in parts) for p in parents.values()),
        'distinct_cone_candidates':len(cones),'cone_candidates_fitting_group_cap':sum(len(m['edges'])<=s for m in cones),
        'cone_candidates_fully_acquired':sum(set(map(tuple,m['edges']))<=selected for m in cones),
        'cone_candidates_coobserved':sum(any(set(map(tuple,m['edges']))<=q for q in parts) for m in cones),
        'singleton_groups':sum(len(p)==1 for p in parts),'selected_edges_outside_coobserved_parent_neighborhood':len(selected-set(covered))}

def allocate(graph,k,s,selector='parent_complete',grouper=None,seed=0,depth=2,rotation_step=0,*,path_hops=3,path_cap=20000):
    if grouper is None:grouper=selector if selector in ('parent_complete','bounded_cone') else 'directed_path'
    selected=select(graph,k,s,selector,seed,depth,rotation_step,path_hops=path_hops,path_cap=path_cap);groups=group(graph,selected,s,grouper,seed,depth)
    result={'version':VERSION,'selector':selector,'grouper':grouper,'k':k,'s':s,'seed':seed,'depth':depth,
        'rotation_step':rotation_step,'selected':selected,'groups':groups,
        'group_sizes':sorted(map(len,groups)) if groups is not None else None,'feasible_group_cap':groups is not None,
        'unused_edge_budget':k-len(selected),'information_access':'anonymous directed topology only',
        'structural_diagnostics':packing_diagnostics(graph,selected,groups,s,depth)}
    if selector=='path_coverage':result['path_coverage']=path_coverage_plan(graph,k,seed,path_hops,path_cap)
    return result

def partition_count(sizes):
    if not isinstance(sizes,(list,tuple)):raise ValueError('sizes must be a sequence')
    for size in sizes:integer(size,'group size',1)
    n=sum(sizes);denominator=math.prod(math.factorial(s) for s in sizes)*math.prod(math.factorial(c) for c in Counter(sizes).values())
    return math.factorial(n)//denominator

def _partitions(edges,sizes):
    """Unique unlabeled partitions: force smallest remaining edge into next group."""
    if not edges:
        yield ()
        return
    first,*rest=edges
    for size in sorted(set(sizes)):
        next_sizes=list(sizes);next_sizes.remove(size)
        for companions in combinations(rest,size-1):
            part=tuple(sorted((first,*companions)));remaining=sorted(set(edges)-set(part))
            for tail in _partitions(remaining,next_sizes):yield tuple(sorted((part,*tail)))

def _mode(universe,draws,exact_limit):
    integer(exact_limit,'exact_limit')
    if draws is not None:integer(draws,'draws',1)
    if universe<=exact_limit:return 'exact',universe
    if draws is None:raise ValueError('explicit Monte Carlo draws required when exact universe exceeds limit')
    if draws>100000:raise ValueError('draws exceeds offline materialization safety limit100000')
    return 'monte_carlo',draws

def uniform_selection_reference(graph,k,draws=None,seed=0,exact_limit=20000):
    """Uniform k-subsets, exact or fixed-count IID MC; duplicate draws retained."""
    g=validate_graph(graph);integer(k,'k');seed_for(seed,'validate')
    if k>g.number_of_edges():raise ValueError('k exceeds available edges')
    edges=sorted(g.edges());universe=math.comb(len(edges),k);mode,n=_mode(universe,draws,exact_limit)
    rng=random.Random(seed_for(seed,'reference:uniform-selection'))
    choices=combinations(edges,k) if mode=='exact' else (rng.sample(edges,k) for _ in range(n))
    return {'version':VERSION,'kind':'uniform_selection','mode':mode,'universe_size':universe,'draws':n,'requested_mc_draws':draws,'exact_limit':exact_limit,'seed':seed,
        'scope':'Exact finite expectation or fixed-count IID Monte Carlo over k-subsets; no model outcomes enter sampling.',
        'selections':[{'draw_id':i,'selected':serial_edges(es),'weight_numerator':1,'weight_denominator':n} for i,es in enumerate(choices)]}

def matched_grouping_reference(graph,selected,group_sizes,s,draws=None,seed=0,exact_limit=20000):
    """Full partitions support additive AND nonlinear metric averaging.

    MC shuffles all selected edges uniformly and chunks by a fixed size multiset.
    Every unlabeled partition has the same number of shuffle preimages. Keep
    duplicate draws; do not replace sampled repetitions with unique partitions.
    """
    g=validate_graph(graph);edges=edges_checked(g,selected);integer(s,'s',1);seed_for(seed,'validate')
    universe=partition_count(group_sizes)
    if sum(group_sizes)!=len(edges) or any(n>s for n in group_sizes):raise ValueError('group sizes do not match selected edges/cap')
    mode,n=_mode(universe,draws,exact_limit);rng=random.Random(seed_for(seed,'reference:matched-grouping'))
    def samples():
        for _ in range(n):
            order=list(edges);rng.shuffle(order);parts=[];offset=0
            for size in sorted(group_sizes):parts.append(tuple(sorted(order[offset:offset+size])));offset+=size
            yield tuple(sorted(parts))
    partitions=_partitions(edges,sorted(group_sizes)) if mode=='exact' else samples()
    result=[{'draw_id':i,'groups':serial_groups(parts),'weight_numerator':1,'weight_denominator':n} for i,parts in enumerate(partitions)]
    if len(result)!=n:raise AssertionError('partition census count mismatch')
    return {'version':VERSION,'kind':'matched_grouping','mode':mode,'universe_size':universe,'draws':n,'requested_mc_draws':draws,'exact_limit':exact_limit,'seed':seed,
        'selected':serial_edges(edges),'group_sizes':sorted(group_sizes),'partitions':result,
        'scope':'Same selection and exact size multiset. Average metrics per partition; MC uncertainty is separate from task uncertainty.'}

def candidate_group_weights(reference):
    """Rational expected group occurrences; only additive metrics use these alone."""
    if reference.get('kind')!='matched_grouping':raise ValueError('matched-grouping reference required')
    counts=Counter(tuple(map(tuple,p)) for draw in reference['partitions'] for p in draw['groups'])
    denominator=reference['draws']
    return [{'edges':serial_edges(es),'weight_numerator':n,'weight_denominator':denominator} for es,n in sorted(counts.items())]
