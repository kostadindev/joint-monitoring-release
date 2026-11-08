"""Pure offline analytics for anonymous, loopless simple directed graphs.

Input is exactly {'nodes': [integer IDs], 'edges': [[u,v], ...]}; no semantic
attributes, roles, content, labels or weights are accepted. IDs are bookkeeping.
All distances count edges. Directed reachability excludes self-pairs. A global
directed diameter exists here only for a nonempty strongly connected graph;
max finite distance and component diameters are different explicitly named values.
Undirected projection statistics collapse reciprocal edges. Cycle enumeration and
automorphisms have explicit caps: incomplete enumeration never implies exactness.
WL hashes only screen candidates; every isomorphism-class merge uses exact VF2.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
from collections.abc import Mapping
import warnings
import networkx as nx

SCHEMA_VERSION='anonymous-directed-graph-analytics-v1'


def validate_graph(graph):
    if type(graph) is not dict or set(graph)!={'nodes','edges'}:raise ValueError('graph must contain exactly anonymous nodes and edges')
    if not isinstance(graph['nodes'],(list,tuple)) or not isinstance(graph['edges'],(list,tuple)):raise ValueError('nodes and edges must be sequences')
    nodes=graph['nodes']
    if any(type(n) is not int for n in nodes) or len(set(nodes))!=len(nodes):raise ValueError('unique integer node IDs required')
    edges=[];universe=set(nodes)
    for edge in graph['edges']:
        if not isinstance(edge,(list,tuple)) or len(edge)!=2 or any(type(x) is not int for x in edge):raise ValueError('two integer endpoints required')
        u,v=edge
        if u not in universe or v not in universe or u==v:raise ValueError('unknown endpoint or unsupported self-loop')
        edges.append((u,v))
    if len(set(edges))!=len(edges):raise ValueError('parallel/duplicate edges unsupported')
    g=nx.DiGraph();g.add_nodes_from(sorted(nodes));g.add_edges_from(sorted(edges));return g


def _limit(value,name,minimum=1):
    if type(value) is not int or value<minimum:raise ValueError(name+' must be an integer >= '+str(minimum))


def _distribution(values,name):
    return [{name:value,'count':count} for value,count in sorted(Counter(values).items())]


def _pagerank(g,alpha=.85,tolerance=1e-12,max_iterations=1000):
    """Unweighted PageRank power iteration, uniform teleport/dangling mass.

    Uses only NetworkX adjacency; avoids making optional SciPy a requirement.
    Same L1 < n*tolerance convergence rule as NetworkX's power iteration.
    """
    n=len(g)
    if not n:return {},0,True
    values={u:1/n for u in g}
    for iteration in range(1,max_iterations+1):
        dangling=math.fsum(values[u] for u in g if not g.out_degree(u))
        updated={u:(1-alpha)/n+alpha*dangling/n for u in g}
        for u in g:
            if g.out_degree(u):
                mass=alpha*values[u]/g.out_degree(u)
                for v in g[u]:updated[v]+=mass
        error=math.fsum(abs(updated[u]-values[u]) for u in g);values=updated
        if error<n*tolerance:return values,iteration,True
    return values,max_iterations,False


def _automorphisms(g,node_limit,mapping_limit):
    common={'node_limit':node_limit,'mapping_limit':mapping_limit,
        'scope':'Exact directed adjacency-preserving permutations; no attributes. Partial equivalence classes are proven subsets of full orbits, never claimed complete.'}
    if len(g)>node_limit:return {**common,'status':'unknown_node_limit','complete':False,'automorphism_count':None,'automorphism_count_lower_bound':1,'enumerated_mappings':0,'node_orbits':None,'edge_orbits':None}
    node_sets=nx.utils.UnionFind(g.nodes);edge_sets=nx.utils.UnionFind(g.edges);count=0;complete=True
    for mapping in nx.algorithms.isomorphism.DiGraphMatcher(g,g).isomorphisms_iter():
        count+=1
        for u in g:node_sets.union(u,mapping[u])
        for u,v in g.edges:edge_sets.union((u,v),(mapping[u],mapping[v]))
        if count>mapping_limit:complete=False;break
    nodes=sorted((sorted(group) for group in node_sets.to_sets()),key=lambda x:(len(x),x))
    edges=sorted(([list(e) for e in sorted(group)] for group in edge_sets.to_sets()),key=lambda x:(len(x),x))
    return {**common,'status':'exact' if complete else 'unknown_mapping_limit','complete':complete,
        'automorphism_count':count if complete else None,'automorphism_count_lower_bound':count,'enumerated_mappings':count,
        'node_orbits':nodes if complete else None,'edge_orbits':edges if complete else None,
        'proven_node_equivalence_subsets':nodes,'proven_edge_equivalence_subsets':edges}


def _cycle_metrics(g,projection,limit,distances,is_dag):
    triangles=sum(nx.triangles(projection).values())//3
    directed_triangles=sum(len(set(g.successors(v))&set(g.predecessors(u))) for u,v in g.edges)//3
    lengths=[];complete=True
    if not is_dag:
        for cycle in nx.simple_cycles(g):
            lengths.append(len(cycle))
            if len(lengths)>limit:complete=False;break
    girth=min((distances[v][u]+1 for u,v in g.edges if u in distances[v]),default=None)
    return {'undirected_projection_triangles':triangles,'directed_three_cycles':directed_triangles,'directed_girth':girth,
        'simple_cycle_enumeration':{'limit':limit,'complete':complete,'count':len(lengths) if complete else None,
            'count_lower_bound':len(lengths),'enumerated_length_distribution':_distribution(lengths,'length'),
            'scope':'Distinct directed simple cycles, modulo rotation but not reversal; reciprocal dyads count as length two. At most limit+1 cycles examined to detect truncation.'}}


def _reconvergence(g,is_dag,reachable):
    if not is_dag:return {'scope':'At least two distinct directed paths from a source to a later vertex; exact capped-at-two DAG dynamic programming. Includes shortcut/transitive-triangle path alternatives, not only internally disjoint branches.',
        'status':'undefined_for_cyclic_graph','ordered_pairs_with_multiple_paths':None,'fraction_of_reachable_pairs':None},None
    order=list(nx.topological_sort(g));counts={u:0 for u in g};total=0
    for source in order:
        paths={u:0 for u in g};paths[source]=1
        for u in order:
            for v in g[u]:paths[v]=min(2,paths[v]+paths[u])
        for target in g:
            if paths[target]>=2:counts[target]+=1;total+=1
    return {'scope':'At least two distinct directed paths from a source to a later vertex; exact capped-at-two DAG dynamic programming. Includes shortcut/transitive-triangle path alternatives, not only internally disjoint branches.',
        'status':'exact_dag','ordered_pairs_with_multiple_paths':total,'fraction_of_reachable_pairs':total/reachable if reachable else None},counts


def _wl_bucket(g):
    # NetworkX >=3.5 changed directed WL semantics. Record version; never use
    # hashes across versions as evidence of equality, and always verify exactly.
    if not len(g):return 'empty'
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='The hashes produced for directed graphs changed.*')
        warnings.filterwarnings('ignore',message='The hashes produced for graphs without node or edge attributes changed.*')
        return nx.weisfeiler_lehman_graph_hash(g,iterations=3,digest_size=16)


def analyze_graph(graph,*,automorphism_node_limit=12,automorphism_mapping_limit=10000,cycle_limit=10000,include_communities=False):
    """Return JSON-ready graph, node and edge statistics. No side effects/calls.

    Automorphisms exceeding either cap are unknown, not asymmetric. Directed
    cycles are bounded-enumerated separately from exact girth/triangle counts.
    Optional communities use greedy modularity on the undirected projection;
    this heuristic is not an exact optimum and ties can depend on opaque IDs.
    """
    _limit(automorphism_node_limit,'automorphism_node_limit',0);_limit(automorphism_mapping_limit,'automorphism_mapping_limit');_limit(cycle_limit,'cycle_limit')
    if type(include_communities) is not bool:raise ValueError('include_communities must be boolean')
    g=validate_graph(graph);undirected=g.to_undirected();n=len(g);m=g.number_of_edges();is_dag=nx.is_directed_acyclic_graph(g)
    distances={u:dict(ds) for u,ds in nx.all_pairs_shortest_path_length(g)}
    lengths=[length for u,ds in distances.items() for v,length in ds.items() if u!=v];reachable=len(lengths)
    weak=sorted((sorted(c) for c in nx.weakly_connected_components(g)),key=lambda x:(-len(x),x))
    strong=sorted((sorted(c) for c in nx.strongly_connected_components(g)),key=lambda x:(-len(x),x))
    weak_rows=[{'nodes':c,'size':len(c),'undirected_projection_diameter':nx.diameter(undirected.subgraph(c))} for c in weak]
    strong_rows=[{'nodes':c,'size':len(c),'directed_diameter':nx.diameter(g.subgraph(c))} for c in strong]
    indegrees=dict(g.in_degree());outdegrees=dict(g.out_degree());totaldegrees=dict(g.degree())
    bridges={tuple(sorted(e)) for e in nx.bridges(undirected)};articulation=set(nx.articulation_points(undirected))
    between=nx.betweenness_centrality(g,normalized=True,endpoints=False);edge_between=nx.edge_betweenness_centrality(g,normalized=True)
    pr,iterations,converged=_pagerank(g)
    reconvergence,reconvergent_ancestors=_reconvergence(g,is_dag,reachable)
    reciprocal_dyads=sum(g.has_edge(v,u) for u,v in g.edges)//2
    walks=sum(indegrees[u]*outdegrees[u] for u in g);simple_paths=walks-2*reciprocal_dyads
    receiver_pairs=sum(math.comb(d,2) for d in indegrees.values());sender_pairs=sum(math.comb(d,2) for d in outdegrees.values())
    total_pairs=math.comb(m,2);endpoint_pairs=simple_paths+receiver_pairs+sender_pairs+reciprocal_dyads
    possible=n*(n-1)
    def centralization(values,denominator):return sum(max(values,default=0)-v for v in values)/denominator if denominator else None
    node_metrics=[]
    for u in g:
        node_metrics.append({'node':u,'in_degree':indegrees[u],'out_degree':outdegrees[u],'total_degree':totaldegrees[u],
            'in_degree_centrality':indegrees[u]/(n-1) if n>1 else 0.,'out_degree_centrality':outdegrees[u]/(n-1) if n>1 else 0.,
            'total_degree_centrality':totaldegrees[u]/(2*(n-1)) if n>1 else 0.,
            'pagerank':pr[u] if converged else None,'betweenness':between[u],
            'reachable_other_nodes':len(distances[u])-1,'other_nodes_reaching_this':sum(u in distances[v] for v in g if v!=u),
            'source':indegrees[u]==0,'sink':outdegrees[u]==0,'isolated':totaldegrees[u]==0,
            'undirected_projection_articulation':u in articulation,
            'ancestors_with_multiple_paths':reconvergent_ancestors[u] if reconvergent_ancestors is not None else None})
    edge_metrics=[{'source':u,'target':v,'betweenness':edge_between[(u,v)],'reciprocal':g.has_edge(v,u),
        'simple_two_path_predecessors':indegrees[u]-int(g.has_edge(v,u)),
        'simple_two_path_successors':outdegrees[v]-int(g.has_edge(v,u)),
        'other_edges_same_sender':outdegrees[u]-1,'other_edges_same_receiver':indegrees[v]-1,
        'undirected_projection_bridge':tuple(sorted((u,v))) in bridges} for u,v in sorted(g.edges)]
    communities={'enabled':include_communities,'scope':'Greedy modularity on unweighted undirected projection, reciprocal edges collapsed. Heuristic, not optimality or semantic communities; tie choices may depend on opaque IDs.'}
    if include_communities:
        if undirected.number_of_edges():
            groups=list(nx.community.greedy_modularity_communities(undirected,resolution=1))
            communities.update(modularity=nx.community.modularity(undirected,groups,resolution=1),communities=sorted([sorted(c) for c in groups],key=lambda x:(-len(x),x)),resolution=1.)
        else:communities.update(modularity=None,communities=[[u] for u in g],resolution=1.)
    return {'schema_version':SCHEMA_VERSION,'networkx_version':nx.__version__,
        'graph_sha256':hashlib.sha256(json.dumps({'nodes':list(g),'edges':[list(e) for e in sorted(g.edges)]},sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'graph_sha256_scope':'Exact anonymous labeled adjacency, not an isomorphism-invariant identifier.',
        'n_nodes':n,'n_edges':m,'density':m/possible if possible else 0.,'is_dag':is_dag,
        'dag_longest_path_edges':nx.dag_longest_path_length(g) if is_dag else None,
        'sources':[u for u in g if indegrees[u]==0],'sinks':[u for u in g if outdegrees[u]==0],'isolates':list(nx.isolates(g)),
        'source_sink_scope':'Zero indegree/outdegree respectively; an isolate is both.',
        'degree_distribution':{'in':_distribution(indegrees.values(),'degree'),'out':_distribution(outdegrees.values(),'degree'),'total':_distribution(totaldegrees.values(),'degree')},
        'fanin_fanout':{'nodes_with_fanin_at_least_two':[u for u in g if indegrees[u]>=2],'nodes_with_fanout_at_least_two':[u for u in g if outdegrees[u]>=2],
            'max_fanin':max(indegrees.values(),default=0),'max_fanout':max(outdegrees.values(),default=0)},
        'components':{'weak_count':len(weak),'strong_count':len(strong),'weak':weak_rows,'strong':strong_rows},
        'reachability':{'ordered_distinct_pairs':reachable,'possible_ordered_distinct_pairs':possible,'fraction':reachable/possible if possible else None,
            'mean_shortest_path_reached':math.fsum(lengths)/reachable if reachable else None,'distance_distribution':_distribution(lengths,'distance'),
            'maximum_finite_directed_distance':max(lengths,default=None),
            'directed_diameter':strong_rows[0]['directed_diameter'] if len(strong)==1 else None,
            'undirected_projection_diameter':weak_rows[0]['undirected_projection_diameter'] if len(weak)==1 else None,
            'scope':'Self-pairs excluded. Unreachable pairs are not zero-distance observations. Directed diameter requires nonempty strong connectivity; projection diameter requires nonempty weak connectivity.'},
        'triangles_cycles':_cycle_metrics(g,undirected,cycle_limit,distances,is_dag),'reconvergence':reconvergence,
        'undirected_projection':{'n_edges':undirected.number_of_edges(),'articulation_nodes':sorted(articulation),'bridges':[list(e) for e in sorted(bridges)],
            'scope':'Reciprocal edges collapse; these are undirected cut structures, not automatically directed failure criticality.'},
        'centralization':{'in_degree':centralization(list(indegrees.values()),(n-1)**2 if n>1 else 0),
            'out_degree':centralization(list(outdegrees.values()),(n-1)**2 if n>1 else 0),
            'directed_total_degree':centralization(list(totaldegrees.values()),2*(n-1)*(n-2) if n>2 else 0),
            'undirected_projection_degree':centralization([d for _,d in undirected.degree()],(n-1)*(n-2) if n>2 else 0),
            'scope':'Freeman sum(max degree - degree), divided by maximum possible for a simple graph of this size. Directed in/out use (n-1)^2; directed total uses 2(n-1)(n-2); undirected uses (n-1)(n-2). Undefined zero denominators are null.'},
        'centrality_settings':{'pagerank_alpha':.85,'pagerank_converged':converged,'pagerank_iterations':iterations,
            'pagerank_scope':'Uniform teleportation/dangling redistribution; no weights. Nonconvergence produces null node scores.',
            'degree_scope':'in/out divided by n-1; total divided by 2(n-1), bounded in [0,1]. Unlike NetworkX total-degree centrality on digraphs, this never reaches 2.',
            'betweenness_scope':'Exact normalized unweighted directed shortest-path betweenness, excluding endpoints; edge betweenness uses directed normalization.'},
        'edge_pair_structure':{'all_unordered_edge_pairs':total_pairs,'directed_two_edge_walks':walks,'simple_directed_two_paths':simple_paths,
            'reciprocal_edge_pairs':reciprocal_dyads,'shared_receiver_pairs':receiver_pairs,'shared_sender_pairs':sender_pairs,
            'shared_endpoint_pairs':endpoint_pairs,'endpoint_disjoint_pairs':total_pairs-endpoint_pairs,
            'not_simple_two_path_pairs':total_pairs-simple_paths,'not_shared_receiver_pairs':total_pairs-receiver_pairs,
            'scope':'Pairs are unordered distinct edges. Simple two-paths have three distinct vertices; reciprocal dyads are separate. Two-edge walks include the two closed walks per reciprocal dyad.'},
        'communities':communities,'automorphism':_automorphisms(g,automorphism_node_limit,automorphism_mapping_limit),
        'node_metrics':node_metrics,'edge_metrics':edge_metrics}


def isomorphism_groups(graphs):
    """Group mapping[name, anonymous graph] by exact directed graph isomorphism.

    IDs identify output members, never graph features. WL(n,m,hash) buckets avoid
    unnecessary checks but every successful class merge uses nx.is_isomorphic.
    Class IDs depend on sorted input names, not canonical graph labeling. Exact
    checks have no wall-time guarantee; caller should deduplicate labeled graphs.
    """
    if not isinstance(graphs,Mapping) or any(type(name) is not str for name in graphs):raise ValueError('mapping of string names to graphs required')
    buckets={};groups=[]
    for name in sorted(graphs):
        g=validate_graph(graphs[name]);bucket=(len(g),g.number_of_edges(),_wl_bucket(g));match=None
        for group in buckets.get(bucket,[]):
            if nx.is_isomorphic(g,group['_graph']):match=group;break
        if match is None:
            match={'class_id':f'iso_{len(groups):04d}','representative':name,'members':[],'n_nodes':len(g),'n_edges':g.number_of_edges(),
                'wl_bucket':bucket[2],'wl_iterations':3,'networkx_version':nx.__version__,'verification':'exact directed VF2; WL is screening only','_graph':g}
            groups.append(match);buckets.setdefault(bucket,[]).append(match)
        match['members'].append(name)
    return [{k:v for k,v in group.items() if k!='_graph'} for group in groups]


graph_metrics=analyze_graph
