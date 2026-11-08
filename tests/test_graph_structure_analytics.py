"""Analytic fixtures: no agent data, provider calls or semantic graph features."""
import itertools
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import graph_structure_analytics as ga


def graph(n,edges):return {'nodes':list(range(n)),'edges':[list(e) for e in edges]}


def test_chain_directed_distance_is_not_undirected_or_strong_diameter():
    a=ga.analyze_graph(graph(3,[(0,1),(1,2)]))
    assert (a['n_nodes'],a['n_edges'],a['density'])==(3,2,pytest.approx(1/3))
    assert a['is_dag'] and a['dag_longest_path_edges']==2
    assert a['reachability']['ordered_distinct_pairs']==3
    assert a['reachability']['mean_shortest_path_reached']==pytest.approx(4/3)
    assert a['reachability']['directed_diameter'] is None
    assert a['reachability']['maximum_finite_directed_distance']==2
    assert a['reachability']['undirected_projection_diameter']==2
    assert a['components']['strong_count']==3
    assert a['undirected_projection']['articulation_nodes']==[1]
    assert a['undirected_projection']['bridges']==[[0,1],[1,2]]
    assert a['node_metrics'][1]['betweenness']==.5
    assert a['edge_metrics'][0]['betweenness']==pytest.approx(1/3)
    assert a['automorphism']['automorphism_count']==1


def test_diamond_reconvergence_is_multiple_paths_not_just_indegree():
    a=ga.analyze_graph(graph(4,[(0,1),(0,2),(1,3),(2,3)]))
    assert a['reconvergence']['ordered_pairs_with_multiple_paths']==1
    assert a['reconvergence']['fraction_of_reachable_pairs']==pytest.approx(1/5)
    assert a['node_metrics'][3]['ancestors_with_multiple_paths']==1
    assert a['fanin_fanout']['nodes_with_fanin_at_least_two']==[3]
    p=a['edge_pair_structure']
    assert p['simple_directed_two_paths']==2 and p['shared_sender_pairs']==1 and p['shared_receiver_pairs']==1
    assert p['endpoint_disjoint_pairs']==2
    assert a['automorphism']['node_orbits']==[[0],[3],[1,2]]
    assert a['automorphism']['automorphism_count']==2
    independent_sources=ga.analyze_graph(graph(3,[(0,2),(1,2)]))
    assert independent_sources['fanin_fanout']['max_fanin']==2
    assert independent_sources['reconvergence']['ordered_pairs_with_multiple_paths']==0


def test_reciprocal_edges_cycles_and_projection_have_distinct_meanings():
    a=ga.analyze_graph(graph(2,[(0,1),(1,0)]))
    assert not a['is_dag'] and a['dag_longest_path_edges'] is None
    assert a['reachability']['directed_diameter']==1
    assert a['triangles_cycles']['directed_girth']==2
    assert a['triangles_cycles']['simple_cycle_enumeration']['count']==1
    assert a['edge_pair_structure']['directed_two_edge_walks']==2
    assert a['edge_pair_structure']['simple_directed_two_paths']==0
    assert a['edge_pair_structure']['reciprocal_edge_pairs']==1
    assert a['undirected_projection']['n_edges']==1
    assert a['reconvergence']['status']=='undefined_for_cyclic_graph'
    assert a['centralization']['directed_total_degree'] is None


def test_directed_and_undirected_triangles_not_conflated():
    transitive=ga.analyze_graph(graph(3,[(0,1),(1,2),(0,2)]))
    assert transitive['triangles_cycles']['undirected_projection_triangles']==1
    assert transitive['triangles_cycles']['directed_three_cycles']==0
    assert transitive['triangles_cycles']['simple_cycle_enumeration']['count']==0
    assert transitive['reconvergence']['ordered_pairs_with_multiple_paths']==1
    full=ga.analyze_graph(graph(3,list(itertools.permutations(range(3),2))))
    assert full['triangles_cycles']['directed_three_cycles']==2
    assert full['triangles_cycles']['simple_cycle_enumeration']['count']==5


def test_disconnected_empty_and_singleton_denominators_are_explicit():
    a=ga.analyze_graph(graph(5,[(0,1),(1,2),(3,4)]))
    assert a['components']['weak_count']==2
    assert a['reachability']['undirected_projection_diameter'] is None
    assert [c['undirected_projection_diameter'] for c in a['components']['weak']]==[2,1]
    assert a['reachability']['fraction']==pytest.approx(4/20)
    for n in (0,1):
        a=ga.analyze_graph(graph(n,[]),include_communities=True)
        assert a['density']==0 and a['dag_longest_path_edges']==0
        assert a['reachability']['fraction'] is None
        assert a['reachability']['mean_shortest_path_reached'] is None
        assert a['centralization']['in_degree'] is None
        assert a['communities']['modularity'] is None
        assert a['automorphism']['automorphism_count']==1
        json.dumps(a,allow_nan=False)
    assert ga.analyze_graph(graph(1,[]))['node_metrics'][0]['pagerank']==1


def test_degree_centralization_and_pagerank_conventions():
    outward=ga.analyze_graph(graph(4,[(0,i) for i in (1,2,3)]))
    assert outward['centralization']['out_degree']==1
    assert outward['centralization']['in_degree']==pytest.approx(1/9)
    assert outward['centralization']['directed_total_degree']==.5
    assert outward['centralization']['undirected_projection_degree']==1
    assert sum(n['pagerank'] for n in outward['node_metrics'])==pytest.approx(1)
    assert outward['node_metrics'][0]['pagerank']<outward['node_metrics'][1]['pagerank']
    bidirectional=ga.analyze_graph(graph(4,[(0,i) for i in (1,2,3)]+[(i,0) for i in (1,2,3)]))
    assert bidirectional['centralization']['directed_total_degree']==1
    assert bidirectional['node_metrics'][0]['total_degree_centrality']==1


def test_caps_never_turn_partial_cycle_or_automorphism_search_into_exact_claims():
    a=ga.analyze_graph(graph(4,[]),automorphism_mapping_limit=2)
    assert a['automorphism']['status']=='unknown_mapping_limit'
    assert a['automorphism']['automorphism_count'] is None
    assert a['automorphism']['automorphism_count_lower_bound']==3
    assert a['automorphism']['node_orbits'] is None and a['automorphism']['edge_orbits'] is None
    skipped=ga.analyze_graph(graph(4,[]),automorphism_node_limit=3)['automorphism']
    assert skipped['status']=='unknown_node_limit' and skipped['enumerated_mappings']==0
    a=ga.analyze_graph(graph(4,list(itertools.permutations(range(4),2))),cycle_limit=2)
    assert not a['triangles_cycles']['simple_cycle_enumeration']['complete']
    assert a['triangles_cycles']['simple_cycle_enumeration']['count'] is None
    assert a['triangles_cycles']['simple_cycle_enumeration']['count_lower_bound']==3
    assert a['triangles_cycles']['directed_three_cycles']==8 # exact despite cycle cap


def test_wl_collision_is_not_isomorphism_proof_and_relabelings_are_grouped():
    cycle=graph(6,[(i,(i+1)%6) for i in range(6)])
    triangles=graph(6,[(0,1),(1,2),(2,0),(3,4),(4,5),(5,3)])
    renamed={'nodes':[10+i for i in range(6)],'edges':[[10+u,10+v] for u,v in cycle['edges']]}
    groups=ga.isomorphism_groups({'a_cycle':cycle,'b_renamed':renamed,'c_triangles':triangles})
    assert len(groups)==2 and groups[0]['members']==['a_cycle','b_renamed']
    assert groups[0]['wl_bucket']==groups[1]['wl_bucket'] # actual regular-graph WL collision
    assert groups[1]['members']==['c_triangles']


def test_isomorphism_comparison_still_runs_if_hash_function_collides(monkeypatch):
    monkeypatch.setattr(ga,'_wl_bucket',lambda g:'forced-collision')
    groups=ga.isomorphism_groups({'chain':graph(4,[(0,1),(1,2),(2,3)]),'star':graph(4,[(0,1),(0,2),(0,3)])})
    assert len(groups)==2


def test_unordered_edge_pair_partition_by_bruteforce():
    # Exhaust all64 three-node digraphs, including reciprocal edges and cycles.
    universe=list(itertools.permutations(range(3),2))
    for mask in range(64):
        es=[e for i,e in enumerate(universe) if mask&(1<<i)]
        p=ga.analyze_graph(graph(3,es))['edge_pair_structure']
        disjoint=sum(not set(e)&set(f) for e,f in itertools.combinations(es,2))
        receiver=sum(e[1]==f[1] for e,f in itertools.combinations(es,2))
        sender=sum(e[0]==f[0] for e,f in itertools.combinations(es,2))
        simple=sum(len(set(e+f))==3 and (e[1]==f[0] or f[1]==e[0]) for e,f in itertools.combinations(es,2))
        assert (p['endpoint_disjoint_pairs'],p['shared_receiver_pairs'],p['shared_sender_pairs'],p['simple_directed_two_paths'])==(disjoint,receiver,sender,simple)
        assert p['shared_endpoint_pairs']+p['endpoint_disjoint_pairs']==p['all_unordered_edge_pairs']


@pytest.mark.parametrize('bad',[
    {'nodes':[0,1],'edges':[[0,1]],'roles':{}}, {'nodes':['worker'],'edges':[]},
    {'nodes':[True],'edges':[]}, {'nodes':[0,0],'edges':[]},graph(1,[(0,0)]),
    graph(1,[(0,2)]),graph(2,[(0,1),(0,1)]),{'nodes':[0,1],'edges':[[0,1,2]]}])
def test_semantic_metadata_and_nonsimple_graphs_rejected(bad):
    with pytest.raises(ValueError):ga.analyze_graph(bad)


def test_relabeling_preserves_structural_statistics_and_mapped_centralities():
    original=graph(5,[(0,1),(0,2),(1,3),(2,3),(3,4)])
    rename={0:49,1:-7,2:80,3:13,4:2}
    relabeled={'nodes':list(reversed(list(rename.values()))),'edges':[[rename[u],rename[v]] for u,v in reversed(original['edges'])]}
    a,b=ga.analyze_graph(original),ga.analyze_graph(relabeled)
    for key in ('n_nodes','n_edges','density','is_dag','dag_longest_path_edges','degree_distribution','centralization','edge_pair_structure','reconvergence'):
        assert a[key]==b[key]
    mapped={r['node']:r for r in b['node_metrics']}
    for row in a['node_metrics']:
        other=mapped[rename[row['node']]]
        for key in ('pagerank','betweenness','in_degree','out_degree','reachable_other_nodes','ancestors_with_multiple_paths'):
            assert row[key]==pytest.approx(other[key])
    assert a['graph_sha256']!=b['graph_sha256'] # labeled identity deliberately separate from iso class


def test_optional_communities_are_projection_only_and_modularity_is_scoped():
    a=ga.analyze_graph(graph(6,[(0,1),(1,2),(2,0),(3,4),(4,5),(5,3)]),include_communities=True)
    assert a['communities']['communities']==[[0,1,2],[3,4,5]]
    assert a['communities']['modularity']==pytest.approx(.5)
    assert 'Heuristic' in a['communities']['scope']
