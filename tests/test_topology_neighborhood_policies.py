"""Offline anonymous-graph allocation and exact/MC randomization contracts."""
from collections import Counter
import copy
from itertools import combinations
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import topology_neighborhood_policies as p

GRAPH={'nodes':list(range(8)),'edges':[[0,2],[1,2],[2,3],[3,4],[3,5],[4,6],[5,6],[6,7],[7,3]]}
FANIN={'nodes':[0,1,2,3],'edges':[[0,2],[1,2],[2,3]]}
def flat(groups):return [tuple(e) for g in groups for e in g]


@pytest.mark.parametrize('field',['roles','trust','labels','messages','lengths','sink','node_attrs','edge_attrs'])
def test_all_allocation_entries_reject_metadata(field):
    bad=copy.deepcopy(GRAPH);bad[field]={'secret':'not allowed'}
    for call in [lambda:p.select(bad,3,3),lambda:p.group(bad,[],3),lambda:p.allocate(bad,3,3),
                 lambda:p.parent_neighborhoods(bad),lambda:p.bounded_cones(bad),
                 lambda:p.uniform_selection_reference(bad,3),lambda:p.matched_grouping_reference(bad,[],[],3)]:
        with pytest.raises(ValueError):call()


@pytest.mark.parametrize('bad',[
    {'nodes':[False,1],'edges':[[False,1]]},
    {'nodes':['agent','judge'],'edges':[['agent','judge']]},
    {'nodes':[0,1],'edges':[[0,1,{'trusted':True}]]},
    {'nodes':[0,1],'edges':[[0,1],[0,1]]}])
def test_rejects_named_or_annotated_graphs(bad):
    with pytest.raises(ValueError):p.allocate(bad,1,1)


def test_parent_complete_never_drops_one_required_input_to_fit_cap():
    motifs=p.parent_neighborhoods(FANIN)
    assert motifs==[{'edges':[[0,2],[1,2],[2,3]],'anchor_outputs':[[2,3]],'depth':1}]
    large=p.allocate(FANIN,3,3)
    assert large['groups']==[[[0,2],[1,2],[2,3]]]
    assert large['structural_diagnostics']['parent_anchors_coobserved']==1
    small=p.allocate(FANIN,3,2)
    assert len(small['groups'])==3 and small['structural_diagnostics']['parent_anchors_coobserved']==0
    assert small['structural_diagnostics']['parent_candidates_fitting_group_cap']==0
    assert small['unused_edge_budget']==0


def test_bounded_cones_include_entire_predecessor_layers_and_terminate_on_cycles():
    graph={'nodes':list(range(5)),'edges':[[0,2],[1,2],[2,3],[3,4]]}
    cones=p.bounded_cones(graph,2)
    top=next(m for m in cones if [3,4] in m['anchor_outputs'])
    assert top['edges']==[[0,2],[1,2],[2,3],[3,4]]
    cycle={'nodes':[0,1,2],'edges':[[0,1],[1,2],[2,0]]}
    assert len(p.bounded_cones(cycle,100000))==1
    assert p.bounded_cones(cycle,100000)[0]['edges']==cycle['edges']


def test_all_methods_exact_edge_budget_and_disjoint_group_caps():
    for selector in p.SELECTORS:
        for k in (0,1,4,9):
            for s in (1,2,4,9):
                allocation=p.allocate(GRAPH,k,s,selector,seed=19)
                assert len(allocation['selected'])==len({tuple(e) for e in allocation['selected']})==k
                assert all(type(n) is int for e in allocation['selected'] for n in e)
                for grouper in p.GROUPERS:
                    groups=p.group(GRAPH,allocation['selected'],s,grouper,seed=19)
                    if grouper=='full_selected' and k>s:assert groups is None
                    else:
                        assert sorted(flat(groups))==sorted(map(tuple,allocation['selected']))
                        assert len(flat(groups))==len(set(flat(groups)))
                        assert all(1<=len(g)<=s for g in groups)


def test_canonical_serialization_and_motif_relabeling():
    reordered={'nodes':list(reversed(GRAPH['nodes'])),'edges':list(reversed(GRAPH['edges']))}
    mapping={i:100-7*i for i in GRAPH['nodes']}
    renamed={'nodes':list(mapping.values()),'edges':[[mapping[u],mapping[v]] for u,v in GRAPH['edges']]}
    for selector in p.SELECTORS:assert p.allocate(GRAPH,6,4,selector,seed=31)==p.allocate(reordered,6,4,selector,seed=31)
    for depth in (1,2,4):
        lhs={frozenset((mapping[u],mapping[v]) for u,v in m['edges']) for m in p.bounded_cones(GRAPH,depth)}
        rhs={frozenset(map(tuple,m['edges'])) for m in p.bounded_cones(renamed,depth)}
        assert lhs==rhs


def test_tied_motifs_do_not_always_choose_low_ids_or_a_single_branch():
    graph={'nodes':list(range(6)),'edges':[[0,1],[1,2],[3,4],[4,5]]}
    counts=Counter(tuple(map(tuple,p.select(graph,2,2,seed=seed))) for seed in range(800))
    assert len(counts)==2
    assert all(340<n<460 for n in counts.values())
    mapping={0:30,1:20,2:10,3:-1,4:-2,5:-3}
    renamed={'nodes':list(mapping.values()),'edges':[[mapping[u],mapping[v]] for u,v in graph['edges']]}
    reverse={v:k for k,v in mapping.items()}
    counts2=Counter(tuple(sorted((reverse[u],reverse[v]) for u,v in p.select(renamed,2,2,seed=seed))) for seed in range(800))
    assert set(counts2)==set(counts) and all(340<n<460 for n in counts2.values())


def test_rotation_is_fixed_coverage_schedule_not_independent_draws():
    counts=Counter()
    for step in range(9):counts.update(map(tuple,p.select(GRAPH,4,2,'rotation',seed=8,rotation_step=step)))
    assert set(counts.values())=={4}
    assert p.select(GRAPH,4,2,'rotation',seed=8,rotation_step=0)==p.select(GRAPH,4,2,'rotation',seed=8,rotation_step=9)


def test_uniform_exact_acquisition_and_declared_mc_with_duplicates():
    ref=p.uniform_selection_reference(FANIN,2)
    assert ref['mode']=='exact' and ref['universe_size']==ref['draws']==3
    assert {tuple(map(tuple,d['selected'])) for d in ref['selections']}==set(combinations(map(tuple,FANIN['edges']),2))
    with pytest.raises(ValueError,match='explicit Monte Carlo'):p.uniform_selection_reference(GRAPH,4,exact_limit=1)
    mc=p.uniform_selection_reference(FANIN,2,draws=900,exact_limit=0,seed=7)
    assert mc['mode']=='monte_carlo' and len(mc['selections'])==900
    counts=Counter(tuple(map(tuple,d['selected'])) for d in mc['selections'])
    assert len(counts)==3 and all(240<n<360 for n in counts.values())
    assert mc==p.uniform_selection_reference(FANIN,2,draws=900,exact_limit=0,seed=7)


@pytest.mark.parametrize('sizes,expected',[([1,2,2,2],105),([1,1,2],6),([2,2],3),([1,1,1],1),([],1)])
def test_exact_partition_census_counts_multiplicity_and_preserves_budget(sizes,expected):
    selected=GRAPH['edges'][:sum(sizes)];s=max(sizes,default=1)
    ref=p.matched_grouping_reference(GRAPH,selected,sizes,s)
    assert p.partition_count(sizes)==expected==len(ref['partitions'])
    keys={tuple(tuple(map(tuple,g)) for g in d['groups']) for d in ref['partitions']}
    assert len(keys)==expected
    for d in ref['partitions']:
        assert sorted(flat(d['groups']))==sorted(map(tuple,selected))
        assert sorted(map(len,d['groups']))==sorted(sizes)
    weights=p.candidate_group_weights(ref)
    assert sum(len(w['edges'])*w['weight_numerator']/w['weight_denominator'] for w in weights)==pytest.approx(len(selected))
    assert sum(w['weight_numerator']/w['weight_denominator'] for w in weights)==pytest.approx(len(sizes))


def test_uniform_matched_mc_and_full_partition_nonlinear_metric():
    selected=GRAPH['edges'][:4];ref=p.matched_grouping_reference(GRAPH,selected,[2,2],2,draws=900,exact_limit=0,seed=19)
    counts=Counter(tuple(tuple(map(tuple,g)) for g in d['groups']) for d in ref['partitions'])
    assert len(counts)==3 and all(240<n<360 for n in counts.values())
    exact=p.matched_grouping_reference(GRAPH,selected,[2,2],2)
    target=set(map(tuple,selected[:2]))
    chance=sum(any(target<=set(map(tuple,g)) for g in d['groups']) for d in exact['partitions'])/exact['draws']
    assert chance==pytest.approx(1/3)


def test_invalid_budget_or_partition_settings_raise():
    for call in [lambda:p.allocate(GRAPH,10,2),lambda:p.allocate(GRAPH,True,2),lambda:p.allocate(GRAPH,2,0),
        lambda:p.allocate(GRAPH,2,2,depth=False),lambda:p.allocate(GRAPH,2,2,seed=False),
        lambda:p.matched_grouping_reference(GRAPH,GRAPH['edges'][:3],[2,2],2),
        lambda:p.matched_grouping_reference(GRAPH,GRAPH['edges'][:3],[3],2),
        lambda:p.group(GRAPH,[[0,2],[0,2]],2),lambda:p.uniform_selection_reference(GRAPH,3,draws=False,exact_limit=0)]:
        with pytest.raises(ValueError):call()


def test_path_universe_is_exhaustive_bounded_and_cap_never_silently_truncates():
    graph={'nodes':[0,1,2,3],'edges':[[0,1],[1,2],[2,0],[2,3]]}
    universe=p.bounded_path_universe(graph,3,100)
    # Independently enumerate simple vertex sequences, not the implementation DFS.
    from itertools import permutations
    wanted=set()
    for length in (1,2,3):
        for nodes in permutations(graph['nodes'],length+1):
            edges=tuple(zip(nodes,nodes[1:]))
            if all(list(e) in graph['edges'] for e in edges):wanted.add(edges)
    assert {tuple(map(tuple,es)) for es in universe['paths']}==wanted
    assert universe['count']==len(wanted)
    with pytest.raises(ValueError,match='no truncated'):p.bounded_path_universe(graph,3,len(wanted)-1)
    bad=copy.deepcopy(graph);bad['critical_targets']=[3]
    with pytest.raises(ValueError):p.path_coverage_plan(bad,1)


def test_greedy_sensor_coverage_uses_or_not_complete_path_or_judge_reward():
    graph={'nodes':[0,1,2,3],'edges':[[0,1],[1,2],[2,3]]}
    result=p.path_coverage_plan(graph,1,seed=5,max_path_edges=3)
    assert result['selected']==[[1,2]]
    # Six paths total; middle edge hits four, but cannot observe any complete
    # two-edge relation by itself. Do not call surrogate coverage source recall.
    assert result['scenario_count']==6 and result['covered_scenarios']==4
    allocation=p.allocate(graph,1,2,'path_coverage',seed=5)
    assert allocation['structural_diagnostics']['parent_anchors_coobserved']==0
    assert allocation['path_coverage']==result


def test_path_coverage_and_scenario_family_are_relabeling_consistent_when_unique():
    graph={'nodes':[0,1,2,3],'edges':[[0,1],[1,2],[2,3]]}
    mapping={0:90,1:50,2:-1,3:4}
    renamed={'nodes':list(mapping.values()),'edges':[[mapping[u],mapping[v]] for u,v in graph['edges']]}
    for seed in range(12):assert p.path_coverage_plan(renamed,1,seed)['selected']==[[50,-1]]
    lhs={tuple((mapping[u],mapping[v]) for u,v in path) for path in p.bounded_path_universe(graph)['paths']}
    rhs={tuple(map(tuple,path)) for path in p.bounded_path_universe(renamed)['paths']}
    assert lhs==rhs
