from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import networkx as nx
import pytest
from network_strategies import SELECTORS,GROUPERS,select,group,allocations

@pytest.mark.parametrize('selector',SELECTORS)
def test_selection_is_graph_only_budgeted_and_repeatable(selector):
    g=nx.DiGraph([(0,1),(0,2),(1,3),(2,3),(3,4),(4,0)])
    for k in (0,1,3,6,20):
        es=select(g,k,selector,37)
        assert len(es)==min(k,6) and len(set(es))==len(es)
        assert set(es)<=set(g.edges()) and es==select(g,k,selector,37)

@pytest.mark.parametrize('grouper',GROUPERS)
def test_variable_groups_obey_cap_and_preserve_selection(grouper):
    g=nx.DiGraph([(0,1),(2,1),(3,1),(1,4),(4,5),(5,0)])
    es=list(g.edges())
    for s in (1,2,3,6):
        groups=group(g,es,s,grouper,42)
        if grouper=='all_selected' and s<6: assert groups is None;continue
        assert sorted(e for c in groups for e in c)==sorted(es)
        assert all(0<len(c)<=s for c in groups)

def test_receiver_retains_unequal_sizes():
    g=nx.DiGraph([(0,3),(1,3),(2,3),(3,4)])
    assert sorted(map(len,group(g,list(g.edges()),3,'receiver')))==[1,3]

def test_no_silent_full_view_when_infeasible():
    g=nx.DiGraph([(0,1),(1,2)])
    assert group(g,list(g.edges()),1,'all_selected') is None

def test_caps_not_confused_with_edge_budget():
    g=nx.DiGraph([(0,1),(0,2),(1,3),(2,3)])
    for a in allocations(g,budgets=(2,4),caps=(1,2,4)):
        assert len(a['selected'])==a['k']
        assert all(len(c)<=a['s'] for c in a['groups'])
