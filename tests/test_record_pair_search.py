import copy
from itertools import combinations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import numpy as np
from record_pair_search import pair_features, optimize, small_views, structural_policy
from record_allocation import view_key
from record_workflow import WorkflowRun, case
from test_record_workflow import MechanicalProvider


def test_pair_features_do_not_read_content_labels_or_field_roles():
    r = WorkflowRun(case(123), 'sequential', 'combined', MechanicalProvider()).run()
    other = copy.deepcopy(r); other['labels'] = {'SECRET': 1}; other['messages'] = []
    other['graph']['roles'] = {'SECRET': 1}
    for g in small_views(r): assert np.array_equal(pair_features(r, g), pair_features(other, g))


def test_pair_features_invariant_to_reporter_relabeling():
    r = WorkflowRun(case(123), 'flat', 'combined', MechanicalProvider()).run()
    mapping = {n: f'anon_{i}' if n.startswith('reporter') else n for i, n in enumerate(r['graph']['nodes'])}
    other = copy.deepcopy(r)
    other['graph']['nodes'] = [mapping[n] for n in r['graph']['nodes']]
    other['graph']['edges'] = [[mapping[u], mapping[v]] for u, v in r['graph']['edges']]
    for g in small_views(r):
        assert np.array_equal(pair_features(r, g), pair_features(other, [(mapping[u], mapping[v]) for u, v in g]))


def test_optimizer_matches_independent_bruteforce_and_handles_negative_rewards():
    es = [('a','b'), ('b','c'), ('c','d'), ('d','e')]
    r = {'graph': {'nodes': list('abcde'), 'edges': es}}
    views = small_views(r); rng = np.random.default_rng(3)
    weights = {view_key(g): float(rng.normal()) for g in views}
    for k in range(5):
        feasible = []
        for count in range(5):
            for groups in combinations(views, count):
                edges = [e for g in groups for e in g]
                if len(edges) == k and len(set(edges)) == k:
                    feasible.append(sum(weights[view_key(g)] for g in groups))
        result = optimize(r, k, weights)
        assert abs(sum(weights[view_key(g)] for g in result) - max(feasible)) < 1e-8
    negative={view_key(g):-1. for g in views}
    assert optimize(r,4,negative,exact=False)==[]
    assert sum(map(len,optimize(r,4,negative,exact=True)))==4


def test_global_policy_obeys_budget_with_no_labels():
    r = WorkflowRun(case(123), 'hierarchical', 'combined', MechanicalProvider()).run()
    r.pop('labels'); r.pop('messages')
    for k in (4, 8, 15):
        p = structural_policy(r, k)
        assert len(set(e for g in p['groups'] for e in g)) == k
        assert all(1 <= len(g) <= 2 for g in p['groups'])
