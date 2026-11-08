import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pytest
from record_workflow import WorkflowRun, case
from test_record_workflow import MechanicalProvider
from record_allocation import library, counts, rates, policy_scores, measure, view_key
from record_strategies import policies
from monitor_record_workflows import edge_ids


def test_library_covers_every_pair_and_every_policy_group():
    r = WorkflowRun(case(123), 'flat', 'combined', MechanicalProvider()).run()
    views = set(library(r)); n = len(r['graph']['edges'])
    assert sum(len(g) == 2 for g in views) == n * (n - 1) // 2
    assert all(tuple(sorted(g)) in views for p in policies(r) for g in p['groups'])


def test_recall_includes_unselected_attacks_and_precision_can_be_undefined():
    c = rates(counts({1}, {1, 2}, {1, 3}, {1, 2, 3, 4}))
    assert c['recall'] == .5 and c['precision'] == 1 and c['inspected_fpr'] == 0
    assert rates(counts(set(), {1}, {2}, {1, 2}))['precision'] is None
    with pytest.raises(ValueError): counts({1}, {1}, {2}, {1, 2})


def test_source_and_propagated_content_and_cost_are_distinct():
    r = WorkflowRun(case(123), 'sequential', 'combined', MechanicalProvider()).run()
    es = sorted(map(tuple, r['graph']['edges'])); ids = edge_ids(r)
    truth = set(map(tuple, r['labels']['direct_false_edges']))
    propagated = set(map(tuple, r['labels']['propagated_false_edges'])) - truth
    assert propagated
    p = {'selector': 'test', 'grouper': 'all', 'k': len(es), 's': len(es), 'seed': 0,
         'selected': es, 'groups': [es]}
    judgments = {name: {cid: float(e in truth or e in propagated) for e, cid in ids.items()}
                 for name in ('source_scores', 'content_scores')}
    views = {view_key(es): {'judgment': judgments, 'usage': {'input_tokens': 12, 'output_tokens': 4, 'cost': .01}, 'request_count': 2}}
    scores = policy_scores(r, p, views); m = measure(r, p, scores)
    assert m['source']['fp'] == len(propagated) and m['content']['fp'] == 0
    assert m['requests'] == 2 and m['logical_calls'] == 1 and m['usage']['cost'] == .01
    del views[view_key(es)]['judgment']
    with pytest.raises(ValueError): policy_scores(r, p, views)
    views[view_key(es)]['abstention']={'reason':'invalid schema after retry'}
    scores=policy_scores(r,p,views);m=measure(r,p,scores)
    assert m['source']['tp']==0 and m['source']['n_truth']==2
    assert m['source_truth_abstained']==2 and m['failed_views']==1
    assert not scores['source']  # No numeric benign scores have been fabricated.
