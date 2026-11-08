from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from record_workflow import WorkflowRun,case
from test_record_workflow import MechanicalProvider
from record_policy_extensions import enumerated_bundle_policies,receipt_origin_rule


def test_exact_bundle_expectation_is_half_even_when_seed_draws_are_lucky():
    r=WorkflowRun(case(123),'flat','combined',MechanicalProvider()).run()
    ps=list(enumerated_bundle_policies(r,7));truth=set(map(tuple,r['labels']['direct_false_edges']))
    assert len(ps)==140  # choose 3 of 6 pairs, then 1 of 7 remaining edges
    covered=0
    for p in ps:
        assert len(p['selected'])==7 and len(set(e for g in p['groups'] for e in g))==7
        covered+=sum(e in truth for g in p['groups'] if len(g)==2 for e in g)
    assert covered/(len(ps)*len(truth))==.5
    risk=list(enumerated_bundle_policies(r,7,True))
    assert all(truth<={e for g in p['groups'] if len(g)==2 for e in g} for p in risk)


def test_receipt_origin_rule_does_not_blame_propagation():
    r=WorkflowRun(case(123),'sequential','combined',MechanicalProvider()).run()
    es=list(map(tuple,r['graph']['edges']));scores=receipt_origin_rule(r,es)
    truth=set(map(tuple,r['labels']['direct_false_edges']))
    assert {e for e,s in scores.items() if s}==truth
    r['labels']={'SECRET':True}
    assert receipt_origin_rule(r,es)==scores
