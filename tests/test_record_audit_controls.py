import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import json
from collections import Counter
from record_audit_controls import records, diagnostic_views, FIELDS
from monitor_record_workflows import view_prompt


def test_balanced_targets_and_versions():
    rs=records(12)
    assert len(rs)==288
    assert set(Counter(r['labels']['target_field'] for r in rs).values())=={48}
    for r in rs:
        false=r['condition']!='clean'
        assert len(r['labels']['false_origin_edges'])==int(false)
        assert len(r['labels']['malicious_edges'])==int(r['condition']=='false_report')
        assert len(r['messages'])==13+(r['history']=='updated')
        assert all([m['sender'],m['recipient']] in r['graph']['edges'] for m in r['messages'])


def test_attack_error_observational_identity_and_private_exclusion():
    rs=records(6);indexed={(r['case_id'],r['condition'],r['history'],r['environment'],r['rendering']):r for r in rs}
    for r in rs:
        if r['condition']!='false_report':continue
        twin=indexed[(r['case_id'],'honest_error',r['history'],r['environment'],r['rendering'])]
        for name,edges in diagnostic_views(r).items():
            assert edges==diagnostic_views(twin)[name]
            prompt=view_prompt(r,edges)
            assert prompt==view_prompt(twin,edges)
            obj=json.loads(prompt)
            assert set(obj)=={'workflow','channels','messages'}
            assert all(set(m)=={'id','channel','sender','recipient','order','protected_registry_receipt','content'} for m in obj['messages'])
            assert 'labels' not in obj and 'condition' not in obj


def test_routing_changes_location_not_fact_semantics():
    rs=records(6);indexed={(r['case_id'],r['condition'],r['history'],r['environment'],r['rendering']):r for r in rs}
    for r in rs:
        if r['environment']!='aligned':continue
        other=indexed[(r['case_id'],r['condition'],r['history'],'shuffled',r['rendering'])]
        assert sorted(m['content'] for m in r['messages'])==sorted(m['content'] for m in other['messages'])
        for f in FIELDS:
            assert r['labels']['reference_edges'][f][1]==r['labels']['report_edges'][f][0]
            assert other['labels']['reference_edges'][f][1]!=other['labels']['report_edges'][f][0]


def test_unit_rendering_roundtrip_and_rule_reference_limits():
    from record_audit_analysis import parse_claim,rule_flags
    for r in records(6):
        for m in r['messages']:
            parsed=parse_claim(m['content'])
            h=m['hidden']
            if h['field'] is None:assert parsed is None
            else:assert parsed==(r['case_id'],h['field'],h['revision'],h['value'])
        views=diagnostic_views(r)
        assert not rule_flags(r,views['report_only'])
        assert not rule_flags(r,views['unrelated_reference'])
        truth=set(map(tuple,r['labels']['false_origin_edges']))
        assert rule_flags(r,views['correct_reference'])==truth
        assert rule_flags(r,views['full_graph'])==truth


def test_unique_calls_share_confounded_truth_and_error_twins():
    from run_record_audit_diagnostics import make_plan
    rs=records(12);unique,assignments=make_plan(rs)
    assert len(assignments)==2304 and len(unique)==1344
    assert all(len({m['sender'] for m in r['messages']})==8 for r in rs)


def test_prompt_adoption_rejects_improvement_from_abstention_or_lost_detection():
    from record_audit_analysis import adoption
    pairs=[]
    for i in range(12):
        pairs.append({'case_id':str(i),'condition':'clean','view':'report_only','original_flag':True,
                      'clarified_flag':False,'original_abstention':False,'clarified_abstention':False})
        pairs.append({'case_id':str(i),'condition':'false_report','view':'correct_reference','original_flag':True,
                      'clarified_flag':True,'original_abstention':False,'clarified_abstention':False})
    assert adoption(pairs)['decision']=='clarified_candidate_for_fresh_evaluation'
    pairs[0]['clarified_abstention']=True
    assert adoption(pairs)['decision']=='retain_original_primary'
    pairs[0]['clarified_abstention']=False;pairs[1]['clarified_flag']=False
    assert adoption(pairs)['decision']=='retain_original_primary'
