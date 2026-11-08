from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from record_audit_workloads import workload,preflight
from record_audit_analysis import rule_flags
from monitor_record_workflows import view_prompt


def test_multicase_graph_labels_and_observational_twins():
    for route in ('aligned','shuffled'):
        rec=workload(1,4,'false_report',route)
        twin=workload(1,4,'honest_error',route)
        edges=list(map(tuple,rec['graph']['edges']))
        assert len(edges)==13 and len(rec['labels']['case_ids'])==4
        assert len({m['id'] for m in rec['messages']})==len(rec['messages'])
        assert view_prompt(rec,edges)==view_prompt(twin,edges)
        assert len(rec['labels']['false_origin_messages'])==1
        assert rule_flags(rec,edges)==set(rec['labels']['false_origin_edges'])
        for cl in rec['labels']['case_labels'].values():
            assert all(e in edges for e in cl['reference_edges'].values())
            assert all(e in edges for e in cl['report_edges'].values())


def test_larger_workload_has_distinct_cases_not_padding():
    one=workload(0,1);many=workload(0,16)
    assert len(many['labels']['case_ids'])==16
    assert len(many['messages'])>len(one['messages'])*15
    edges=list(map(tuple,many['graph']['edges']))
    assert rule_flags(many,edges)==set(many['labels']['false_origin_edges'])
    # Measure exact serialized bytes rather than claiming a model token capacity.
    from run_record_audit_diagnostics import CLARIFIED
    from monitor_record_workflows import FORMAT_RETRY
    n=lambda r:len((CLARIFIED+FORMAT_RETRY+view_prompt(r,list(map(tuple,r['graph']['edges'])))).encode())
    assert n(one)<24000<n(many)


def test_target_history_crossed_independently_of_field_and_direction():
    for i in range(12):
        for history,revision in [('stable',1),('updated',2)]:
            r=workload(i,4,target_history=history)
            mids=set(r['labels']['false_origin_messages'])
            target=r['labels']['target_case_id'];cl=r['labels']['case_labels'][target]
            wanted=2 if history=='updated' and cl['update_field']==r['labels']['target_field'] else 1
            assert {m['hidden']['revision'] for m in r['messages'] if m['id'] in mids}=={wanted}
            assert sum(m['trusted'] and m['hidden']['revision']==2 and m['id'].startswith(target+':') for m in r['messages'])==int(history=='updated')
            assert r['history']==history


def test_revision_field_does_not_identify_target_and_truth_labels_hold():
    updates=[]
    for i in range(12):
        r=workload(i,4,target_history='updated')
        cl=r['labels']['case_labels'][r['labels']['target_case_id']]
        updates.append(cl['update_field']==r['labels']['target_field'])
        assert rule_flags(r,list(map(tuple,r['graph']['edges'])))==set(r['labels']['false_origin_edges'])
    assert any(updates) and not all(updates)
