import copy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from record_audit_controls import build_record,make_case,diagnostic_views
from record_audit_policies import policies,group_selected,measure


def test_policies_ignore_hidden_labels_and_respect_budgets():
    rec=build_record(make_case(0),'false_report','shuffled','prose_units','updated')
    other=copy.deepcopy(rec);other.pop('labels');other.pop('condition')
    for m in other['messages']:m.pop('hidden')
    ps=list(policies(rec,seeds=(0,),budgets=(7,)))
    assert ps==list(policies(other,seeds=(0,),budgets=(7,)))
    for p in ps:
        assert len(p['selected'])==7
        assert sorted(e for g in p['groups'] for e in g)==sorted(p['selected'])
        assert all(1<=len(g)<=p['s'] for g in p['groups'])


def test_unselected_contents_never_parsed_for_grouping():
    rec=build_record(make_case(0),'false_report','shuffled','prose_units')
    chosen=diagnostic_views(rec)['correct_reference']
    for m in rec['messages']:
        if (m['sender'],m['recipient']) not in chosen:m['content']='invalid unselected content'
    assert group_selected(rec,chosen,2,'observed_keys',0)==[sorted(chosen)]


def test_alignment_ablation_with_all_edges_separates_proximity_from_observed_matching():
    for route,want in [('aligned',1),('shuffled',0)]:
        rec=build_record(make_case(0),'false_report',route,'prose_units')
        for grouper in ('provenance','observed_keys'):
            p=next(p for p in policies(rec,seeds=(0,),budgets=(13,)) if p['selector']=='random' and p['s']==2 and p['grouper']==grouper)
            row=measure(rec,p)
            assert row['covered']==row['joined_tp']==1
            assert row['grouped_tp']==(want if grouper=='provenance' else 1)


def test_large_workload_marks_infeasible_llm_views_but_preserves_rule_result():
    from record_audit_workloads import workload
    from record_audit_policies import byte_profile
    rec=workload(0,16)
    profile=byte_profile(rec)
    ps=list(policies(rec,seeds=(0,),budgets=(13,)))
    full=next(p for p in ps if p['selector']=='random' and p['grouper']=='all_selected')
    pair=next(p for p in ps if p['selector']=='random' and p['grouper']=='provenance' and p['s']==2)
    a,b=measure(rec,full,profile),measure(rec,pair,profile)
    assert a['workload_size']==b['workload_size']==16
    assert not a['llm_byte_feasible'] and b['llm_byte_feasible']
    assert a['joined_tp']==a['grouped_tp']==1
