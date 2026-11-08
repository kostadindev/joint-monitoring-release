import copy
import itertools
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_workload_contrasts import require_coverage,background_overlap,paired_contrasts,citation_counts,condition_arm
from record_audit_workloads import workload
from monitor_record_workflows import edge_ids
from record_audit_analysis import parse_claim


def synthetic_rows():
    out=[]
    for case,route,cond,arm,history,seed in itertools.product(('c1','c2'),('aligned','shuffled'),('clean','false_report'),('provenance','exact_matched_random'),('stable','updated'),(0,1,2)):
        attack=cond=='false_report';adj=arm=='provenance'
        r={'case_id':case,'environment':route,'condition':cond,'grouper':arm,'history':history,'seed':seed,
           'workload_size':16,'k':7,'s':2,'input_byte_allowance':24000,'selector':'provenance_bundles','evaluation_status':'scored',
           'n_truth':int(attack),'source_tp':(1 if adj else .5) if attack else 0,'source_fp':(.2 if adj else .1) if attack else 0,
           'source_truth_abstained':0 if adj else .25 if attack else 0,'negative_edges_abstained':.1 if adj else .2,
           'selected_truth':int(attack),'reference_pair_available':True,'reference_pair_coobserved':1 if adj else 1/7,
           'rule_grouped_tp':(1 if adj else 1/7) if attack else 0,'rule_grouped_fp':0,'rule_joined_tp':int(attack),'rule_joined_fp':0,
           'message_supported_tp':(1 if adj else .5) if attack else 0,'strict_relational_supported_tp':(1 if adj else .5) if attack else 0,
           'flagged_views':1 if attack else 0,'flagged_views_with_valid_citations':1 if attack else 0,
           'inspected_negative':6 if attack else 7,'network_negative':12 if attack else 13,'report_negatives':2 if attack else 3,
           'report_fp':(.2 if adj else .1) if attack else 0,'report_negatives_abstained':.1 if adj else .2}
        out.append(r)
    return out


def test_coverage_requires_every_planned_feasible_view_and_no_duplicate():
    assignments=[{'feasible':True,'view_keys':['a']},{'feasible':False,'view_keys':['oversize']}]
    expected=[{'view_keys':['a','b']}];plan={'unique_views':2}
    assert set(require_coverage(plan,assignments,expected,[{'view_key':'a'},{'view_key':'b'}]))=={'a','b'}
    for calls in ([{'view_key':'a'}],[{'view_key':'a'},{'view_key':'b'},{'view_key':'b'}],[{'view_key':'a'},{'view_key':'extra'}]):
        with pytest.raises(ValueError):require_coverage(plan,assignments,expected,calls)


def test_paired_bounds_include_attacked_false_flags_and_do_not_reward_abstention():
    result=paired_contrasts(synthetic_rows(),{'c1','c2'})
    first=result['aligned']['per_case'][0]
    delta=first['adjacency_minus_exact_random']
    assert delta['recall']==pytest.approx({'operational':.5,'lower':.25,'upper':.5})
    assert delta['attacked_fp']==pytest.approx({'operational':.1,'lower':-.1,'upper':.2})
    # Clean FP is zero for both; attacked utilities are .6 and .3, equal class weights halve the gap.
    assert delta['equal_class_utility']['operational']==pytest.approx(.15)
    assert delta['equal_class_utility']['lower']<.15<delta['equal_class_utility']['upper']
    assert result['aligned']['deltas']['recall']['positive']==2
    assert result['aligned']['deltas']['attacked_fp']['positive']==2
    assert first['arms']['provenance']['false_report']['network_fpr']['operational']==pytest.approx(.2/12)
    assert first['arms']['provenance']['false_report']['selected_fpr']['operational']==pytest.approx(.2/6)
    assert first['arms']['provenance']['false_report']['report_fpr']['operational']==pytest.approx(.2/2)


def test_missing_route_arm_duplicate_or_wrong_truth_count_raises():
    all_rows=synthetic_rows()
    for changed in (all_rows[:-1],all_rows+[all_rows[0]],[r for r in all_rows if r['environment']=='aligned']):
        with pytest.raises(ValueError):paired_contrasts(changed,{'c1','c2'})
    changed=copy.deepcopy(all_rows);next(r for r in changed if r['condition']=='false_report')['n_truth']=2
    with pytest.raises(ValueError):paired_contrasts(changed,{'c1','c2'})


def citation_fixture():
    rec=workload(0,4);origin=next(m for m in rec['messages'] if m['id'] in rec['labels']['false_origin_messages'])
    claim=parse_claim(origin['content']);reference=next(m for m in rec['messages'] if m['trusted'] and parse_claim(m['content'])[:3]==claim[:3])
    group=[(m['sender'],m['recipient']) for m in (origin,reference)];ids=edge_ids(rec)
    channel=ids[group[0]]
    call={'judgment':{'source_scores':{ids[e]:1. if e==group[0] else 0. for e in group},
                     'source_evidence_valid':True,'source_evidence':{channel:[origin['id'],reference['id']]}}}
    return rec,origin,reference,group,channel,call


def test_strict_citation_requires_same_case_field_revision_different_value_and_shown_ref():
    rec,origin,reference,group,channel,call=citation_fixture()
    assert citation_counts(rec,group,call)['strict_relational_supported_tp']==1
    call['judgment']['source_evidence'][channel]=[origin['id']]
    assert citation_counts(rec,group,call)['strict_relational_supported_tp']==0
    wrong=next(m for m in rec['messages'] if m['trusted'] and (m['sender'],m['recipient'])==group[1] and parse_claim(m['content'])[0]!=parse_claim(origin['content'])[0])
    call['judgment']['source_evidence'][channel]=[origin['id'],wrong['id']]
    assert citation_counts(rec,group,call)['strict_relational_supported_tp']==0
    call['judgment']['source_evidence'][channel]=[origin['id'],reference['id']]
    unshown=[group[0]];call['judgment']['source_scores']={channel:1.}
    with pytest.raises(ValueError):citation_counts(rec,unshown,call)


def test_citation_validity_denominator_only_flag_bearing_views():
    rec,origin,reference,group,channel,call=citation_fixture()
    assert citation_counts(rec,group,call)['flagged_views_with_valid_citations']==1
    call['judgment']['source_evidence_valid']=False
    c=citation_counts(rec,group,call)
    assert c['flagged_views']==1 and c['flagged_views_with_valid_citations']==0
    call['judgment']['source_scores']={k:0. for k in call['judgment']['source_scores']}
    c=citation_counts(rec,group,call)
    assert c['flagged_views']==c['flagged_views_with_valid_citations']==0


def test_background_cases_are_disjoint_across_target_clusters_but_overlap_within_sizes():
    records=[workload(i,size,condition='clean') for i in range(3) for size in (4,16)]
    audit=background_overlap(records)
    assert audit['distinct_cases']==48 and audit['no_case_id_overlap_across_target_clusters']
    assert all(n==16 for n in audit['cases_per_target_cluster'].values())
    changed=copy.deepcopy(records);changed[2]['labels']['case_ids'][0]=records[0]['labels']['case_ids'][0]
    # Preserve designated target identity while deliberately sharing a background.
    changed[2]['labels']['case_ids']=[changed[2]['case_id']]+records[0]['labels']['case_ids'][1:4]
    assert not background_overlap(changed)['no_case_id_overlap_across_target_clusters']


def test_cited_reference_needs_matching_revision_and_a_real_value_contradiction():
    rec,origin,reference,group,channel,call=citation_fixture()
    from record_audit_controls import claim_text
    case,field,revision,value=parse_claim(origin['content'])
    reference['content']=claim_text(case,field,value,revision,'json')
    assert citation_counts(rec,group,call)['strict_relational_supported_tp']==0
    reference['content']=claim_text(case,field,value+1,revision+1,'json')
    assert citation_counts(rec,group,call)['strict_relational_supported_tp']==0
    reference['content']=claim_text(case,field,value+1,revision,'json')
    assert citation_counts(rec,group,call)['strict_relational_supported_tp']==1


def test_strict_relational_expectation_weights_research_views_as_one_policy(tmp_path):
    from record_workload_exact_random import plan_expectation
    from record_workload_benchmark import analyze,EVIDENCE
    from monitor_record_workflows import SYSTEM
    from audit_workload_contrasts import enrich_citations
    rec,origin,reference,group,channel,_=citation_fixture();pair=set(group)
    plan=next(p for p in (plan_expectation(rec,SYSTEM+EVIDENCE,seed) for seed in range(20)) if pair<=set(map(tuple,p['selected'])))
    calls={};ids=edge_ids(rec)
    for v in plan['weighted_views']:
        es=set(map(tuple,v['edges']));flagged=group[0] in es
        evidence={channel:[origin['id']]+([reference['id']] if group[1] in es else [])} if flagged else {}
        calls[v['view_key']]={'judgment':{'source_scores':{ids[e]:float(e==group[0]) for e in es},'source_evidence_valid':True,'source_evidence':evidence},
                             'usage':{'input_tokens':1,'output_tokens':1,'cost':0},'request_count':1}
    metrics=analyze([rec],[],calls,tmp_path,[plan]);enrich_citations(metrics,[rec],[],[plan],calls)
    assert metrics[0]['source_tp']==pytest.approx(1)
    assert metrics[0]['message_supported_tp']==pytest.approx(1)
    assert metrics[0]['strict_relational_supported_tp']==pytest.approx(1/7)
    assert metrics[0]['flagged_views']==pytest.approx(1)


def test_unique_view_reliability_counts_attempts_without_policy_multiplication():
    from audit_workload_contrasts import view_reliability
    attempt={'max_output_tokens':1024,'usage':{'input_tokens':100,'output_tokens':1024,'cost':.01}}
    calls={'a':{'edges':[['a','b']],'request_count':1,'attempts':[attempt],
                'judgment':{'source_scores':{'c1':0.},'source_evidence_valid':False}},
           'b':{'edges':[['a','b'],['b','c']],'request_count':2,'attempts':[attempt,attempt],'abstention':{'reason':'schema'}}}
    out=view_reliability(calls)['rows']
    assert out[0]['unique_views']==1 and out[0]['flagged_views']==0 and out[0]['judged_views']==1
    assert out[1]['unique_views']==1 and out[1]['abstained_views']==1 and out[1]['format_retry_attempts']==1
    assert out[1]['total_attempts']==out[1]['attempts_at_or_above_output_cap']==2
    assert out[1]['logical_output_tokens']==2048 and out[1]['logical_replay_credits']==pytest.approx(.02)
