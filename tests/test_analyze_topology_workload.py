import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_topology_workload import (primary,paired_contrasts,require_complete,partitions,
    episode_alarm,research_spend,finite_oracles,analyze_complete)
from record_workflow import canonical
from monitor_record_workflows import edge_ids
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




def topology_rows():
    rows=synthetic_rows()
    for r in rows:
        r['selector']='structural_path_pairs'
        if r['grouper']=='provenance':r['grouper']='directed_path'
    return rows


def test_new_primary_does_not_accept_prior_semantic_policy():
    assert not any(primary(r) for r in synthetic_rows())
    assert all(primary(r) for r in topology_rows())
    r=topology_rows()[0];r['input_byte_allowance']=60000
    assert not primary(r)


def test_pairing_has_attacked_false_flags_bounds_and_conditionals():
    result=paired_contrasts(topology_rows(),{'c1','c2'})
    first=result['aligned']['per_case'][0];d=first['directed_path_minus_exact_random']
    assert d['recall']==pytest.approx({'operational':.5,'lower':.25,'upper':.5})
    assert d['attacked_fp']==pytest.approx({'operational':.1,'lower':-.1,'upper':.2})
    assert d['equal_class_utility']['operational']==pytest.approx(.15)
    assert first['conditional_pair_available']['paired_assignments']==6
    assert result['aligned']['conditional_pair_available_recall_delta']['cases']==2
    arms=first['arms']['directed_path']['false_report']
    assert arms['selected_fpr']['operational']==pytest.approx(.2/6)
    assert arms['network_fpr']['operational']==pytest.approx(.2/12)
    assert arms['report_fpr']['operational']==pytest.approx(.2/2)


def test_conditional_empty_is_undefined_and_missing_factorial_rejected():
    rows=topology_rows()
    for r in rows:r['reference_pair_available']=False
    result=paired_contrasts(rows,{'c1','c2'})
    assert result['aligned']['conditional_pair_available_recall_delta']['mean'] is None
    assert result['aligned']['per_case'][0]['conditional_pair_available']['recall_delta'] is None
    for bad in (rows[:-1],rows+[rows[0]]):
        with pytest.raises(ValueError):paired_contrasts(bad,{'c1','c2'})


def test_global_gate_requires_all_views_not_just_primary():
    assignments=[{'feasible':True,'view_keys':['primary']},{'feasible':True,'view_keys':['nonprimary']},
                 {'feasible':False,'view_keys':['infeasible']}]
    plan={'unique_views':2,'view_keys_sha256':hashlib.sha256(canonical(['nonprimary','primary']).encode()).hexdigest()}
    with pytest.raises(ValueError,match='no global ranking'):require_complete(plan,assignments,[],[{'view_key':'primary'}])
    assert len(require_complete(plan,assignments,[],[{'view_key':'primary'},{'view_key':'nonprimary'}]))==2
    with pytest.raises(ValueError,match='duplicate'):require_complete(plan,assignments,[],[{'view_key':'primary'}]*2)


def test_partition_enumeration_is_unlabeled_and_uniform():
    edges=[('a',str(i)) for i in range(7)]
    ps=list(partitions(edges,[1,2,2,2]))
    assert len(ps)==105
    assert len({frozenset(map(frozenset,p)) for p in ps})==105
    assert all(sorted(e for g in p for e in g)==edges for p in ps)


def alarm_fixture():
    edges=[('a',str(i)) for i in range(4)]
    rec={'graph':{'edges':edges},'labels':{'false_origin_edges':[]}}
    ids=edge_ids(rec);views=[];calls={}
    for n,g in enumerate(itertools.combinations(edges,2)):
        key=str(n);views.append({'edges':g,'view_key':key})
        calls[key]={'judgment':{'source_scores':{ids[e]:float(set(g)==set(edges[:2])) for e in g}}}
    plan={'selected':edges,'group_sizes':[2,2],'randomization_partitions':3,'weighted_views':views}
    return rec,plan,calls


def test_episode_alarm_not_additive_expected_fp_or_independence():
    rec,plan,calls=alarm_fixture()
    # Two flags occur together in one of three partitions: E[FP]=2/3, P(any)=1/3.
    result=episode_alarm(rec,plan,calls)
    assert result==pytest.approx({'operational':1/3,'lower':1/3,'upper':1/3})
    calls['1']={'abstention':'malformed'}
    result=episode_alarm(rec,plan,calls)
    assert result['operational']==pytest.approx(1/3)
    assert result['upper']==pytest.approx(2/3)


def test_reused_cost_is_not_incremental_and_uncertain_is_retained():
    calls={'a':{'reuse_provenance':{},'usage':{'cost':.2}},'b':{'usage':{'cost':.3}}}
    ledger={'calls':{'a':{'batch':'new_phase','state':'settled','charge':.3},
                     'b':{'batch':'new_phase','state':'uncertain_retained','charge':.05},
                     'c':{'batch':'old_phase','state':'settled','charge':.2}}}
    result=research_spend(calls,ledger,'new_')
    assert result['incremental_confirmed_credits']==.3
    assert result['incremental_uncertain_retained_credits']==.05
    assert result['reused_source_logical_replay_credits']==.2
    assert result['reused_unique_views']==result['new_unique_views']==1


def test_oracle_does_not_assume_missing_byte_metadata_fits():
    rec={'id':'x','graph':{'edges':[['a','b']]},'labels':{'false_origin_edges':[['a','b']]}}
    p={'episode_id':'x','feasible':True,'groups':[[['a','b']]],'view_keys':['v']}
    with pytest.raises(ValueError,match='byte metadata'):finite_oracles([rec],[p],[],{'v':{}})
    ids=edge_ids(rec);c={'planned_input_bytes':24001,'judgment':{'source_scores':{ids[('a','b')]:1.}}}
    assert all(o['reward']==0 for o in finite_oracles([rec],[p],[],{'v':c}))
    c['planned_input_bytes']=24000
    assert all(o['reward']==1 for o in finite_oracles([rec],[p],[],{'v':c}))


def test_incomplete_run_writes_no_analysis_artifacts(tmp_path):
    records=[{'id':str(i),'case_id':str(i)} for i in range(12)]
    (tmp_path/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    assignments=[{'feasible':True,'view_keys':['v']}]
    (tmp_path/'assignments.jsonl').write_text(json.dumps(assignments[0])+'\n')
    (tmp_path/'exact_random_plans.jsonl').write_text('');(tmp_path/'views.jsonl').write_text('')
    h=lambda x:hashlib.sha256(canonical(x).encode()).hexdigest()
    plan={'episodes':12,'case_clusters':12,'records_sha256':hashlib.sha256((tmp_path/'episodes.jsonl').read_bytes()).hexdigest(),
          'assignments':1,'expectations':0,'assignments_sha256':h(assignments),'expectations_sha256':h([]),'unique_views':1}
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='no global ranking'):analyze_complete(tmp_path)
    assert not (tmp_path/'analysis').exists()


def test_real_topology_assignments_replay_summary_and_nonlinear_alarm(tmp_path):
    from topology_workload_plan import plan_library
    from record_audit_workloads import workload
    from record_workload_benchmark import analyze,EVIDENCE
    from monitor_record_workflows import SYSTEM
    from analyze_topology_workload import enrich,summary_table
    rec=workload(0,4,condition='clean')
    jobs,plans,expectations=plan_library([rec],SYSTEM+EVIDENCE,seeds=(0,))
    calls={}
    for key,(r,group) in jobs.items():
        ids=edge_ids(r)
        calls[key]={'view_key':key,'judgment':{'source_scores':{ids[tuple(e)]:0. for e in group},
            'source_evidence_valid':True,'source_evidence':{}},
            'usage':{'input_tokens':10,'output_tokens':2,'cost':.001},'request_count':1}
    metrics=analyze([rec],plans,calls,tmp_path,expectations)
    enrich(metrics,[rec],plans,expectations,calls)
    summary=summary_table(metrics)
    assert len(metrics)==len(plans)+len(expectations)
    for row in summary:
        if row['llm_performance_defined']:
            assert row['mean_any_benign_flag_probability']=={'operational':0.,'lower':0.,'upper':0.}
            assert row['network_benign_fpr']['operational']==0
            assert row['flagged_view_citation_validity_rate'] is None
            assert row['mean_calls']>0
    exact=next(r for r in metrics if r['grouper']=='exact_matched_random')
    assert exact['expectation_not_a_single_execution']
    assert exact['comparison_scope']=='prospective_topology_only_exact_expectation'


def test_logged_jobs_restores_hashable_edges_for_frozen_prompt_validator():
    from analyze_topology_workload import logged_jobs
    records=[{'id':'x'}]
    assignments=[{'episode_id':'x','feasible':True,'groups':[[['a','b']]],'view_keys':['v']}]
    jobs=logged_jobs(records,assignments,[])
    assert jobs['v'][1]==[('a','b')]
    assert set(jobs['v'][1])=={('a','b')}


def test_primary_recall_keeps_unacquired_sources_and_conditional_result_separate():
    values=topology_rows()
    for row in values:
        if row['case_id']!='c2':continue
        row['reference_pair_available']=False;row['reference_pair_coobserved']=0
        if row['condition']=='false_report':
            row['selected_truth']=0;row['inspected_negative']=7
            for key in ('source_tp','source_truth_abstained','rule_grouped_tp','rule_joined_tp',
                        'message_supported_tp','strict_relational_supported_tp'):row[key]=0
    result=paired_contrasts(values,{'c1','c2'})
    # One case has the original .5 recovery improvement; the other source was
    # never acquired. It still contributes zero to the population recall delta.
    assert result['aligned']['deltas']['recall']['mean']==pytest.approx(.25)
    conditional=result['aligned']['conditional_pair_available_recall_delta']
    assert conditional['mean']==pytest.approx(.5) and conditional['cases']==1
    assert result['estimands']['primary_route']=='aligned'
    assert result['estimands']['routing_sensitivity']=='shuffled'


def test_descriptive_candidate_does_not_imply_dominance_over_abstention_bounds():
    result=paired_contrasts(topology_rows(),{'c1','c2'})
    assert result['aligned']['descriptive_improvement_candidate'] is True
    # The predeclared rule compares each arm's pessimistic utility. It is not a
    # claim that every possible resolution of both arms preserves that ordering.
    assert result['aligned']['deltas']['equal_class_utility']['mean_lower']<0
