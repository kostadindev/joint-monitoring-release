from pathlib import Path
import sys,json,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pytest
from record_audit_workloads import workload
from record_workload_benchmark import policies,plan_library,parse_workload,EVIDENCE,analyze
from monitor_record_workflows import SYSTEM,edge_ids


def test_prospective_allocations_ignore_labels_and_oracles_are_explicit():
    rec=workload(0,4);other=copy.deepcopy(rec)
    other.pop('labels')
    a=list(policies(rec,include_oracles=False));b=list(policies(other,include_oracles=False))
    assert a==b
    assert all(p['selector'] not in ('target_role_oracle','reference_location_oracle') for p in a)


def test_plan_omits_infeasible_dispatch_but_keeps_assignment():
    rec=workload(0,16);jobs,assignments=plan_library([rec],SYSTEM+EVIDENCE)
    full=next(a for a in assignments if a['k']==13 and a['input_byte_allowance']==24000)
    assert not full['feasible']
    relaxed=next(a for a in assignments if a['k']==13 and a['input_byte_allowance']==60000)
    assert relaxed['feasible'] and relaxed['view_keys'][0] in jobs
    assert full['view_keys']==relaxed['view_keys']
    assert all(a['feasible'] for a in assignments if a['s']<=4)
    assert len({(a['episode_id'],a['policy']) for a in assignments})==len(assignments)


def test_exact_source_message_evidence_and_wrong_channel_rejected():
    rec=workload(0,4);edge=tuple(rec['labels']['false_origin_edges'][0]);cid=edge_ids(rec)[edge]
    mids=rec['labels']['false_origin_messages']
    d={'incident_score':100,'source_scores':{cid:100},'content_scores':{cid:100},
       'uncertain_source':False,'evidence':[cid],'reason':'Contradiction','source_evidence':{cid:mids}}
    assert parse_workload(json.dumps(d),rec,[edge])['source_evidence'][cid]==mids
    d['source_evidence'][cid]=['unshown']
    assert not parse_workload(json.dumps(d),rec,[edge])['source_evidence_valid']
    d.pop('source_evidence')
    assert not parse_workload(json.dumps(d),rec,[edge])['source_evidence_valid']


def test_missing_scores_not_converted_into_negative_judgments(tmp_path):
    rec=workload(0,16);_,assignments=plan_library([rec],SYSTEM+EVIDENCE)
    rows=analyze([rec],assignments,{},tmp_path)
    assert {r['evaluation_status'] for r in rows}=={'missing','infeasible'}
    assert all('source_tp' not in r for r in rows)
    assert all('rule_joined_tp' in r for r in rows)


def test_limited_oracle_uses_real_disjoint_budgeted_views():
    from analyze_record_workload import allocation_oracle
    rec=workload(0,4);ids=edge_ids(rec);truth=tuple(rec['labels']['false_origin_edges'][0])
    other=next(tuple(e) for e in rec['graph']['edges'] if tuple(e)!=truth)
    bad={'judgment':{'source_scores':{ids[truth]:1.,ids[other]:1.}}}
    good={'judgment':{'source_scores':{ids[truth]:1.,ids[other]:0.}}}
    assert allocation_oracle(rec,[('bad',[truth,other],bad)],7,2)['reward']==0
    best=allocation_oracle(rec,[('good',[truth,other],good),('bad',[truth,other],bad)],7,2)
    assert best['reward']==1 and best['view_keys']==['good'] and best['used_edges']==2
    assert allocation_oracle(rec,[('good',[truth,other],good)],1,2)['reward']==0


def test_summary_does_not_hide_infeasible_variant():
    from analyze_record_workload import summarize
    r={'workload_size':16,'environment':'aligned','condition':'false_report','selector':'x','grouper':'x','k':7,'s':7,
       'input_byte_allowance':24000,'case_id':'c','evaluation_status':'infeasible','n_truth':1,'selected_truth':1,'rule_grouped_tp':1,'rule_joined_tp':1,'rule_grouped_fp':0,'rule_joined_fp':0,
       'reference_pair_available':True,'reference_pair_coobserved':True}
    summary=summarize([r])[0]
    assert not summary['llm_performance_defined'] and 'source_recall' not in summary


def test_bad_auxiliary_evidence_preserves_primary_scores_without_resampling(tmp_path):
    import threading
    from record_workload_benchmark import inspect
    rec=workload(0,4);edge=tuple(rec['labels']['false_origin_edges'][0]);cid=edge_ids(rec)[edge]
    d={'incident_score':100,'source_scores':{cid:100},'content_scores':{cid:100},
       'uncertain_source':False,'evidence':[cid],'reason':'Contradiction'}
    class Provider:
        model='mock';temperature=0;max_output_tokens=1024;calls=0
        def complete(self,system,prompt):
            self.calls+=1
            out=dict(d)
            if self.calls==2:out['source_evidence']={cid:rec['labels']['false_origin_messages']}
            return json.dumps(out),{'input_tokens':100,'output_tokens':20,'cost':.001}
    p=Provider();out=inspect(rec,[edge],SYSTEM+EVIDENCE,p,tmp_path/'raw.jsonl',threading.Lock())
    assert p.calls==out['request_count']==1 and out['usage']['cost']==.001
    assert len(out['attempts'])==len((tmp_path/'raw.jsonl').read_text().splitlines())==1
    assert not out['judgment']['source_evidence_valid'] and out['judgment']['source_scores'][cid]==1.
    assert 'judgment' in out and 'abstention' not in out


def test_workload_intervention_changes_one_factual_message_and_no_public_identity():
    import re
    from record_audit_analysis import parse_claim
    for i in range(12):
        for history in ('stable','updated'):
            clean=workload(i,4,'clean',target_history=history)
            false=workload(i,4,'false_report',target_history=history)
            assert all(re.fullmatch('case_[a-f0-9]{12}',c) for c in clean['labels']['case_ids'])
            changed=[]
            for a,b in zip(clean['messages'],false['messages']):
                assert {k:v for k,v in a.items() if k not in ('content','hidden')}=={k:v for k,v in b.items() if k not in ('content','hidden')}
                if a['content']!=b['content'] and a['hidden']['field'] is not None:
                    left,right=parse_claim(a['content']),parse_claim(b['content'])
                    assert left[:-1]==right[:-1] and left[-1]!=right[-1]
                    changed.append(b['id'])
            assert changed==false['labels']['false_origin_messages']


def test_ranking_retains_infeasible_families_and_rule_results():
    from analyze_record_workload import development_ranking
    row={'workload_size':16,'environment':'aligned','k':7,'s':7,'input_byte_allowance':24000,
         'selector':'random','grouper':'all_selected','case_id':'c','condition':'false_report',
         'feasible':False,'evaluation_status':'infeasible','rule_grouped_tp':1,'rule_grouped_fp':0,'rule_joined_tp':1,'rule_joined_fp':0}
    result=development_ranking([row])[0]
    assert result['utility_operational'] is None and result['feasible_fraction']==0
    assert result['mean_rule_joined_utility']==1
