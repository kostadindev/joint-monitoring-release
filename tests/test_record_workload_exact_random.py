from collections import Counter
import copy
from fractions import Fraction
import hashlib
from itertools import combinations
import math
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from record_workload_exact_random import weighted_subsets,plan_expectation,plan_expected_library,evaluate_expected
from record_audit_workloads import workload
from record_exact_grouping import partitions
from record_strategies import select,group,policy_id
from record_workload_benchmark import EVIDENCE,analyze
from monitor_record_workflows import SYSTEM,FORMAT_RETRY,view_prompt,edge_ids
from record_workflow import canonical


def fake_calls(rec,plan):
    ids=edge_ids(rec);truth=set(map(tuple,rec['labels']['false_origin_edges']));calls={}
    origin=set(rec['labels']['false_origin_messages'])
    for i,v in enumerate(plan['weighted_views']):
        edges=list(map(tuple,v['edges']));key=v['view_key']
        call={'view_key':key,'system':SYSTEM+EVIDENCE,'prompt':view_prompt(rec,edges),'usage':{'input_tokens':101+i*7,'output_tokens':11+i,'cost':.001*(i+1)},'request_count':2 if i%4==0 else 1}
        if i%7==0:call['abstention']={'reason':'controlled schema failure'}
        else:
            scores={ids[e]:.9 if (int(ids[e][1:])+i)%3==0 else .2 for e in edges}
            evidence_ok=i%5!=0;evidence={}
            for e in edges:
                if scores[ids[e]]<=.5:continue
                candidates=[m['id'] for m in rec['messages'] if (m['sender'],m['recipient'])==e]
                if e in truth and i%2==0:candidates=[m for m in candidates if m in origin]+candidates
                evidence[ids[e]]=[candidates[0]]
            call['judgment']={'source_scores':scores,'source_evidence_valid':evidence_ok,'source_evidence':evidence}
        calls[key]=call
    return calls


def test_subset_probabilities_equal_uniform_partition_marginals():
    selected=[('u',str(i)) for i in range(7)]
    for shape in ([2,2,2,1],[2,2,1,1,1],[1]*7):
        possibilities=partitions(selected,shape);counts=Counter(tuple(sorted(g)) for p in possibilities for g in p)
        expected=dict(weighted_subsets(selected,shape))
        assert expected=={g:Fraction(n,len(possibilities)) for g,n in counts.items()}
        assert sum(expected.values())==len(shape)
        assert sum(len(g)*w for g,w in expected.items())==7
        for e in selected:assert sum(w for g,w in expected.items() if e in g)==1


def test_plan_is_public_only_budget_valid_and_hash_identical_to_runner():
    rec=workload(0,4);other=copy.deepcopy(rec);other.pop('labels');other.pop('condition')
    for m in other['messages']:m.pop('hidden')
    system=SYSTEM+EVIDENCE
    for seed in (0,1,2):
        plan=plan_expectation(rec,system,seed)
        assert plan==plan_expectation(other,system,seed)
        selected=select({'graph':rec['graph']},7,'provenance_bundles',seed)
        assert list(map(tuple,plan['selected']))==selected
        assert plan['group_sizes']==sorted(map(len,group({'graph':rec['graph']},selected,2,'provenance',seed)))
        assert plan['expected_inspected_edges']==7 and plan['expected_calls']==len(plan['group_sizes'])
        assert all(x['probability']==1. for x in plan['edge_inclusion_probabilities'])
        for v in plan['weighted_views']:
            prompt=view_prompt(rec,list(map(tuple,v['edges'])))
            assert v['view_key']==hashlib.sha256(canonical([system,prompt]).encode()).hexdigest()
            assert v['input_bytes']==len((system+FORMAT_RETRY+prompt).encode())<=24000
        assert plan['expectation_not_a_single_execution'] and 'groups' not in plan
    jobs,plans=plan_expected_library([rec],system)
    assert len(plans)==3 and set(jobs)=={v['view_key'] for p in plans for v in p['weighted_views']}


@pytest.mark.parametrize('case_index',[0,1,3])
def test_expected_counts_match_complete_partition_replay(tmp_path,case_index):
    rec=workload(case_index,4);plan=plan_expectation(rec,SYSTEM+EVIDENCE,0);calls=fake_calls(rec,plan)
    expected=evaluate_expected(rec,plan,calls)
    selected=list(map(tuple,plan['selected']));parts=partitions(selected,plan['group_sizes'])
    keys={tuple(sorted(map(tuple,v['edges']))):v['view_key'] for v in plan['weighted_views']}
    assignments=[]
    for i,groups in enumerate(parts):
        a={'episode_id':rec['id'],'selector':'provenance_bundles','grouper':'enumeration_reference','k':7,'s':2,'seed':i,
           'selected':selected,'groups':groups,'input_byte_allowance':24000,'feasible':True,
           'view_keys':[keys[tuple(sorted(g))] for g in groups]}
        a['policy']=policy_id(a);assignments.append(a)
    rows=analyze([rec],assignments,calls,tmp_path)
    fields=('n_truth','selected_truth','rule_grouped_tp','rule_grouped_fp','rule_joined_tp','rule_joined_fp',
            'reference_pair_available','reference_pair_coobserved','source_tp','source_fp','n_flags','inspected_negative',
            'report_fp','report_negatives','message_supported_tp','source_truth_abstained','negative_edges_abstained',
            'report_negatives_abstained','abstained_edges','tokens','replay_credits','requests','logical_calls',
            'evidence_invalid_views','evidence_valid_flags')
    for name in fields:
        assert expected[name]==pytest.approx(sum(r[name] for r in rows)/len(rows)),name
    assert expected['logical_calls']<expected['candidate_library_views']
    assert expected['randomization_partitions']==len(parts)
    assert expected['evaluation_status']=='scored' and expected['missing_views']==0
    if case_index==1:assert expected['source_truth_abstained']>0
    if case_index==3:assert expected['source_tp']>0 and expected['message_supported_tp']>0


def test_missing_or_invalid_views_cannot_become_zero_scores():
    rec=workload(0,4);plan=plan_expectation(rec,SYSTEM+EVIDENCE,0);calls=fake_calls(rec,plan)
    missing=dict(calls);missing.pop(next(iter(missing)))
    row=evaluate_expected(rec,plan,missing)
    assert row['evaluation_status']=='missing' and row['missing_views']==1
    assert 'source_tp' not in row and 'tokens' not in row and 'rule_grouped_tp' in row
    corrupted=copy.deepcopy(calls);key=next(iter(corrupted));corrupted[key]['usage']['cost']=None
    with pytest.raises(ValueError,match='usage'):evaluate_expected(rec,plan,corrupted)
    corrupted=copy.deepcopy(calls);corrupted[key]['prompt']='wrong public view'
    with pytest.raises(ValueError,match='prompt mismatch'):evaluate_expected(rec,plan,corrupted)
    corrupted=copy.deepcopy(calls);corrupted[key]['system']='another system prompt'
    with pytest.raises(ValueError,match='system mismatch'):evaluate_expected(rec,plan,corrupted)
    with pytest.raises(ValueError,match='byte allowance'):plan_expectation(rec,'x'*24001,0)


def test_mutated_weight_plan_is_rejected():
    rec=workload(0,4);plan=plan_expectation(rec,SYSTEM+EVIDENCE,0);calls=fake_calls(rec,plan)
    broken=copy.deepcopy(plan);broken['weighted_views'].pop()
    with pytest.raises(ValueError,match='weights'):evaluate_expected(rec,broken,calls)
    broken=copy.deepcopy(plan);broken['weighted_views'][0]['weight']=0
    with pytest.raises(ValueError,match='weight disagreement'):evaluate_expected(rec,broken,calls)
