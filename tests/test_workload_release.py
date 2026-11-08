from pathlib import Path
import copy
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pytest
from build_workload_release import split_record,remap,build,opaque
from verify_workload_release import join_observations,prompt,validate_dataset,expected_weights,canonical,RETRY
from record_audit_workloads import workload
from record_workload_benchmark import analyze,EVIDENCE
from record_workload_exact_random import plan_expectation
from monitor_record_workflows import SYSTEM,view_prompt,edge_ids


def fixture_data(tmp_path):
    rec=workload(0,4);system=SYSTEM+EVIDENCE;expected=plan_expectation(rec,system,0)
    selected=expected['selected'];groups=[selected[:2],selected[2:4],selected[4:6],selected[6:]]
    a={'episode_id':rec['id'],'policy':'test/provenance/k7/s2/seed0','selector':'test','grouper':'provenance','k':7,'s':2,'seed':0,
       'selected':selected,'groups':groups,'view_keys':[],'group_input_bytes':[],'input_byte_allowance':24000,'feasible':True}
    jobs={}
    for g in groups:
        q=view_prompt(rec,list(map(tuple,g)));key=hashlib.sha256(canonical([system,q]).encode()).hexdigest()
        jobs[key]=g;a['view_keys'].append(key);a['group_input_bytes'].append(len((system+RETRY+q).encode()))
    for v in expected['weighted_views']:jobs[v['view_key']]=v['edges']
    calls={};ids=edge_ids(rec);truth=set(map(tuple,rec['labels']['false_origin_edges']))
    for key,g in jobs.items():
        es=list(map(tuple,g));q=view_prompt(rec,es);flags=set(es)&truth
        evidence={ids[e]:list(rec['labels']['false_origin_messages']) for e in flags}
        judgment={'source_scores':{ids[e]:float(e in flags) for e in es},'source_evidence_valid':True,'source_evidence':evidence}
        attempt={'episode_id':rec['id'],'edges':g,'system':system,'prompt':q,'raw':json.dumps(judgment),
                 'usage':{'input_tokens':10,'output_tokens':3,'cost':.001},'attempt':0,'model':'mock','temperature':0,
                 'max_output_tokens':1024,'planned_input_bytes':len((system+RETRY+q).encode()),'prompt_sha256':key}
        calls[key]={**attempt,'view_key':key,'judgment':judgment,'attempts':[copy.deepcopy(attempt)],'request_count':1}
    metrics=analyze([rec],[a],calls,tmp_path,[expected])
    plan={'episodes':1,'assignments':1,'expected_random_assignments':1,'unique_views':len(calls),'system':system,'model':'mock','max_output_tokens':1024}
    return rec,a,expected,calls,metrics,plan


def test_public_projection_isolates_labels_but_reproduces_exact_prompt():
    rec=workload(0,4);public,labels=split_record('workload',rec);other=copy.deepcopy(rec)
    other['labels']={'secret':'poison'};other['condition']='poison';other['case_id']='poison';other['resolution']='poison';other['history']='poison'
    for m in other['messages']:m['hidden']={'secret':'poison'}
    assert split_record('workload',other)[0]==public
    assert set(public)=={'id','workflow','graph','messages'}
    reconstructed=join_observations([public],[labels])[0]
    assert reconstructed['id']==opaque('workload',rec['id'])
    assert prompt(public,rec['graph']['edges'])==view_prompt(rec,list(map(tuple,rec['graph']['edges'])))
    assert reconstructed['labels']==rec['labels']
    public['messages'][0]['hidden']={'secret':'poison'}
    with pytest.raises(ValueError,match='private'):join_observations([public],[labels])


def test_exact_expectation_and_partition_metrics_replay(tmp_path):
    rec,a,expected,calls,metrics,plan=fixture_data(tmp_path)
    report=validate_dataset('workload',[rec],[a],[expected],list(calls.values()),metrics,plan)
    assert report['expectation_assignments']==1 and report['assignments']==1
    assert expected['expected_calls']==4 and len(expected['weighted_views'])==28
    assert expected['randomization_partitions']==105
    assert metrics[-1]['logical_calls']==4 and metrics[-1]['requests']==pytest.approx(4.)
    ids={rec['id']:opaque('workload',rec['id'])};public,labels=split_record('workload',rec)
    assert validate_dataset('workload',join_observations([public],[labels]),remap([a],ids),remap([expected],ids),
        remap(list(calls.values()),ids),remap(metrics,ids),plan)==report


def test_missing_duplicate_and_extra_views_rejected(tmp_path):
    rec,a,expected,calls,metrics,plan=fixture_data(tmp_path);values=list(calls.values())
    with pytest.raises(ValueError,match='incomplete'):validate_dataset('workload',[rec],[a],[expected],values[:-1],metrics,plan)
    with pytest.raises(ValueError,match='duplicate'):validate_dataset('workload',[rec],[a],[expected],values+[values[0]],metrics,plan)
    extra=copy.deepcopy(values[0]);extra['view_key']='extra';larger=dict(plan,unique_views=len(values)+1)
    with pytest.raises(ValueError):validate_dataset('workload',[rec],[a],[expected],values+[extra],metrics,larger)


def test_wrong_prompt_metric_budget_or_weight_rejected(tmp_path):
    rec,a,expected,calls,metrics,plan=fixture_data(tmp_path);values=list(calls.values())
    altered=copy.deepcopy(values);altered[0]['prompt']+=' hidden oracle'
    with pytest.raises(ValueError,match='public prompt'):validate_dataset('workload',[rec],[a],[expected],altered,metrics,plan)
    wrong=copy.deepcopy(metrics);wrong[-1]['source_tp']+=.1
    with pytest.raises(ValueError,match='metric mismatch'):validate_dataset('workload',[rec],[a],[expected],values,wrong,plan)
    wrong=copy.deepcopy(a);wrong['k']=6
    with pytest.raises(ValueError,match='partition/acquisition'):validate_dataset('workload',[rec],[wrong],[expected],values,metrics,plan)
    wrong=copy.deepcopy(expected);wrong['weighted_views'][0]['weight_numerator']=2
    with pytest.raises(ValueError,match='expectation weight'):expected_weights(wrong)


def test_infeasible_assignment_is_retained_without_fabricated_llm_counts(tmp_path):
    rec=workload(0,16);system=SYSTEM+EVIDENCE;g=rec['graph']['edges'];q=view_prompt(rec,list(map(tuple,g)))
    key=hashlib.sha256(canonical([system,q]).encode()).hexdigest()
    a={'episode_id':rec['id'],'policy':'full','selector':'full','grouper':'full','k':13,'s':13,'seed':0,
       'selected':g,'groups':[g],'view_keys':[key],'group_input_bytes':[len((system+RETRY+q).encode())],
       'input_byte_allowance':24000,'feasible':False}
    metrics=analyze([rec],[a],{},tmp_path)
    plan={'episodes':1,'assignments':1,'expected_random_assignments':0,'unique_views':0,'system':system,'model':'mock','max_output_tokens':1024}
    assert validate_dataset('workload',[rec],[a],[],[],metrics,plan)['metrics']==1
    metrics[0]['source_tp']=0
    with pytest.raises(ValueError,match='fabricated'):validate_dataset('workload',[rec],[a],[],[],metrics,plan)


def test_incomplete_build_creates_no_release_directory(tmp_path):
    diagnostic=tmp_path/'diagnostic';workload_root=tmp_path/'workload';out=tmp_path/'release'
    diagnostic.mkdir();workload_root.mkdir()
    with pytest.raises(ValueError,match='missing completed artifact'):build(diagnostic,workload_root,out,tmp_path)
    assert not out.exists()


def test_complete_small_candidate_builds_and_manifest_tampering_fails(tmp_path):
    from build_workload_release import REQUIRED,ROWS
    from verify_workload_release import verify,core_counts
    rec,a,expected,calls,metrics,plan=fixture_data(tmp_path)
    diagnostic=tmp_path/'diagnostic';workload_root=tmp_path/'workload';project=tmp_path/'project'
    (project/'src/jmon').mkdir(parents=True);(project/'src/jmon/__init__.py').write_text('')
    (project/'pyproject.toml').write_text('[project]\nname="test"\nversion="0"\n')
    one=next(iter(calls.values()));dassign={'id':rec['id'],'view':'mock','prompt_arm':'original','view_key':one['view_key'],'edges':one['edges']}
    counts=core_counts(rec,one['edges'],one);truth=set(map(tuple,rec['labels']['false_origin_edges']))
    dmetric=dict(dassign,**{k:counts[k] for k in ('source_tp','source_fp','n_flags','tokens')},
                 n_truth=len(truth),inspected_negative=len(set(map(tuple,one['edges']))-truth),abstention=False)
    dplan={'episodes':1,'assignments':1,'unique_views':1,'model':'mock','max_output_tokens':1024,
           'prompt_hashes':{'original':hashlib.sha256(plan['system'].encode()).hexdigest()},
           'records_sha256':hashlib.sha256(canonical([rec]).encode()).hexdigest(),'source_hashes':{}}
    plan.update(source_hashes={},assignments_sha256=hashlib.sha256(canonical([a]).encode()).hexdigest(),
                expected_random_sha256=hashlib.sha256(canonical([expected]).encode()).hexdigest(),
                view_keys_sha256=hashlib.sha256(canonical(sorted(calls)).encode()).hexdigest())
    for kind,root,pl,assign,expect,cs,ms in [('diagnostic',diagnostic,dplan,[dassign],[],[one],[dmetric]),
                                         ('workload',workload_root,plan,[a],[expected],list(calls.values()),metrics)]:
        root.mkdir();(root/'source').mkdir()
        for name in REQUIRED[kind]:(root/name).write_text('{}')
        for name in ROWS[kind]:(root/name).write_text('')
        (root/'plan.json').write_text(json.dumps(pl))
        rows={'episodes.jsonl':[rec],'assignments.jsonl':assign,'metrics.jsonl':ms,'views.jsonl':cs,
              'judge_calls.jsonl':[x for c in cs for x in c['attempts']]}
        if kind=='workload':
            rows['exact_random_plans.jsonl']=expect
            (root/'research_spend.json').write_text('{"active_reservations":0}')
        for name,rs in rows.items():(root/name).write_text(''.join(json.dumps(r)+'\n' for r in rs))
    out=tmp_path/'release';report=build(diagnostic,workload_root,out,project)
    assert report['workload']['views']==len(calls)
    assert verify(out)==report
    (out/'README.md').write_text('changed')
    with pytest.raises(ValueError,match='file hash mismatch'):verify(out)


@pytest.mark.parametrize('field,value',[('planned_input_bytes',None),('planned_input_bytes',0),
    ('planned_input_bytes','4037'),('planned_input_bytes',True),('max_output_tokens',None),
    ('max_output_tokens',512),('max_output_tokens','1024')])
def test_workload_call_capacity_metadata_is_required_and_verified(tmp_path,field,value):
    rec,a,expected,calls,metrics,plan=fixture_data(tmp_path);values=copy.deepcopy(list(calls.values()))
    if value is None:values[0].pop(field)
    else:values[0][field]=value
    with pytest.raises(ValueError,match='workload.*(bytes|allowance)'):
        validate_dataset('workload',[rec],[a],[expected],values,metrics,plan)
