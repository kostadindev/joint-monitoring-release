"""Offline release contract tests; generated judge text is explicitly synthetic."""
from pathlib import Path
import copy
import fcntl
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pytest
from build_topology_release import build,compressed_bytes,collect_reused
from verify_topology_release import (strict_allocator_input,anonymous_graph,allocator_catalog,
    verify_allocator_catalog,split_observation,join_judge_data,acquisition_prompts,
    verify_reuse,verify_attempt_journal,verify_model,allowed_path,FEATURES)
from verify_workload_release import canonical,digest,prompt,validate_dataset,RETRY,expected_weights
from record_audit_workloads import workload
from topology_workload_plan import plan_library
from record_workload_benchmark import analyze,EVIDENCE,parse_workload
from monitor_record_workflows import SYSTEM,edge_ids
from run_topology_workload import validate_call,MODEL


def simple_assignment(rec):
    return {'episode_id':rec['id'],'k':7,'s':2,'information_access':'anonymous directed topology only',
            'comparison_scope':'prospective_topology_only'}


def test_allocator_projection_is_insensitive_to_non_graph_information():
    rec=workload(0,4);a=simple_assignment(rec);public,mapping=allocator_catalog([rec],[a])
    other=copy.deepcopy(rec)
    for key in set(other)-{'id','graph'}:other[key]={'secret':'poison'}
    other['graph']['roles']={'secret':'role poison'};other['graph']['sink']='poison'
    assert allocator_catalog([other],[a])== (public,mapping)
    assert set(public[0])=={'graph','k','s'} and set(public[0]['graph'])=={'nodes','edges'}
    assert verify_allocator_catalog([rec],[a],public,mapping)==1
    altered=copy.deepcopy(mapping);altered[0]['anonymous_to_named'][0][1]='secret'
    with pytest.raises(ValueError,match='mapping'):verify_allocator_catalog([rec],[a],public,altered)


@pytest.mark.parametrize('where,key,value',[
    ('root','messages',[]),('root','episode_id','secret'),('root','length',3),
    ('graph','roles',{}),('graph','trusted',True),('graph','targets',[]),
    ('graph','nodes',['registry',1]),('graph','nodes',[False,1]),
    ('graph','edges',[[0,1,{'weight':1}]]),('root','k',True)])
def test_allocator_rejects_every_extra_information_path(where,key,value):
    public={'graph':{'nodes':[0,1],'edges':[[0,1]]},'k':1,'s':1}
    (public if where=='root' else public['graph'])[key]=value
    with pytest.raises(ValueError):strict_allocator_input(public)


def test_private_labels_roundtrip_and_exact_selected_prompts():
    rec=workload(0,4);obs,labels=split_observation(rec)
    other=copy.deepcopy(rec);other['labels']={'secret':True};other['case_id']='secret';other['condition']='secret'
    for m in other['messages']:m['hidden']={'secret':True}
    assert split_observation(other)[0]==obs
    assert join_judge_data([obs],[labels])==[rec]
    graph,restore=anonymous_graph(rec['graph']);selected=graph['edges'][:2]
    _,mappings=allocator_catalog([rec],[simple_assignment(rec)])
    result=acquisition_prompts({rec['id']:rec},mappings[0],2,2,selected,[selected])
    assert result==[prompt(rec,[(restore[u],restore[v]) for u,v in selected])]
    acquired=set(map(tuple,selected));named={(restore[u],restore[v]) for u,v in acquired}
    shown=json.loads(result[0])['messages']
    assert {m['id'] for m in shown}=={m['id'] for m in rec['messages'] if (m['sender'],m['recipient']) in named}
    obs['messages'][0]['hidden']={'secret':True}
    with pytest.raises(ValueError,match='leaked'):join_judge_data([obs],[labels])


def test_acquisition_enforces_budget_partition_and_mapping():
    rec=workload(0,4);graph,_=anonymous_graph(rec['graph']);selected=graph['edges'][:2]
    _,maps=allocator_catalog([rec],[simple_assignment(rec)]);mapping=maps[0];records={rec['id']:rec}
    for k,s,groups in [(1,2,[selected]),(2,1,[selected]),(2,2,[selected,selected[:1]]),(2,2,[selected[:1]])]:
        with pytest.raises(ValueError):acquisition_prompts(records,mapping,k,s,selected,groups)
    bad=copy.deepcopy(mapping);bad['anonymous_to_named'][0][1],bad['anonymous_to_named'][1][1]=bad['anonymous_to_named'][1][1],bad['anonymous_to_named'][0][1]
    with pytest.raises(ValueError,match='mapping'):acquisition_prompts(records,bad,2,2,selected,[selected])


def model_fixture():
    model={'features':list(FEATURES),'coef':[0.]*len(FEATURES),'alpha':.01,
           'supervision':'numerical development rewards; topology-only deployment'}
    return {'model':model,'training_cases':['train'+str(i) for i in range(5)],'training_episodes':51,
        'source_sha256':{'trainer.py':'hash'},'input_sha256':{'records':'hash'},
        'workload_identity_check':{'overlap':[]},'reward_definition':{'false_positive_penalty':2,'strict_threshold':True}}


def test_runtime_model_is_topology_only_and_training_identity_disjoint():
    rec=workload(0,4);artifact=model_fixture()
    assert verify_model(artifact,artifact['model'],[rec])==5
    runtime=copy.deepcopy(artifact['model']);runtime['training_cases']=artifact['training_cases']
    with pytest.raises(ValueError,match='schema'):verify_model(artifact,runtime,[rec])
    artifact['model']['features'][0]='sender_role'
    with pytest.raises(ValueError,match='schema'):verify_model(artifact,artifact['model'],[rec])
    artifact=model_fixture();artifact['training_cases'][0]=rec['labels']['case_ids'][-1]
    with pytest.raises(ValueError,match='overlap'):verify_model(artifact,artifact['model'],[rec])


def test_reused_line_provenance_preserves_original_and_never_recharges(tmp_path):
    original={'view_key':'abc','raw':'recorded response','usage':{'cost':.07}}
    raw=json.dumps(original)+'\n';source=tmp_path/'old_views.jsonl';source.write_text(raw)
    provenance={'path':str(source),'file_sha256':digest(source.read_bytes()),'line':1,'record_sha256':digest(raw.encode())}
    call=dict(original,reuse_provenance=provenance);plan={'reuse_sources':[{'path':str(source),'sha256':provenance['file_sha256']}]}
    rows=collect_reused([call],plan)
    assert rows[0]['incremental_credits']==0 and verify_reuse([call],rows,plan)==1
    bad=copy.deepcopy(rows);bad[0]['incremental_credits']=.07
    with pytest.raises(ValueError,match='cost'):verify_reuse([call],bad,plan)
    bad=copy.deepcopy(call);bad['raw']='changed judgment'
    with pytest.raises(ValueError,match='altered'):verify_reuse([bad],rows,plan)
    bad=copy.deepcopy(rows);bad[0]['raw_record_line']=raw+' '
    with pytest.raises(ValueError,match='hash'):verify_reuse([call],bad,plan)
    with pytest.raises(ValueError,match='coverage'):verify_reuse([call],[],plan)


def synthetic_topology_fixture(tmp_path):
    rec=workload(0,4,condition='false_report');system=SYSTEM+EVIDENCE
    jobs,assignments,expectations=plan_library([rec],system,seeds=(0,));calls={}
    for key,(r,edges) in jobs.items():
        ids=edge_ids(r);zero={ids[tuple(e)]:0 for e in edges}
        raw=canonical({'source_scores':zero,'content_scores':zero,'incident_score':0,
                       'uncertain_source':False,'evidence':[],'reason':'synthetic neutral test response','source_evidence':{}})
        q=prompt(r,edges)
        attempt={'episode_id':r['id'],'edges':[list(e) for e in edges],'system':system,'prompt':q,'raw':raw,
            'usage':{'input_tokens':10,'output_tokens':5,'cost':.001},'attempt':0,'model':MODEL,'temperature':0,
            'max_output_tokens':1024,'planned_input_bytes':len((system+RETRY+q).encode()),'prompt_sha256':key}
        call={**attempt,'view_key':key,'judgment':parse_workload(raw,r,edges),'attempts':[copy.deepcopy(attempt)],'request_count':1}
        validate_call(call,key,r,edges,system);calls[key]=call
    metrics=analyze([rec],assignments,calls,tmp_path,expectations)
    plan={'episodes':1,'assignments':len(assignments),'expected_random_assignments':len(expectations),
          'unique_views':len(calls),'system':system,'model':MODEL,'max_output_tokens':1024}
    return rec,assignments,expectations,calls,metrics,plan


def test_real_topology_plans_exact_expectations_and_call_replay(tmp_path):
    rec,assignments,expectations,calls,metrics,plan=synthetic_topology_fixture(tmp_path)
    result=validate_dataset('workload',[rec],assignments,expectations,list(calls.values()),metrics,plan)
    assert result['assignments']==len(assignments)
    assert len(expectations)==1 and expectations[0]['selector']=='structural_path_pairs'
    expected_weights(expectations[0])
    primary=next(a for a in assignments if a['selector']=='structural_path_pairs' and a['grouper']=='directed_path' and a['k']==7 and a['s']==2)
    assert canonical(primary['selected'])==canonical(expectations[0]['selected'])
    assert sorted(map(len,primary['groups']))==sorted(expectations[0]['group_sizes'])
    assert expectations[0]['expected_calls']==len(primary['groups'])
    assert sum(v['weight'] for v in expectations[0]['weighted_views'])==pytest.approx(len(primary['groups']))
    bad=copy.deepcopy(expectations[0]);bad['weighted_views'][0]['weight_numerator']+=1
    with pytest.raises(ValueError,match='weight'):expected_weights(bad)
    with pytest.raises(ValueError,match='incomplete'):validate_dataset('workload',[rec],assignments,expectations,list(calls.values())[:-1],metrics,plan)
    bad=copy.deepcopy(metrics);bad[0]['source_fp']=123
    with pytest.raises(ValueError,match='metric'):validate_dataset('workload',[rec],assignments,expectations,list(calls.values()),bad,plan)


def test_missing_wrong_bytes_output_budget_or_raw_judgment_rejected(tmp_path):
    rec,assignments,expectations,calls,metrics,plan=synthetic_topology_fixture(tmp_path)
    key=next(iter(calls));call=calls[key]
    for field,value in [('planned_input_bytes',None),('planned_input_bytes',0),('max_output_tokens',None),('max_output_tokens',512)]:
        bad=copy.deepcopy(call)
        if value is None:del bad[field]
        else:bad[field]=value
        changed=list(calls.values());changed[0]=bad
        with pytest.raises(ValueError):validate_dataset('workload',[rec],assignments,expectations,changed,metrics,plan)
    bad=copy.deepcopy(call);bad['judgment']['source_scores'][next(iter(bad['judgment']['source_scores']))]=1.
    with pytest.raises(ValueError,match='altered'):validate_call(bad,key,rec,list(map(tuple,call['edges'])),plan['system'])
    bad=copy.deepcopy(call);bad['attempts'][0]['raw']='invalid JSON'
    with pytest.raises(ValueError,match='bounded abstention'):validate_call(bad,key,rec,list(map(tuple,call['edges'])),plan['system'])


def test_incomplete_or_active_run_creates_no_candidate(tmp_path):
    run=tmp_path/'run';run.mkdir();out=tmp_path/'release'
    with pytest.raises(ValueError,match='missing run artifact'):build(run,out,tmp_path)
    assert not out.exists()
    with (run/'.run.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(ValueError,match='still running'):build(run,out,tmp_path)
    assert not out.exists()


def test_explicit_file_allowlist_excludes_secrets_and_external_raw_data():
    for p in ('source/src/jmon/llm/adaption.py','analysis/source/scripts/analyze_topology_workload.py',
              'evaluation/labels.jsonl.gz','allocator_inputs.jsonl.gz','audit/phase_pilot.json'):
        assert allowed_path(p)
    for p in ('.env','source/.env','source/src/jmon/.env.py','cache/views.jsonl','MESA/raw.jsonl',
              'audit/raw_incident.json','../secret.py','/etc/passwd','source/scripts/../../secret.py'):
        assert not allowed_path(p)


def test_reproducible_gzip_preserves_exact_raw_jsonl_bytes(tmp_path):
    raw=b'{"value": 3, "other": "unicode \\u00e9"}\n';a=tmp_path/'a.gz';b=tmp_path/'b.gz'
    compressed_bytes(a,raw);compressed_bytes(b,raw)
    assert a.read_bytes()==b.read_bytes()
    import gzip
    assert gzip.decompress(a.read_bytes())==raw


def test_journal_keeps_unfinalized_attempts_but_requires_completed_attempts(tmp_path):
    rec,assignments,expectations,calls,metrics,plan=synthetic_topology_fixture(tmp_path)
    journal=[copy.deepcopy(a) for c in calls.values() for a in c['attempts']]
    result=verify_attempt_journal(list(calls.values()),journal)
    assert result['additional_historical_attempts']==0
    old=copy.deepcopy(journal[0]);old['raw']='historical malformed answer';old['latency_s']=9.0
    journal.append(old)
    assert verify_attempt_journal(list(calls.values()),journal)['additional_historical_attempts']==1
    with pytest.raises(ValueError,match='missing'):verify_attempt_journal(list(calls.values()),journal[1:])
    bad=copy.deepcopy(journal);bad[-1]['prompt']+=' unplanned'
    with pytest.raises(ValueError,match='unplanned'):verify_attempt_journal(list(calls.values()),bad)
