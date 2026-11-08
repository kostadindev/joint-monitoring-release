"""Fake-transport tests only; no model API calls or real experiment preparation."""
import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_topology_causal_monitor as runner
import run_topology_causal_workflows as generation
from topology_causal_workflows import task_manifest,run_record,digest
from topology_causal_monitor import plan_library
from monitor_record_workflows import view_prompt
from night_budget import NightBudget
from test_topology_causal_workflows import ScriptedProvider


def response(text,cost=.0001,tokens=20):
    return {'model':runner.CONFIG['model'],'output':[{'type':'message','content':[{'type':'output_text','text':text}]}],
            'usage':{'input_tokens':20,'output_tokens':tokens,'cost':cost}}


def judgment(prompt,aux=True):
    ids=json.loads(prompt)['channels']
    data={'source_scores':dict.fromkeys(ids,0),'content_scores':dict.fromkeys(ids,0),'incident_score':0,
          'uncertain_source':True,'evidence':[],'reason':'No supplied source evidence.'}
    if aux:data['source_evidence']={}
    return json.dumps(data)


def job():
    rec=run_record(task_manifest()[0],'flat','combined',ScriptedProvider())
    jobs,_,_=plan_library([rec],seeds=(0,));key=next(iter(jobs))
    return key,*jobs[key]


def provider(tmp_path,transport,phase='test'):
    ledger=NightBudget(tmp_path/'budget.json','phase_'+phase,.3)
    return runner.DurableJudgeProvider(tmp_path/'judge',ledger,transport,phase),ledger


@pytest.fixture(scope='module')
def completed_generation(tmp_path_factory):
    directory=tmp_path_factory.mktemp('synthetic_native');out=directory/'generation'
    plan,tasks,suites=generation.prepare(out,directory/'budget.json')
    scripted=ScriptedProvider()
    def transport(body):
        raw,_=scripted.complete(body['input'][0]['content'],body['input'][1]['content'])
        return response(raw)
    result=generation.run_phase(out,plan,tasks,suites,phase='fixture',phase_cap=.3,new_suite_limit=12,workers=2,transport_factory=lambda:transport)
    assert result['complete']
    return out


def test_prepare_complete_generation_freezes_sources_uniform_cap_and_all_worlds(tmp_path,completed_generation,monkeypatch):
    import jmon.llm
    monkeypatch.setattr(jmon.llm,'make_provider',lambda *a,**kw:pytest.fail('prepare constructed provider'))
    before=Path(json.loads((completed_generation/'plan.json').read_text())['ledger']).read_bytes()
    result=runner.prepare(completed_generation,tmp_path/'judge');plan,records,jobs,rows,expectations,_=result
    assert plan['episodes']==60 and plan['case_clusters']==4
    assert plan['judge_config']['max_output_tokens']==2048 and len(rows)==60*34
    assert len(expectations)==180 and all('unknown_source_edges' in r['labels'] for r in records)
    assert len([r for r in rows if r['comparison_scope']=='privileged_location_diagnostic'])==360
    assert {'scripts/situation_suite.py','scripts/record_allocation.py','scripts/record_pair_search.py',
            'scripts/record_policy_extensions.py','src/jmon/llm/base.py'}<=set(plan['source_sha256'])
    assert runner.prepare(completed_generation,tmp_path/'judge')[0]==plan
    assert Path(plan['ledger']).read_bytes()==before
    with pytest.raises(ValueError,match='complete all feasible'):
        runner.complete_analysis(tmp_path/'judge',records,rows,expectations,jobs,{})
    (tmp_path/'judge'/'source'/'scripts'/'topology_causal_monitor.py').write_text('changed')
    with pytest.raises(ValueError,match='frozen source'):
        runner.prepare(completed_generation,tmp_path/'judge')


def test_incomplete_native_generation_cannot_be_prepared(tmp_path):
    generation.prepare(tmp_path/'generation',tmp_path/'budget.json')
    with pytest.raises(ValueError,match='complete all frozen generation'):
        runner.prepare(tmp_path/'generation',tmp_path/'judge')
    assert not (tmp_path/'judge'/'plan.json').exists()


def test_valid_answer_with_missing_auxiliary_evidence_is_never_retried(tmp_path):
    key,rec,edges=job();requests=[]
    def transport(body):requests.append(body);return response(judgment(body['input'][1]['content'],aux=False))
    p,ledger=provider(tmp_path,transport)
    call=runner.inspect_view(p.out,key,rec,edges,p)
    assert len(requests)==call['request_count']==1
    assert call['judgment']['source_evidence_valid'] is False
    assert requests[0]['max_output_tokens']==2048
    assert runner.load_calls(p.out,{key:(rec,edges)})[key]['raw']==call['raw']
    # Reconstructing after a view-commit failure retains the first raw answer.
    second=runner.inspect_view(p.out,key,rec,edges,p)
    assert second==call and len(requests)==1 and ledger.summary()['requests']==1


def test_invalid_core_gets_one_retry_and_double_failure_abstains(tmp_path):
    key,rec,edges=job();requests=[]
    def transport(body):
        requests.append(body)
        return response('invalid' if len(requests)==1 else judgment(body['input'][1]['content']))
    p,ledger=provider(tmp_path,transport);call=runner.inspect_view(p.out,key,rec,edges,p)
    assert call['request_count']==2 and call['usage']['cost']==pytest.approx(.0002)
    assert requests[1]['input'][0]['content'].endswith(runner.FORMAT_RETRY)
    p2,_=provider(tmp_path/'bad',lambda body:response('invalid'))
    failed=runner.inspect_view(p2.out,key,rec,edges,p2)
    assert failed['request_count']==2 and 'abstention' in failed
    assert runner.validate_call(p2.out,failed,key,rec,edges)==failed


def test_interruption_after_raw_response_replays_without_model_draw(tmp_path,monkeypatch):
    key,rec,edges=job();requests=[]
    def transport(body):requests.append(body);return response(judgment(body['input'][1]['content']))
    p,ledger=provider(tmp_path,transport);original=runner.frozen
    def interrupted(path,data):
        if Path(path).parent.name=='attempts':raise KeyboardInterrupt('fixture after durable HTTP result')
        return original(path,data)
    monkeypatch.setattr(runner,'frozen',interrupted)
    with pytest.raises(KeyboardInterrupt):runner.inspect_view(p.out,key,rec,edges,p)
    assert len(requests)==1 and not (p.out/'completed_views').exists()
    monkeypatch.setattr(runner,'frozen',original)
    restarted=runner.DurableJudgeProvider(p.out,ledger,lambda body:pytest.fail('raw answer resampled'),'resume')
    call=runner.inspect_view(p.out,key,rec,edges,restarted)
    assert 'judgment' in call and len(requests)==ledger.summary()['requests']==1


def test_pending_request_blocks_without_resampling_and_received_response_recovers(tmp_path):
    p,ledger=provider(tmp_path,lambda body:pytest.fail('ambiguous request sent'))
    body=runner.payload('s','p');rid=ledger.reserve_call();path=p._path('s','p')
    runner.atomic(path,{'request_payload':body,'request_sha256':runner.request_hash(body),'state':'pending','ledger_request_id':rid})
    with pytest.raises(runner.RetainedRequestError,match='persisted'):p.complete('s','p')
    assert ledger.summary()['blocked'] and ledger.summary()['charged_or_reserved']==.05
    p,ledger=provider(tmp_path/'recover',lambda body:pytest.fail('received response resampled'))
    rid=ledger.reserve_call();raw=response('original',tokens=1200)
    runner.atomic(p._path('s','p'),{'request_payload':body,'request_sha256':runner.request_hash(body),
        'state':'response_received','ledger_request_id':rid,'provider_response':raw,'provider_response_canonical_sha256':digest(raw)})
    assert p.complete('s','p')[0]=='original' #2048 applies, not inherited actor512
    assert ledger.summary()['charged_or_reserved']==.0001


def test_unknown_cost_and_oversized_output_retain_reservations_and_stop(tmp_path):
    for name,cost,tokens in [('unknown',None,20),('oversized',.0001,2049),('large_cost',.07,20)]:
        p,ledger=provider(tmp_path/name,lambda body,c=cost,t=tokens:response('raw retained',c,t))
        with pytest.raises(runner.RetainedRequestError):p.complete('s','p')
        saved=json.loads(p._path('s','p').read_text())
        assert saved['provider_response']['output'][0]['content'][0]['text']=='raw retained'
        assert ledger.summary()['blocked']
        if cost==.07:assert ledger.summary()['charged_or_reserved']==.07


def test_tampered_judgment_config_resource_or_request_is_rejected(tmp_path):
    key,rec,edges=job();p,_=provider(tmp_path,lambda body:response(judgment(body['input'][1]['content'])))
    call=runner.inspect_view(p.out,key,rec,edges,p)
    for mutate in (lambda c:c['judgment']['source_scores'].update({next(iter(c['judgment']['source_scores'])):.9}),
                   lambda c:c['attempts'][0].update(max_output_tokens=1024),
                   lambda c:c['usage'].update(cost=float('nan'))):
        damaged=copy.deepcopy(call);mutate(damaged)
        with pytest.raises(ValueError):runner.validate_call(p.out,damaged,key,rec,edges)
    path=p.out/call['request_lineage']['request_artifact'];saved=json.loads(path.read_text());saved['native_text']='changed'
    runner.atomic(path,saved)
    with pytest.raises(ValueError,match='lineage'):runner.validate_call(p.out,call,key,rec,edges)


def test_fake_phase_bounds_shared_ledger_and_completion_gate(tmp_path,completed_generation):
    prepared=runner.prepare(completed_generation,tmp_path/'judge');requests=[]
    def transport(body):requests.append(body);return response(judgment(body['input'][1]['content']))
    phase=runner.run_phase(tmp_path/'judge',prepared,phase='small_fixture',phase_cap=.3,new_view_limit=3,workers=2,transport_factory=lambda:transport)
    assert phase['completed_after']==3 and len(requests)==3
    calls=runner.load_calls(tmp_path/'judge',prepared[2]);assert len(calls)==3
    coverage=runner.coverage(prepared[3],prepared[4],prepared[2],calls)
    assert not coverage['complete'] and coverage['assignments_by_status']['missing_views']>0
    with pytest.raises(ValueError,match='phase already attempted'):
        runner.run_phase(tmp_path/'judge',prepared,phase='small_fixture',phase_cap=.3,new_view_limit=3,transport_factory=lambda:pytest.fail('duplicate phase dispatched'))


def test_prior_pending_request_blocks_entire_phase_before_transport(tmp_path,completed_generation):
    prepared=runner.prepare(completed_generation,tmp_path/'judge');plan=prepared[0]
    ledger=NightBudget(plan['ledger'],'orphan_fixture',.3);backup=Path(plan['ledger']).read_bytes()
    try:
        rid=ledger.reserve_call();p=runner.DurableJudgeProvider(tmp_path/'judge',ledger,None,'orphan')
        body=runner.payload('orphan system','orphan prompt')
        runner.atomic(p._path('orphan system','orphan prompt'),{'state':'pending','request_payload':body,
                      'request_sha256':runner.request_hash(body),'ledger_request_id':rid})
        with pytest.raises(ValueError,match='aggregate blocked'):
            runner.run_phase(tmp_path/'judge',prepared,phase='blocked_fixture',new_view_limit=2,
                             transport_factory=lambda:pytest.fail('provider constructed despite orphan pending request'))
        assert ledger.summary()['blocked']
        assert json.loads(p._path('orphan system','orphan prompt').read_text())['state']=='uncertain_retained'
    finally:
        Path(plan['ledger']).write_bytes(backup)
