"""Prepare and crash-safe actor-generation tests, using only fake HTTP transport."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_topology_causal_workflows import (prepare,DurableActorProvider,RetainedRequestError,run_phase,
    CONFIG,MAX_INPUT_BYTES,request_hash,payload,atomic)
from night_budget import NightBudget
from test_topology_causal_workflows import ScriptedProvider


def fake_response(text='{}',cost=.0001):
    return {'model':CONFIG['model'],'output':[{'type':'message','content':[{'type':'output_text','text':text}]}],
            'usage':{'input_tokens':10,'output_tokens':10,'cost':cost}}


def provider(tmp_path,transport,phase='pilot'):
    ledger=NightBudget(tmp_path/'ledger.json','test_'+phase,.3)
    return DurableActorProvider(tmp_path/'run',ledger,transport,phase,'suite_fixture'),ledger


def test_prepare_freezes_all_topologies_before_case_block_without_provider_or_ledger(tmp_path,monkeypatch):
    import jmon.llm
    monkeypatch.setattr(jmon.llm,'make_provider',lambda *a,**k:pytest.fail('prepare constructed provider'))
    out=tmp_path/'run';ledger=tmp_path/'budget.json';plan,tasks,suites=prepare(out,ledger)
    assert not ledger.exists()
    assert len(tasks)==4 and len(suites)==12
    assert [s['topology'] for s in suites[:3]]==['flat','sequential','fork_join']
    assert len({s['case_id'] for s in suites[:3]})==1
    assert plan['model_config']=={'model':'gemma-4-26b-a4b-it','temperature':0,'max_output_tokens':512}
    assert all(s['outcomes'].startswith('unknown') for s in suites)
    assert prepare(out,ledger)==(plan,tasks,suites)
    with pytest.raises(ValueError,match='frozen'):prepare(out,ledger,seed=91)


def test_exact_raw_request_replay_and_lineage_never_resamples_valid_or_empty_output(tmp_path):
    calls=[]
    def http(body):calls.append(body);return fake_response('')
    p,ledger=provider(tmp_path,http)
    assert p.complete('system','prompt')[0]==''
    assert p.complete('system','prompt')[0]==''
    assert len(calls)==1 and ledger.summary()['requests']==1
    lineage=p.lineage({'system':'system','prompt':'prompt'})
    assert lineage['transport_request_sha256']==hashlib.sha256(json.dumps(calls[0]).encode()).hexdigest()
    assert lineage['native_text_sha256']==hashlib.sha256(b'').hexdigest()
    assert lineage['request_state']=='complete'
    p2=DurableActorProvider(p.out,ledger,lambda b:pytest.fail('resampled'),'resume','suite_fixture')
    assert p2.complete('system','prompt')[0]==''
    assert p2.lineage({'system':'system','prompt':'prompt'})['origin_phase']=='pilot'


def test_pending_crash_is_retained_unknown_and_never_retried(tmp_path):
    p,ledger=provider(tmp_path,lambda b:pytest.fail('ambiguous request resent'))
    body=payload('s','p');reservation=ledger.reserve_call()
    atomic(p._path('s','p'),{'request_payload':body,'request_sha256':request_hash(body),'state':'pending','ledger_request_id':reservation})
    with pytest.raises(RetainedRequestError,match='persisted'):p.complete('s','p')
    summary=ledger.summary();assert summary['blocked'] and summary['charged_or_reserved']==.05
    assert json.loads(p._path('s','p').read_text())['state']=='uncertain_retained'


def test_durable_response_after_crash_is_finished_without_http_or_second_charge(tmp_path):
    p,ledger=provider(tmp_path,lambda b:pytest.fail('already received request resent'))
    body=payload('s','p');reservation=ledger.reserve_call();response=fake_response('native original')
    atomic(p._path('s','p'),{'request_payload':body,'request_sha256':request_hash(body),'state':'response_received',
                          'ledger_request_id':reservation,'provider_response':response,'provider_response_canonical_sha256':__import__('topology_causal_workflows').digest(response)})
    assert p.complete('s','p')[0]=='native original'
    assert ledger.summary()['requests']==1 and ledger.summary()['charged_or_reserved']==.0001
    assert p.complete('s','p')[0]=='native original'


def test_transport_failure_and_unknown_usage_retain_raw_or_error_and_budget(tmp_path):
    def fail(body):raise OSError('offline failure')
    p,ledger=provider(tmp_path,fail)
    with pytest.raises(RetainedRequestError):p.complete('s','p')
    record=json.loads(p._path('s','p').read_text());assert record['error']['type']=='OSError'
    assert ledger.summary()['charged_or_reserved']==.05
    p2,l2=provider(tmp_path/'other',lambda body:fake_response('raw survives',cost=None))
    with pytest.raises(RetainedRequestError):p2.complete('s','p')
    record=json.loads(p2._path('s','p').read_text())
    assert record['provider_response']['output'][0]['content'][0]['text']=='raw survives'
    assert l2.summary()['blocked'] and l2.summary()['charged_or_reserved']==.05


def test_input_guard_prevents_reservation_and_provider_dispatch(tmp_path):
    p,ledger=provider(tmp_path,lambda b:pytest.fail('oversized request sent'))
    with pytest.raises(RetainedRequestError,match='input-byte'):p.complete('s','x'*MAX_INPUT_BYTES)
    assert not ledger.path.exists()
    assert p.lineage({'system':'s','prompt':'x'*MAX_INPUT_BYTES})['request_sent'] is False


def test_above_reservation_charge_blocks_and_is_not_hidden(tmp_path):
    p,ledger=provider(tmp_path,lambda b:fake_response('native',cost=.07))
    with pytest.raises(RetainedRequestError):p.complete('s','p')
    assert ledger.summary()['charged_or_reserved']==.07 and ledger.summary()['blocked']


def test_phase_is_suite_bounded_keeps_unknowns_and_cache_lineage(tmp_path):
    out=tmp_path/'run';plan,tasks,suites=prepare(out,tmp_path/'ledger.json')
    native=ScriptedProvider();http_calls=[]
    def transport(body):
        http_calls.append(body)
        raw,_=native.complete(body['input'][0]['content'],body['input'][1]['content'])
        return fake_response(raw)
    coverage=run_phase(out,plan,tasks,suites,new_suite_limit=3,workers=2,transport_factory=lambda:transport)
    assert coverage['completed_suites']==3 and coverage['records']==15
    assert coverage['completed_suite_ids']==[s['suite_id'] for s in suites[:3]]
    records=json.loads((out/'records.json').read_text())
    assert {r['environment'] for r in records}=={'flat','sequential','fork_join'}
    for record in records:
        assert 'unknown_source_edges' in record['labels']
        for attempt in record['agent_attempts']:
            lineage=attempt['request_lineage'];request=json.loads((out/lineage['request_artifact']).read_text())
            assert request['native_text']==attempt['raw']
            assert request['call_key']==attempt['call_key']
            assert lineage['request_artifact_sha256']==hashlib.sha256((out/lineage['request_artifact']).read_bytes()).hexdigest()
    audit=json.loads((out/'phase_pilot.json').read_text())['audit']
    assert audit['native_world_status_counts']=={'complete':12}
    assert audit['retained_uncertain_credits']==0
    assert audit['actual_confirmed_credits']==pytest.approx(len(http_calls)*.0001)
    before=len(http_calls)
    run_phase(out,plan,tasks,suites,new_suite_limit=3,workers=2,transport_factory=lambda:pytest.fail('completed phase created provider'),resume=True)
    assert len(http_calls)==before
    with pytest.raises(ValueError,match='phase exists'):run_phase(out,plan,tasks,suites,new_suite_limit=3,workers=2,transport_factory=lambda:transport)


def test_malformed_native_outputs_are_kept_not_resampled_or_dropped(tmp_path):
    out=tmp_path/'run';plan,tasks,suites=prepare(out,tmp_path/'ledger.json');calls=[]
    def transport(body):calls.append(body);return fake_response('not JSON native output')
    run_phase(out,plan,tasks,suites,new_suite_limit=1,transport_factory=lambda:transport)
    records=json.loads((out/'records.json').read_text())
    assert len(records)==5
    assert all(r['execution_status']=='unresolved' for r in records)
    assert all(len(r['labels']['unknown_source_edges'])==7 for r in records)
    assert all(any(m['content']=='not JSON native output' for m in r['messages']) for r in records)
    assert len(calls)==9 # seven clean actor requests plus the two changed private systems


def test_shared_budget_block_prevents_provider_construction(tmp_path):
    out=tmp_path/'run';plan,tasks,suites=prepare(out,tmp_path/'ledger.json')
    atomic(plan['ledger'],{'authorized_credits':10.,'calls':{},'batches':{},'blocked':True})
    with pytest.raises(ValueError,match='blocked'):
        run_phase(out,plan,tasks,suites,transport_factory=lambda:pytest.fail('blocked phase created provider'))


def test_concurrent_identical_requests_share_one_actual_dispatch(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import time
    calls=[]
    def transport(body):calls.append(body);time.sleep(.02);return fake_response('same native text')
    first,ledger=provider(tmp_path,transport)
    clients=[DurableActorProvider(first.out,ledger,transport,'pilot','suite_'+str(i)) for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda p:p.complete('identical system','identical input'),clients))
    assert len(calls)==1 and ledger.summary()['requests']==1
    assert all(result[0]=='same native text' for result in results)


def test_interruption_after_saved_native_response_replays_prefix_without_resampling(tmp_path):
    from run_topology_causal_workflows import execute_suite
    out=tmp_path/'run';plan,tasks,suites=prepare(out,tmp_path/'ledger.json');native=ScriptedProvider();calls=[]
    def transport(body):
        calls.append(body);raw,_=native.complete(body['input'][0]['content'],body['input'][1]['content']);return fake_response(raw)
    ledger=NightBudget(plan['ledger'],'restart_fixture',.3)
    class InterruptAfterSaved(DurableActorProvider):
        n=0
        def complete(self,system,prompt):
            result=super().complete(system,prompt);self.n+=1
            if self.n==3:raise KeyboardInterrupt('offline interruption after durable completion')
            return result
    first=InterruptAfterSaved(out,ledger,transport,'pilot',suites[0]['suite_id'])
    with pytest.raises(KeyboardInterrupt):execute_suite(out,tasks[0],suites[0],first)
    assert len(calls)==3 and not (out/'completed_suites').exists()
    restarted=DurableActorProvider(out,ledger,transport,'pilot',suites[0]['suite_id'])
    result=execute_suite(out,tasks[0],suites[0],restarted)
    assert len(result['records'])==5 and len(calls)==12
    assert len({request_hash(p) for p in calls})==12
    assert ledger.summary()['requests']==12


def test_cached_native_or_raw_response_tampering_is_rejected(tmp_path):
    p,ledger=provider(tmp_path,lambda body:fake_response('original'))
    p.complete('s','p');path=p._path('s','p');original=json.loads(path.read_text())
    changed=copy.deepcopy(original);changed['native_text']='silently replaced';atomic(path,changed)
    with pytest.raises(ValueError,match='completion mismatch'):p.complete('s','p')
    changed=copy.deepcopy(original);changed['provider_response']['usage']['cost']=.02;atomic(path,changed)
    with pytest.raises(ValueError,match='response hash'):p.complete('s','p')
    assert ledger.summary()['requests']==1


def test_phase_preflight_blocks_interrupted_request_before_constructing_transport(tmp_path):
    out=tmp_path/'run';plan,tasks,suites=prepare(out,tmp_path/'ledger.json')
    old=NightBudget(plan['ledger'],'old_generation_phase',.3);reservation=old.reserve_call()
    atomic(out/'requests'/'interrupted.json',{'state':'pending','ledger_request_id':reservation})
    with pytest.raises(ValueError,match='blocked'):
        run_phase(out,plan,tasks,suites,transport_factory=lambda:pytest.fail('uncertain request did not block dispatch'))
    assert json.loads((out/'requests'/'interrupted.json').read_text())['state']=='uncertain_retained'
    assert old.summary()['charged_or_reserved']==.05


def test_interrupt_after_settlement_replays_saved_response_without_double_charge(tmp_path,monkeypatch):
    calls=[]
    def transport(body):calls.append(body);return fake_response('already delivered native answer')
    p,ledger=provider(tmp_path,transport);settle=ledger.settle
    def interrupted_settle(key,cost):
        settle(key,cost)
        raise KeyboardInterrupt('offline interruption between settlement and completion marker')
    monkeypatch.setattr(ledger,'settle',interrupted_settle)
    with pytest.raises(KeyboardInterrupt):p.complete('s','p')
    assert json.loads(p._path('s','p').read_text())['state']=='response_received'
    assert ledger.summary()['charged_or_reserved']==.0001
    monkeypatch.setattr(ledger,'settle',settle)
    assert p.complete('s','p')[0]=='already delivered native answer'
    assert len(calls)==1 and ledger.summary()['requests']==1
    assert ledger.summary()['charged_or_reserved']==.0001


def test_interrupt_during_transport_preserves_pending_and_forbids_automatic_resend(tmp_path):
    calls=[]
    def transport(body):
        calls.append(body)
        raise KeyboardInterrupt('offline interruption while HTTP disposition is unknown')
    p,ledger=provider(tmp_path,transport)
    with pytest.raises(KeyboardInterrupt):p.complete('s','p')
    assert json.loads(p._path('s','p').read_text())['state']=='pending'
    assert ledger.summary()['charged_or_reserved']==.05
    with pytest.raises(RetainedRequestError):p.complete('s','p')
    assert len(calls)==1 and ledger.summary()['blocked']
    assert ledger.summary()['charged_or_reserved']==.05


def test_concurrent_distinct_requests_cannot_bypass_phase_reservation_cap(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    entered=threading.Event();release=threading.Event();calls=[]
    def transport(body):
        calls.append(body);entered.set()
        assert release.wait(2),'offline test failed to release transport'
        return fake_response('one bounded native answer')
    ledger=NightBudget(tmp_path/'ledger.json','bounded_phase',.05)
    p=DurableActorProvider(tmp_path/'run',ledger,transport,'pilot','bounded_suite')
    with ThreadPoolExecutor(max_workers=2) as pool:
        first=pool.submit(p.complete,'s','one')
        try:
            assert entered.wait(2),'first offline transport never started'
            second=pool.submit(p.complete,'s','two')
            with pytest.raises(RuntimeError,match='budget exhausted'):second.result(timeout=2)
            assert len(calls)==1 and ledger.summary()['charged_or_reserved']==.05
            assert not p._path('s','two').exists()
        finally:release.set()
        assert first.result(timeout=2)[0]=='one bounded native answer'
    assert ledger.summary()['requests']==1 and ledger.summary()['charged_or_reserved']==.0001


def test_output_cap_violation_keeps_raw_response_and_known_charge_without_resampling(tmp_path):
    calls=[]
    def transport(body):
        calls.append(body);result=fake_response('native over-budget output remains auditable')
        result['usage']['output_tokens']=CONFIG['max_output_tokens']+1
        return result
    p,ledger=provider(tmp_path,transport)
    with pytest.raises(RetainedRequestError):p.complete('s','p')
    cached=json.loads(p._path('s','p').read_text())
    assert cached['provider_response']['output'][0]['content'][0]['text']=='native over-budget output remains auditable'
    assert 'output cap' in cached['error']['message']
    with pytest.raises(RetainedRequestError):p.complete('s','p')
    assert len(calls)==1 and ledger.summary()['blocked']
    assert ledger.summary()['charged_or_reserved']==.0001


def completed_fixture(tmp_path):
    out=tmp_path/'run';plan,tasks,suites=prepare(out,tmp_path/'ledger.json');native=ScriptedProvider()
    def transport(body):
        raw,_=native.complete(body['input'][0]['content'],body['input'][1]['content'])
        return fake_response(raw)
    run_phase(out,plan,tasks,suites,new_suite_limit=1,transport_factory=lambda:transport)
    path=out/'completed_suites'/(suites[0]['suite_id']+'.json')
    return out,plan,tasks,suites,path


def resign_fixture(path,result):
    from topology_causal_workflows import digest
    result['artifact_sha256']=digest({k:v for k,v in result.items() if k!='artifact_sha256'})
    atomic(path,result)


def test_completed_suite_hash_and_rehashed_wrong_record_ids_are_rejected(tmp_path):
    from run_topology_causal_workflows import read_completed_suite,checkpoint
    out,plan,tasks,suites,path=completed_fixture(tmp_path);original=json.loads(path.read_text())
    changed=copy.deepcopy(original);changed['records'][0]['id']='wrong-record';atomic(path,changed)
    with pytest.raises(ValueError,match='artifact digest'):read_completed_suite(out,suites[0])
    resign_fixture(path,changed)
    with pytest.raises(ValueError,match='IDs/conditions'):checkpoint(out,suites)
    atomic(path,original)
    assert len(read_completed_suite(out,suites[0])['records'])==5


def test_rehashed_annotation_changes_cannot_pass_native_observation_replay(tmp_path):
    from run_topology_causal_workflows import read_completed_suite
    out,plan,tasks,suites,path=completed_fixture(tmp_path);changed=json.loads(path.read_text())
    changed['records'][0]['labels']['action_correct']=False
    resign_fixture(path,changed)
    with pytest.raises(ValueError,match='deterministic replay'):read_completed_suite(out,suites[0])


def test_invalid_cached_suite_blocks_next_phase_before_provider_or_ledger_mutation(tmp_path):
    out,plan,tasks,suites,path=completed_fixture(tmp_path)
    before=Path(plan['ledger']).read_bytes();changed=json.loads(path.read_text())
    changed['records'].pop();resign_fixture(path,changed)
    with pytest.raises(ValueError,match='IDs/conditions'):
        run_phase(out,plan,tasks,suites,phase='extension',new_suite_limit=1,
                  transport_factory=lambda:pytest.fail('invalid cached suite constructed a transport'))
    assert Path(plan['ledger']).read_bytes()==before
    assert not (out/'phase_extension.json').exists()


def test_completed_suite_binds_native_records_to_durable_request_bytes(tmp_path):
    from run_topology_causal_workflows import read_completed_suite
    out,plan,tasks,suites,path=completed_fixture(tmp_path);suite=json.loads(path.read_text())
    raw_path=out/suite['call_journal'][0]['request_lineage']['request_artifact']
    raw=json.loads(raw_path.read_text());raw['native_text']='later replaced';atomic(raw_path,raw)
    with pytest.raises(ValueError,match='request artifact hash'):read_completed_suite(out,suites[0])


def test_complete_artifact_coverage_can_retain_all_unresolved_empty_native_worlds(tmp_path):
    from run_topology_causal_workflows import read_completed_suite
    out=tmp_path/'run';plan,tasks,suites=prepare(out,tmp_path/'ledger.json')
    coverage=run_phase(out,plan,tasks,suites,new_suite_limit=12,workers=4,
                       transport_factory=lambda:lambda body:fake_response(''))
    assert coverage['complete'] is True and coverage['records']==60 and coverage['completed_suites']==12
    phase=json.loads((out/'phase_pilot.json').read_text())
    assert phase['status']=='complete' and phase['audit']['native_world_status_counts']=={'unresolved':48}
    for suite in suites:
        cached=read_completed_suite(out,suite)
        assert cached['activation']['strict_joint_activation'] is None
        assert all(r['execution_status']=='unresolved' and r['labels']['action_correct'] is None for r in cached['records'])
        assert all(r['labels']['unknown_source_edges'] for r in cached['records'])
        assert all(call['raw']=='' for call in cached['call_journal'])
