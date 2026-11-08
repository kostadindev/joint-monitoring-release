"""Independent crash/concurrency integration review; fake HTTP only."""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_topology_causal_monitor as runner
from test_run_topology_causal_monitor import job,provider,response,judgment,completed_generation


def test_retry_response_survives_crash_without_a_third_model_draw(tmp_path,monkeypatch):
    key,rec,edges=job();requests=[]
    def transport(body):
        requests.append(body)
        return response('invalid core JSON' if len(requests)==1 else judgment(body['input'][1]['content']))
    client,ledger=provider(tmp_path,transport);frozen=runner.frozen
    def interrupt_second_attempt(path,data):
        if Path(path).parent.name=='attempts' and Path(path).stem.endswith('_1'):
            raise KeyboardInterrupt('offline interruption after durable retry response')
        return frozen(path,data)
    monkeypatch.setattr(runner,'frozen',interrupt_second_attempt)
    with pytest.raises(KeyboardInterrupt):runner.inspect_view(client.out,key,rec,edges,client)
    assert len(requests)==2 and ledger.summary()['requests']==2
    monkeypatch.setattr(runner,'frozen',frozen)
    recovered=runner.DurableJudgeProvider(client.out,ledger,lambda body:pytest.fail('saved retry was resampled'),'restart')
    call=runner.inspect_view(client.out,key,rec,edges,recovered)
    assert call['request_count']==2 and 'judgment' in call
    assert call['attempts'][0]['raw']=='invalid core JSON'
    assert len(requests)==2 and ledger.summary()['requests']==2
    assert ledger.summary()['charged_or_reserved']==pytest.approx(.0002)


def test_concurrent_identical_judge_views_share_one_actual_request(tmp_path):
    key,rec,edges=job();requests=[]
    def transport(body):
        requests.append(body);time.sleep(.02)
        return response(judgment(body['input'][1]['content']))
    first,ledger=provider(tmp_path,transport)
    clients=[runner.DurableJudgeProvider(first.out,ledger,transport,'concurrent') for _ in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        calls=list(pool.map(lambda c:runner.inspect_view(c.out,key,rec,edges,c),clients))
    assert len(requests)==1 and ledger.summary()['requests']==1
    assert all(call==calls[0] for call in calls)
    assert len(runner.load_calls(first.out,{key:(rec,edges)}))==1


def test_phase_uses_reconstructed_frozen_jobs_not_caller_mutated_objects(tmp_path,completed_generation):
    out=tmp_path/'judge';prepared=runner.prepare(completed_generation,out);requests=[]
    # A caller can retain the correct plan while changing every mutable derived
    # object. The dispatcher must use its independently reconstructed authority.
    poisoned=(copy.deepcopy(prepared[0]),[],{},[],[],[])
    def transport(body):requests.append(body);return response(judgment(body['input'][1]['content']))
    result=runner.run_phase(out,poisoned,phase='review_reconstruction',new_view_limit=1,workers=1,
                            transport_factory=lambda:transport)
    assert result['completed_after']==1 and len(requests)==1
    assert set(runner.load_calls(out,prepared[2]))=={prepared[5][0]}
