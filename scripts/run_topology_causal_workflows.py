"""Prepare-default, bounded native causal workflow generation. No monitor calls.

Raw HTTP results are persisted before native text is delivered. An interrupted
in-flight request is never resent automatically. All worlds and unknowns remain.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
from topology_causal_workflows import task_manifest,layout,run_suite,canonical,digest,TOPOLOGIES,CONDITIONS
from night_budget import NightBudget

MODEL='gemma-4-26b-a4b-it'
CONFIG={'model':MODEL,'temperature':0,'max_output_tokens':512}
MAX_INPUT_BYTES=24000
RESERVATION=.05


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.tmp{os.getpid()}.{threading.get_ident()}')
    with temporary.open('w') as stream:
        json.dump(data,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def frozen(path,data):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=data:raise ValueError('frozen artifact changed: '+str(path))
    else:atomic(path,data)


def source_paths():
    import jmon
    root=Path(__file__).resolve().parents[1];package=Path(jmon.__file__).resolve().parent
    return {**{'scripts/'+name:root/'scripts'/name for name in ('run_topology_causal_workflows.py','topology_causal_workflows.py','night_budget.py')},
            'src/jmon/llm/adaption.py':package/'llm/adaption.py','src/jmon/llm/__init__.py':package/'llm/__init__.py'}


def prepare(out,ledger,seed=20261107):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    tasks=task_manifest(seed,4);suites=[]
    # Finish the three topology types for the first task before a larger case block.
    for task in tasks:
        for topology in TOPOLOGIES:
            graph,_,_=layout(task,topology)
            suites.append({'suite_id':topology+'_'+task['case_id'],'case_id':task['case_id'],
                           'task_sha256':digest(task),'topology':topology,'graph':graph,
                           'conditions':list(CONDITIONS)+['honest_error_combined'],
                           'planned_record_ids':[f'causal:{topology}:{task["case_id"]}:{c}' for c in list(CONDITIONS)+['honest_error_combined']],
                           'maximum_native_requests_before_exact_reuse':40 if topology=='fork_join' else 28,
                           'outcomes':'unknown until native execution; no compliance or activation promised'})
    sources=source_paths()
    plan={'version':'causal-generation-v1','scope':'Fresh development positive control; native actor execution only, no monitor or held-out claim.',
          'model_config':copy.deepcopy(CONFIG),'tasks_sha256':digest(tasks),'suites_sha256':digest(suites),'tasks':4,'suites':12,'planned_records':60,
          'maximum_native_requests_before_exact_reuse':384,
          'seed':seed,'ledger':str(Path(ledger).resolve()),'authorized_total_credits':10.,'per_request_reservation':RESERVATION,
          'max_input_utf8_bytes':MAX_INPUT_BYTES,'max_workers':4,
          'source_sha256':{name:sha(path) for name,path in sources.items()},
          'cache_policy':'Exact frozen model/configuration and full system/prompt. Reuse native responses, including malformed and empty text; never resample valid outputs. Ambiguous in-flight requests retained, never automatically resent.',
          'price_guard':'Each bounded HTTP request reserves 0.05 credits in shared ledger before dispatch. This conservative reservation is not a provider-enforced price guarantee; unknown or above-reservation billed cost blocks further calls.',
          'label_domain':'Explicit positive/known-benign/unknown source partition; structured facts and declared computations, prose unverified.',
          'phase_scope':'New-suite count bounds newly scheduled suites; exact requests depend on native predecessors and are additionally bounded by phase/shared reservations.'}
    frozen(out/'plan.json',plan);frozen(out/'tasks.json',tasks);frozen(out/'suites.json',suites)
    for name,path in sources.items():
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and dest.read_bytes()!=path.read_bytes():raise ValueError('frozen source changed: '+name)
        if not dest.exists():
            with dest.open('wb') as stream:stream.write(path.read_bytes());stream.flush();os.fsync(stream.fileno())
    return plan,tasks,suites


def payload(system,prompt):
    return {'model':CONFIG['model'],'input':[{'role':'system','content':system},{'role':'user','content':prompt}],
            'temperature':CONFIG['temperature'],'max_output_tokens':CONFIG['max_output_tokens'],
            'agentic_memory':False,'tools':[],'tool_choice':'none','store':False,'stream':False,'background':False}


def request_hash(body):
    # Adaption HTTP transport serializes json.dumps(payload), with this exact order.
    return hashlib.sha256(json.dumps(body).encode()).hexdigest()


class RetainedRequestError(RuntimeError):pass


class DurableActorProvider:
    """HTTP transport plus durable exact-request replay and NightBudget guard.

    Transport is constructed only by the live entrypoint. Empty native text is
    retained as a malformed observation instead of being retried or discarded.
    """
    def __init__(self,out,ledger,transport,phase,suite_id):
        self.out=Path(out);self.ledger=ledger;self.transport=transport;self.phase=phase;self.suite_id=suite_id

    def _path(self,system,prompt):return self.out/'requests'/(digest([CONFIG,system,prompt])+'.json')

    @staticmethod
    def _decode(response):
        usage=response.get('usage') or {}
        for field in ('input_tokens','output_tokens'):
            if type(usage.get(field)) is not int or usage[field]<0:raise ValueError('missing or invalid '+field)
        cost=usage.get('cost')
        if isinstance(cost,bool) or not isinstance(cost,(int,float)) or not math.isfinite(cost) or cost<0:raise ValueError('missing or invalid billed cost')
        text=''.join(part['text'] for item in response.get('output',[]) if item.get('type')=='message'
                     for part in item.get('content',[]) if part.get('type')=='output_text')
        if not isinstance(text,str):raise ValueError('nontext provider output')
        return text,{k:usage[k] for k in ('input_tokens','output_tokens','cost')}

    def complete(self,system,prompt):
        body=payload(system,prompt);path=self._path(system,prompt)
        nbytes=sum(len(m['content'].encode()) for m in body['input'])
        if nbytes>MAX_INPUT_BYTES:raise RetainedRequestError('native request exceeds frozen input-byte guard; not sent')
        # Failed branches can produce identical empty-input requests across
        # topologies. A per-request lock prevents duplicate concurrent dispatch.
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            return self._complete(system,prompt,body,path,nbytes)

    def _complete(self,system,prompt,body,path,nbytes):
        if path.exists():
            record=json.loads(path.read_text())
            if record['request_payload']!=body or record['request_sha256']!=request_hash(body):raise ValueError('durable request payload mismatch')
            if record.get('provider_response') is not None and record.get('provider_response_canonical_sha256')!=digest(record['provider_response']):raise ValueError('cached raw response hash mismatch')
            if record['state']=='complete':
                raw,usage=self._decode(record['provider_response'])
                if raw!=record['native_text'] or usage!=record['usage'] or hashlib.sha256(raw.encode()).hexdigest()!=record['native_text_sha256']:raise ValueError('cached native completion mismatch')
                return raw,usage
            if record['state']=='response_received':
                return self._finish(path,record)
            if record['state']=='pending':
                self.ledger.block(record['ledger_request_id'])
                record.update(state='uncertain_retained',error={'type':'InterruptedRequest','message':'No durable response; automatic resend forbidden.'})
                atomic(path,record)
            raise RetainedRequestError('persisted unsuccessful request: '+record['state'])
        record={'request_sha256':request_hash(body),'call_key':digest([system,prompt]),'request_payload':body,
                'origin_suite':self.suite_id,'origin_phase':self.phase,'planned_input_bytes':nbytes,'started_at':time.time()}
        reservation=self.ledger.reserve_call()
        record.update(state='pending',ledger_request_id=reservation);atomic(path,record)
        try:
            response=self.transport(body)
        except Exception as exc:
            self.ledger.block(reservation)
            record.update(state='uncertain_retained',error={'type':type(exc).__name__,'message':str(exc)[:300]},finished_at=time.time())
            atomic(path,record);raise RetainedRequestError('transport failed; reservation retained') from exc
        record.update(state='response_received',provider_response=response,provider_response_canonical_sha256=digest(response),response_at=time.time())
        atomic(path,record)
        return self._finish(path,record)

    def _finish(self,path,record):
        try:
            text,usage=self._decode(record['provider_response'])
            self.ledger.settle(record['ledger_request_id'],usage['cost'])
            if usage['output_tokens']>CONFIG['max_output_tokens']:
                self.ledger.block(record['ledger_request_id']);raise ValueError('provider exceeded frozen output cap')
        except Exception as exc:
            # A missing cost or invalid response may already have been billed.
            self.ledger.block(record['ledger_request_id'])
            record.update(state='uncertain_retained',error={'type':type(exc).__name__,'message':str(exc)[:300]},finished_at=time.time())
            atomic(path,record);raise RetainedRequestError('response invalid; charges retained') from exc
        record.update(state='complete',native_text=text,native_text_sha256=hashlib.sha256(text.encode()).hexdigest(),usage=usage,finished_at=time.time())
        atomic(path,record);return text,usage

    def lineage(self,attempt):
        path=self._path(attempt['system'],attempt['prompt'])
        if not path.exists():
            return {'request_sent':False,'reason':'Guard or reservation failed before HTTP dispatch.'}
        record=json.loads(path.read_text())
        return {'request_sent':True if 'provider_response' in record else None,'request_may_have_been_sent':True,'request_artifact':str(path.relative_to(self.out)),
                'request_artifact_sha256':sha(path),'transport_request_sha256':record['request_sha256'],
                'provider_response_canonical_sha256':record.get('provider_response_canonical_sha256'),
                'native_text_sha256':record.get('native_text_sha256'),'ledger_request_id':record['ledger_request_id'],
                'origin_phase':record['origin_phase'],'origin_suite':record['origin_suite'],'request_state':record['state']}


def validate_partition(record):
    labels=record['labels'];parts=[set(map(tuple,labels[k])) for k in ('false_origin_edges','known_benign_source_edges','unknown_source_edges')]
    if any(parts[i]&parts[j] for i in range(3) for j in range(i)):raise ValueError('overlapping source labels')
    if set.union(*parts)!=set(map(tuple,record['graph']['edges'])):raise ValueError('incomplete source label partition')



def _validate_saved_call(out,call):
    if call['call_key']!=digest([call['system'],call['prompt']]):raise ValueError('saved call key mismatch')
    lineage=call.get('request_lineage',{})
    if lineage.get('request_sent') is False:
        if call['status']!='provider_error' or 'request_artifact' in lineage:raise ValueError('unsent call has a native response')
        return
    relative=Path(lineage.get('request_artifact',''))
    if len(relative.parts)!=2 or relative.parts[0]!='requests' or relative.name!=digest([CONFIG,call['system'],call['prompt']])+'.json':raise ValueError('invalid request lineage path')
    request_path=Path(out)/relative
    if sha(request_path)!=lineage['request_artifact_sha256']:raise ValueError('durable request artifact hash mismatch')
    request=json.loads(request_path.read_text());body=payload(call['system'],call['prompt'])
    if request['request_payload']!=body or request['call_key']!=call['call_key']:raise ValueError('native call/request binding mismatch')
    if request['request_sha256']!=request_hash(body) or lineage['transport_request_sha256']!=request['request_sha256']:raise ValueError('raw request hash mismatch')
    if request['ledger_request_id']!=lineage['ledger_request_id'] or request['state']!=lineage['request_state'] or any(request[k]!=lineage[k] for k in ('origin_phase','origin_suite')):raise ValueError('request ledger/state/origin lineage mismatch')
    response=request.get('provider_response')
    if response is not None and (request.get('provider_response_canonical_sha256')!=digest(response) or lineage.get('provider_response_canonical_sha256')!=digest(response)):
        raise ValueError('raw response hash mismatch')
    if call['status']=='returned':
        if request['state']!='complete':raise ValueError('returned native call lacks completed request')
        raw,usage=DurableActorProvider._decode(response)
        raw_hash=hashlib.sha256(raw.encode()).hexdigest()
        if raw!=call['raw'] or raw!=request['native_text'] or usage!=call['usage'] or usage!=request['usage']:
            raise ValueError('native text or usage differs from raw response')
        if raw_hash!=lineage['native_text_sha256'] or raw_hash!=request['native_text_sha256']:raise ValueError('native text hash mismatch')
    elif call['status']!='provider_error':raise ValueError('unknown native call status')


def read_completed_suite(out,suite):
    """Validate cached artifacts and deterministically replay native observations.

    Replay supplies saved raw responses/errors only; no provider is constructed.
    An embedded digest and atomic replacement avoid a sidecar commit window.
    """
    out=Path(out);path=out/'completed_suites'/(suite['suite_id']+'.json')
    result=json.loads(path.read_text());unsigned={k:v for k,v in result.items() if k!='artifact_sha256'}
    if result.get('artifact_sha256')!=digest(unsigned):raise ValueError('completed suite artifact digest mismatch')
    if result.get('suite_id')!=suite['suite_id'] or result.get('task_sha256')!=suite['task_sha256']:raise ValueError('completed suite/task binding mismatch')
    records=result.get('records',[])
    if [r.get('id') for r in records]!=suite['planned_record_ids'] or [r.get('condition') for r in records]!=suite['conditions']:raise ValueError('completed suite record IDs/conditions mismatch')
    for rec in records:
        if rec.get('case_id')!=suite['case_id'] or rec.get('environment')!=suite['topology'] or rec.get('graph')!=suite['graph'] or rec.get('task_sha256')!=suite['task_sha256']:
            raise ValueError('completed record task/topology binding mismatch')
        validate_partition(rec)
    tasks=json.loads((out/'tasks.json').read_text());matches=[t for t in tasks if t['case_id']==suite['case_id']]
    if len(matches)!=1 or digest(matches[0])!=suite['task_sha256']:raise ValueError('frozen task binding mismatch')
    calls={c['call_key']:c for c in result['call_journal']}
    if len(calls)!=len(result['call_journal']):raise ValueError('duplicate saved native calls')
    for call in calls.values():_validate_saved_call(out,call)
    used=set()
    class SavedObservations:
        def complete(self,system,prompt):
            key=digest([system,prompt])
            if key not in calls:raise AssertionError('replay requests a missing saved native call')
            call=calls[key];used.add(key)
            if call['system']!=system or call['prompt']!=prompt:raise AssertionError('replay prompt mismatch')
            if call['status']=='returned':return call['raw'],call['usage']
            error=call['error']
            raise type(error['type'],(Exception,),{})(error['message'])
    replay=run_suite(matches[0],suite['topology'],SavedObservations())
    if used!=set(calls):raise ValueError('unused or missing saved native calls')
    # Every record-level attempt must point back to its verified journal entry.
    stripped=copy.deepcopy({k:result[k] for k in ('records','call_journal','activation')})
    for rec in stripped['records']:
        for attempt in rec['agent_attempts']:
            original=calls.get(attempt['call_key'])
            if original is None or attempt.pop('request_lineage',None)!=original['request_lineage']:raise ValueError('native attempt lineage mismatch')
    for call in stripped['call_journal']:call.pop('request_lineage')
    if canonical(replay)!=canonical(stripped):raise ValueError('completed native observations or annotations differ from deterministic replay')
    return result


def execute_suite(out,task,suite,provider):
    if digest(task)!=suite['task_sha256'] or task['case_id']!=suite['case_id']:raise ValueError('execution task does not match frozen suite')
    path=Path(out)/'completed_suites'/(suite['suite_id']+'.json')
    if path.exists():return read_completed_suite(out,suite)
    result=run_suite(task,suite['topology'],provider)
    if [r['id'] for r in result['records']]!=suite['planned_record_ids']:raise ValueError('unexpected native record IDs')
    for record in result['records']:
        validate_partition(record)
        for attempt in record['agent_attempts']:attempt['request_lineage']=provider.lineage(attempt)
    for call in result['call_journal']:call['request_lineage']=provider.lineage(call)
    result.update(suite_id=suite['suite_id'],task_sha256=digest(task),scope='All native worlds retained; honest twin adds no execution. Unknown labels are not benign.')
    result['artifact_sha256']=digest(result)
    frozen(path,result);return read_completed_suite(out,suite)


def checkpoint(out,suites):
    out=Path(out);records=[];done=[]
    for suite in suites:
        path=out/'completed_suites'/(suite['suite_id']+'.json')
        if path.exists():
            result=read_completed_suite(out,suite);records.extend(result['records']);done.append(suite['suite_id'])
    # Derived snapshot only; completed suite files are the immutable authority.
    atomic(out/'records.json',records)
    coverage={'planned_suites':len(suites),'completed_suites':len(done),'completed_suite_ids':done,'records':len(records),
              'complete':len(done)==len(suites),'unknown_outcomes_retained':True,'no_policy_performance_inferred':True}
    atomic(out/'coverage.json',coverage);return coverage



def phase_audit(out,selected,ledger):
    with ledger.edit() as shared:
        charged={key:copy.deepcopy(c) for key,c in shared['calls'].items() if c['batch']==ledger.batch}
    worlds=[]
    for sid in selected:
        path=Path(out)/'completed_suites'/(sid+'.json')
        if path.exists():worlds.extend(r for r in json.loads(path.read_text())['records'] if r['condition']!='honest_error_combined')
    return {'actual_confirmed_credits':math.fsum(c['charge'] for c in charged.values() if c['state']=='settled'),
            'retained_uncertain_credits':math.fsum(c['charge'] for c in charged.values() if c['state'].startswith('uncertain')),
            'active_reserved_credits':math.fsum(c['charge'] for c in charged.values() if c['state']=='reserved'),
            'ledger_request_states':dict(Counter(c['state'] for c in charged.values())),
            'ledger_requests':charged,'native_world_status_counts':dict(Counter(r['execution_status'] for r in worlds)),
            'logical_provider_error_attempts':sum(a['status']=='provider_error' for r in worlds for a in r['agent_attempts']),
            'scope':'Actual charges belong only to this phase batch. Native worlds exclude observational honest twins. Artifact coverage completion does not mean all actors returned valid output. Replayed predecessor usage must not be charged again.'}


def verify_prepared(out,plan,tasks,suites):
    out=Path(out)
    if plan['model_config']!=CONFIG:raise ValueError('actor configuration differs from freeze')
    if json.loads((out/'plan.json').read_text())!=plan:raise ValueError('prepared plan mismatch')
    if digest(tasks)!=plan['tasks_sha256'] or digest(suites)!=plan['suites_sha256']:raise ValueError('prepared task/suite mismatch')
    if json.loads((out/'tasks.json').read_text())!=tasks or json.loads((out/'suites.json').read_text())!=suites:raise ValueError('prepared input file mismatch')
    for name,path in source_paths().items():
        if sha(path)!=plan['source_sha256'][name] or sha(out/'source'/name)!=plan['source_sha256'][name]:raise ValueError('frozen generation source mismatch: '+name)


def run_phase(out,plan,tasks,suites,phase='pilot',phase_cap=.3,new_suite_limit=3,workers=1,transport_factory=None,resume=False):
    if type(workers) is not int or type(new_suite_limit) is not int or not 1<=workers<=4 or not 1<=new_suite_limit<=12 or not RESERVATION<=phase_cap<=10:raise ValueError('invalid phase bounds')
    out=Path(out);verify_prepared(out,plan,tasks,suites);phase_path=out/('phase_'+phase+'.json')
    if not phase or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in phase):raise ValueError('invalid phase name')
    # Validate every cached suite before any ledger/provider construction, even
    # when the selected phase would otherwise skip those files.
    for suite in suites:
        if (out/'completed_suites'/(suite['suite_id']+'.json')).exists():read_completed_suite(out,suite)
    ledger=NightBudget(plan['ledger'],out.name+'_'+phase,phase_cap,total_limit=10.,reserve=RESERVATION)
    if phase_path.exists():
        if not resume:raise ValueError('phase exists; use explicit --resume-phase to resume without resampling')
        phase_data=json.loads(phase_path.read_text())
        if any(phase_data[k]!=v for k,v in [('phase_cap',phase_cap),('workers',workers),('new_suite_limit',new_suite_limit)]):raise ValueError('resume phase bounds changed')
        if phase_data['status']=='complete':return checkpoint(out,suites)
        selected=phase_data['suite_ids']
    else:
        selected=[s['suite_id'] for s in suites if not (out/'completed_suites'/(s['suite_id']+'.json')).exists()][:new_suite_limit]
        phase_data={'phase':phase,'phase_cap':phase_cap,'new_suite_limit':new_suite_limit,'workers':workers,
                    'suite_ids':selected,'started_at':time.time(),'status':'running','errors':[]}
    for request_path in sorted((out/'requests').glob('*.json')):
        prior=json.loads(request_path.read_text())
        if prior['state']=='pending':
            ledger.block(prior['ledger_request_id'])
            prior.update(state='uncertain_retained',error={'type':'InterruptedRequest','message':'No durable response; automatic resend forbidden.'})
            atomic(request_path,prior)
    before=ledger.summary()
    # Reserve-call enforces shared and phase caps for each request, including
    # concurrent phases. This check never resets retained uncertain reservations.
    if before['blocked']:raise ValueError('shared budget blocked; no provider constructed')
    if before['charged_or_reserved']+max(0,phase_cap-before['batch_charged_or_reserved'])>before['authorized']:
        raise ValueError('phase cap exceeds remaining shared authorization')
    if not selected:return checkpoint(out,suites)
    atomic(phase_path,phase_data)
    task_byid={t['case_id']:t for t in tasks};suite_byid={s['suite_id']:s for s in suites}
    if transport_factory is None:
        def transport_factory():
            from jmon.llm import make_provider
            # HTTP only: durable adapter retains raw/empty text and owns budget.
            return make_provider({'kind':'adaption',**CONFIG,'cache_dir':str(out/'unused_provider_cache'),
                'cost':{'allow_live':True,'max_calls':1000,'max_tokens':20000000,'max_credits':phase_cap}}).http
    def one(sid):
        suite=suite_byid[sid]
        provider=DurableActorProvider(out,ledger,transport_factory(),phase,sid)
        return execute_suite(out,task_byid[suite['case_id']],suite,provider)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(one,sid):sid for sid in selected}
            for future in as_completed(futures):
                try:future.result()
                except Exception as exc:
                    phase_data['errors'].append({'suite_id':futures[future],'type':type(exc).__name__,'message':str(exc)[:300]})
                checkpoint(out,suites)
        phase_data['status']='complete' if all((out/'completed_suites'/(sid+'.json')).exists() for sid in selected) else 'incomplete'
    except BaseException:
        phase_data['status']='interrupted';raise
    finally:
        phase_data['finished_at']=time.time();phase_data['ledger']=ledger.summary();phase_data['audit']=phase_audit(out,selected,ledger);atomic(phase_path,phase_data)
    return checkpoint(out,suites)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='results/topology_causal_generation_development_v1')
    parser.add_argument('--ledger',default='results/night_2026-11-06_budget.json');parser.add_argument('--seed',type=int,default=20261107)
    parser.add_argument('--live',action='store_true');parser.add_argument('--phase',default='pilot');parser.add_argument('--phase-cap',type=float,default=.3)
    parser.add_argument('--new-suite-limit',type=int,default=3);parser.add_argument('--workers',type=int,default=1);parser.add_argument('--resume-phase',action='store_true')
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan,tasks,suites=prepare(out,args.ledger,args.seed)
        if not args.live:
            print(json.dumps({'prepared':True,'tasks':len(tasks),'suites':len(suites),'provider_constructed':False,'model_calls':0}));return
        print(json.dumps(run_phase(out,plan,tasks,suites,args.phase,args.phase_cap,args.new_suite_limit,args.workers,resume=args.resume_phase)))

if __name__=='__main__':main()
