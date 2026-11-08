"""Separate, prepare-default causal development judge runner; main run unchanged.

Require complete validated native generation; freeze all worlds including unknowns.
Uniform 2048 output tokens for every arm in this new block. Durable raw responses
survive interruption and are reused exactly; no valid judgment is resampled.
"""
from __future__ import annotations
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import threading
import time

from topology_causal_workflows import canonical,digest,layout
from topology_causal_annotations import corrected_suite,VERSION as ANNOTATION_VERSION
from topology_causal_monitor import (SYSTEM_TEXT,INPUT_ALLOWANCE,plan_library,schema_preflight,
                                     source_masks,analyze)
from monitor_record_workflows import FORMAT_RETRY,view_prompt
from record_workload_benchmark import parse_workload
from night_budget import NightBudget
from run_topology_causal_workflows import (atomic,frozen,sha,request_hash,DurableActorProvider,
    RetainedRequestError,verify_prepared as verify_generation,read_completed_suite)

CONFIG={'model':'gemma-4-31b-it','temperature':0,'max_output_tokens':2048}
RESERVATION=.05


def source_paths():
    import jmon
    root=Path(__file__).resolve().parents[1];package=Path(jmon.__file__).resolve().parent
    # Static closure includes local imports inside helper functions. Freeze the
    # small installed jmon package too, using its actual import location.
    pending=['run_topology_causal_monitor'];visited=set();paths={}
    while pending:
        name=pending.pop()
        if name in visited:continue
        visited.add(name);path=root/'scripts'/(name+'.py')
        if not path.exists():continue
        paths['scripts/'+name+'.py']=path
        for node in ast.walk(ast.parse(path.read_text())):
            modules=([node.module] if isinstance(node,ast.ImportFrom) else
                     [alias.name for alias in node.names] if isinstance(node,ast.Import) else [])
            for module in modules:
                if module and '.' not in module and (root/'scripts'/(module+'.py')).exists():pending.append(module)
    paths.update({'src/jmon/'+str(path.relative_to(package)):path for path in sorted(package.rglob('*.py'))})
    return paths


def load_generation(directory):
    """Read immutable suite authorities, never an unchecked records snapshot."""
    directory=Path(directory)
    generation=json.loads((directory/'plan.json').read_text())
    tasks=json.loads((directory/'tasks.json').read_text());suites=json.loads((directory/'suites.json').read_text())
    verify_generation(directory,generation,tasks,suites)
    records=[];artifacts={};diagnostics={};activations=[];corrections={};alias_actions=[]
    by_case={t['case_id']:t for t in tasks}
    if len(by_case)!=len(tasks) or len({s['suite_id'] for s in suites})!=len(suites):
        raise ValueError('duplicate generation tasks or suites')
    for suite in suites:
        path=directory/'completed_suites'/(suite['suite_id']+'.json')
        if not path.exists():raise ValueError('complete all frozen generation suites before judge preparation')
        raw_result=read_completed_suite(directory,suite)
        result=corrected_suite(raw_result,by_case[suite['case_id']])
        corrections[suite['suite_id']]={**result['annotation_correction'],
            'annotation_artifact_sha256':result['annotation_artifact_sha256'],
            'corrected_record_sha256':{r['id']:digest(r) for r in result['records']}}
        alias_actions.append({'suite_id':suite['suite_id'],**result['secondary_alias_actions']})
        artifacts[str(path.relative_to(directory))]=sha(path)
        records.extend(result['records']);activations.append(result['activation'])
        _,actors,destinations=layout(by_case[suite['case_id']],suite['topology'])
        fields=('purchase_age_days','return_window_days')
        locations={'reports':[(actors[f],destinations[f]) for f in fields],
                   'references':[('registry',actors[f]) for f in fields]}
        for record in result['records']:
            if record['split']!='development':raise ValueError('expected development records')
            source_masks(record);diagnostics[record['id']]=locations
    if len(records)!=generation['planned_records']:raise ValueError('generation record coverage mismatch')
    metadata={'directory':str(directory.resolve()),'plan_sha256':sha(directory/'plan.json'),
              'tasks_sha256':sha(directory/'tasks.json'),'suites_sha256':sha(directory/'suites.json'),
              'suite_artifacts':artifacts,'ledger':generation['ledger'],
              'annotation_correction':{'version':ANNOTATION_VERSION,'suites':corrections},
              'secondary_alias_actions':alias_actions}
    return records,diagnostics,activations,json.loads(canonical(metadata))


def dispatch_order(assignments,expectations,jobs):
    priority={key:1 for key in jobs}
    for row in assignments+expectations:
        if row['feasible'] and (row['selector']=='structural_path_pairs' or row['selector']=='all_edges' or
                               row['comparison_scope']=='privileged_location_diagnostic'):
            for key in row['view_keys']:priority[key]=0
    order=[]
    for tier in (0,1):
        keys=sorted(k for k,p in priority.items() if p==tier)
        random.Random(2026110700+tier).shuffle(keys);order.extend(keys)
    return order


def prepare(generation_dir,out):
    """No provider/ledger construction. Refuse incomplete or changed generation."""
    records,diagnostics,activations,generation=load_generation(generation_dir)
    jobs,assignments,expectations=plan_library(records,judge_config=CONFIG,diagnostic_locations=diagnostics)
    preflight=schema_preflight(jobs,judge_config=CONFIG);order=dispatch_order(assignments,expectations,jobs)
    sources=source_paths();out=Path(out)
    plan={'version':'causal-monitor-v1','scope':'Fresh causal development positive control; no held-out/general-win claim.',
          'generation':generation,'episodes':len(records),'case_clusters':len({r['case_id'] for r in records}),
          'records_sha256':digest(records),'diagnostic_locations_sha256':digest(diagnostics),
          'assignments_sha256':digest(assignments),'expectations_sha256':digest(expectations),
          'view_keys_sha256':digest(sorted(jobs)),'dispatch_order_sha256':digest(order),
          'assignments':len(assignments),'expectations':len(expectations),'unique_views':len(jobs),
          'source_sha256':{name:sha(path) for name,path in sources.items()},'judge_config':CONFIG,
          'system':SYSTEM_TEXT,'strict_threshold':.5,'format_retries':1,'max_input_bytes':INPUT_ALLOWANCE,
          'ledger':generation['ledger'],'authorized_total_credits':10.,'per_request_reservation':RESERVATION,
          'output_cap_rationale':'Uniform2048 across singleton/group/full-network in this separate extension;17-channel all-flag schema stress example is3300bytes on scripted fixtures. Token fit remains unverified, and2048 is not a guarantee. Frozen main remains1024; no cross-config reuse.',
          'native_validity':'All generated worlds, unknown source labels and noncompliance retained. Honest twin shares observations; no independent replication claim.',
          'recovery_unit':'Certified false-origin directed channels, including incidental errors/fanout; not necessarily both designated attackers.',
          'retry_policy':'At most one format-only retry for invalid core JSON; valid scores with invalid auxiliary evidence never resampled.',
          'coverage_gate':'Policy analysis requires every feasible planned view completed, including bounded abstentions; infeasible allocations remain explicit.',
          'schema_preflight_sha256':digest(preflight)}
    frozen(out/'plan.json',plan)
    for name,path in sources.items():
        destination=out/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists() and destination.read_bytes()!=path.read_bytes():raise ValueError('frozen source changed: '+name)
        if not destination.exists():
            with destination.open('wb') as stream:stream.write(path.read_bytes());stream.flush();os.fsync(stream.fileno())
    for name,data in [('records',records),('assignments',assignments),('exact_random_plans',expectations),
                      ('diagnostic_locations',diagnostics),('activation',activations),
                      ('annotation_correction',generation['annotation_correction']),
                      ('secondary_alias_actions',generation['secondary_alias_actions']),
                      ('dispatch_order',order),('schema_preflight',preflight)]:
        frozen(out/(name+'.json'),json.loads(canonical(data)))
    return plan,records,jobs,assignments,expectations,order


def payload(system,prompt):
    return {'model':CONFIG['model'],'input':[{'role':'system','content':system},{'role':'user','content':prompt}],
            'temperature':CONFIG['temperature'],'max_output_tokens':CONFIG['max_output_tokens'],
            'agentic_memory':False,'tools':[],'tool_choice':'none','store':False,'stream':False,'background':False}


class DurableJudgeProvider(DurableActorProvider):
    """Reuse proven request durability, with a separate frozen judge configuration."""
    model=CONFIG['model'];temperature=CONFIG['temperature'];max_output_tokens=CONFIG['max_output_tokens']

    def __init__(self,out,ledger,transport,phase):
        super().__init__(out,ledger,transport,phase,'causal_judge')

    def _path(self,system,prompt):return self.out/'requests'/(digest([CONFIG,system,prompt])+'.json')

    def complete(self,system,prompt):
        body=payload(system,prompt);path=self._path(system,prompt)
        nbytes=sum(len(m['content'].encode()) for m in body['input'])
        if nbytes>INPUT_ALLOWANCE:raise RetainedRequestError('judge request exceeds frozen input-byte guard; not sent')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            return self._complete(system,prompt,body,path,nbytes)

    def _finish(self,path,record):
        try:
            text,usage=self._decode(record['provider_response'])
            self.ledger.settle(record['ledger_request_id'],usage['cost'])
            if usage['output_tokens']>CONFIG['max_output_tokens']:raise ValueError('provider exceeded frozen output cap')
        except Exception as exc:
            self.ledger.block(record['ledger_request_id'])
            record.update(state='uncertain_retained',error={'type':type(exc).__name__,'message':str(exc)[:300]},finished_at=time.time())
            atomic(path,record);raise RetainedRequestError('response invalid; charges retained') from exc
        record.update(state='complete',native_text=text,native_text_sha256=hashlib.sha256(text.encode()).hexdigest(),usage=usage,finished_at=time.time())
        atomic(path,record);return text,usage


def _attempt(out,record,edges,key,index,provider):
    prompt=view_prompt(record,edges);system=SYSTEM_TEXT+(FORMAT_RETRY if index else '')
    raw,usage=provider.complete(system,prompt)
    lineage=provider.lineage({'system':system,'prompt':prompt})
    request=json.loads((Path(out)/lineage['request_artifact']).read_text())
    attempt={'episode_id':record['id'],'edges':edges,'view_key':key,'system':system,'prompt':prompt,
             'raw':raw,'usage':usage,'attempt':index,**CONFIG,
             'planned_input_bytes':len((SYSTEM_TEXT+FORMAT_RETRY+prompt).encode()),
             'prompt_sha256':digest([system,prompt]),'request_lineage':lineage,
             'latency_s':request['finished_at']-request['started_at']}
    frozen(Path(out)/'attempts'/(key+'_'+str(index)+'.json'),json.loads(canonical(attempt)))
    return attempt


def inspect_view(out,key,record,edges,provider):
    """Reconstruct from durable responses after crashes, without another draw."""
    attempts=[];disposition=None
    for index in (0,1):
        attempt=_attempt(out,record,edges,key,index,provider);attempts.append(attempt)
        try:judgment=parse_workload(attempt['raw'],record,edges)
        except (ValueError,TypeError,KeyError) as exc:
            if not index:continue
            disposition={'abstention':{'type':type(exc).__name__,'error':str(exc)[:180],
                                       'reason':'Both bounded core-schema attempts failed; no score inferred.'}}
        else:disposition={'judgment':judgment}
        break
    call={**attempts[-1],**disposition,'attempts':attempts,'request_count':len(attempts),
          'usage':{k:sum(a['usage'][k] for a in attempts) for k in ('input_tokens','output_tokens','cost')},
          'latency_s':sum(a['latency_s'] for a in attempts)}
    validate_call(out,call,key,record,edges)
    frozen(Path(out)/'completed_views'/(key+'.json'),json.loads(canonical(call)))
    return call


def validate_call(out,call,key,record,edges):
    """Validate every raw draw, exact config, durable provenance and retry choice."""
    prompt=view_prompt(record,edges);attempts=call.get('attempts',[])
    if key!=digest([SYSTEM_TEXT,prompt]) or call.get('view_key')!=key:raise ValueError('view identity mismatch')
    if not 1<=len(attempts)<=2 or call.get('request_count')!=len(attempts):raise ValueError('invalid attempt count')
    if ('judgment' in call)==('abstention' in call):raise ValueError('expected one disposition')
    for i,a in enumerate(attempts):
        system=SYSTEM_TEXT+(FORMAT_RETRY if i else '')
        if (a['prompt']!=prompt or a['system']!=system or a['attempt']!=i or a['view_key']!=key or
            sorted(map(tuple,a['edges']))!=sorted(map(tuple,edges)) or
            any(a[k]!=v for k,v in CONFIG.items()) or a['prompt_sha256']!=digest([system,prompt]) or
            a['planned_input_bytes']!=len((SYSTEM_TEXT+FORMAT_RETRY+prompt).encode())):
            raise ValueError('attempt configuration or acquisition mismatch')
        lineage=a['request_lineage'];path=Path(out)/'requests'/(digest([CONFIG,system,prompt])+'.json')
        if lineage['request_artifact']!=str(path.relative_to(out)) or lineage['request_artifact_sha256']!=sha(path):
            raise ValueError('request artifact lineage mismatch')
        request=json.loads(path.read_text());body=payload(system,prompt)
        if (request['state']!='complete' or request['request_payload']!=body or
            request['request_sha256']!=request_hash(body) or request['call_key']!=digest([system,prompt]) or
            request['provider_response_canonical_sha256']!=digest(request['provider_response']) or
            request['native_text_sha256']!=hashlib.sha256(a['raw'].encode()).hexdigest()):
            raise ValueError('durable request/response mismatch')
        raw,usage=DurableJudgeProvider._decode(request['provider_response'])
        if raw!=a['raw'] or raw!=request['native_text'] or usage!=a['usage'] or usage!=request['usage']:
            raise ValueError('raw response differs from saved judgment attempt')
        if usage['output_tokens']>CONFIG['max_output_tokens']:raise ValueError('output token cap exceeded')
        for name in ('ledger_request_id','origin_phase','origin_suite'):
            if lineage[name]!=request[name]:raise ValueError('request origin/ledger mismatch')
        if lineage['transport_request_sha256']!=request['request_sha256'] or lineage['request_state']!='complete':
            raise ValueError('request lineage digest/state mismatch')
        try:judgment=parse_workload(raw,record,edges)
        except (ValueError,TypeError,KeyError):
            if i==len(attempts)-1 and not (len(attempts)==2 and 'abstention' in call):raise ValueError('invalid final schema lacks bounded abstention')
        else:
            if i!=len(attempts)-1 or call.get('judgment')!=judgment or 'abstention' in call:raise ValueError('resampled valid answer or altered judgment')
    for field in ('prompt','system','raw','attempt','model','temperature','max_output_tokens','prompt_sha256','planned_input_bytes','request_lineage','edges'):
        if canonical(call[field])!=canonical(attempts[-1][field]):raise ValueError('final attempt differs from call')
    for field in ('input_tokens','output_tokens','cost'):
        total=call['usage'][field]
        if type(total) not in (int,float) or not math.isfinite(total) or total<0 or not math.isclose(total,sum(a['usage'][field] for a in attempts),rel_tol=0,abs_tol=1e-9):
            raise ValueError('invalid aggregate usage')
    return call


def load_calls(out,jobs):
    calls={}
    for path in sorted((Path(out)/'completed_views').glob('*.json')):
        key=path.stem
        if key not in jobs:raise ValueError('unknown completed view')
        call=json.loads(path.read_text());validate_call(out,call,key,*jobs[key]);calls[key]=call
    return calls


def coverage(assignments,expectations,jobs,calls):
    counts={'scored':0,'missing_views':0,'infeasible':0}
    for row in assignments+expectations:
        status='infeasible' if not row['feasible'] else 'scored' if all(k in calls for k in row['view_keys']) else 'missing_views'
        counts[status]+=1
    return {'unique_planned_views':len(jobs),'completed_views':len(calls),'complete':set(calls)==set(jobs),
            'assignments_by_status':counts,'judge_abstentions':sum('abstention' in c for c in calls.values()),
            'replay_credits':sum(c['usage']['cost'] for c in calls.values()),
            'cost_scope':'Logical replay usage including retries; actual new charges/reservations belong to shared ledger.'}


def checkpoint(out,assignments,expectations,jobs,calls):
    status=coverage(assignments,expectations,jobs,calls);atomic(Path(out)/'coverage.json',status)
    atomic(Path(out)/'views.json',list(calls.values()));return status


def complete_analysis(out,records,assignments,expectations,jobs,calls):
    if set(calls)!=set(jobs):raise ValueError('complete all feasible planned views before policy analysis')
    rows=analyze(records,assignments,expectations,calls)
    atomic(Path(out)/'per_record_metrics.json',rows)
    atomic(Path(out)/'analysis_scope.json',{'scope':'Development positive control; four task clusters, paired topologies/conditions/seeds; observational twins are not independent.',
           'complete_feasible_view_coverage':True,'rows':len(rows),'unknown_sources_retained':True,
           'infeasible_assignments':sum(not r['feasible'] for r in rows),
           'recovery_unit':'Certified false-origin directed channels; does not identify malicious intent.'})
    return rows


def run_phase(out,prepared,phase='pilot',phase_cap=.3,new_view_limit=30,workers=3,transport_factory=None):
    plan,records,jobs,assignments,expectations,order=prepared;out=Path(out)
    if (not phase or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in phase) or
        type(workers) is not int or not 1<=workers<=8 or type(new_view_limit) is not int or new_view_limit<1 or
        type(phase_cap) not in (int,float) or not math.isfinite(phase_cap) or not RESERVATION<=phase_cap<=10):
        raise ValueError('invalid bounded phase')
    # Rebuild/revalidate native authority and all frozen sources before spending.
    verified=prepare(plan['generation']['directory'],out)
    if canonical(verified[0])!=canonical(plan):raise ValueError('prepared plan changed')
    plan,records,jobs,assignments,expectations,order=verified
    calls=load_calls(out,jobs);todo=[k for k in order if k not in calls][:new_view_limit]
    if not todo:return checkpoint(out,assignments,expectations,jobs,calls)
    phase_path=out/('phase_'+phase+'.json')
    if phase_path.exists():raise ValueError('phase already attempted; use new phase name, retaining exact requests')
    ledger=NightBudget(plan['ledger'],out.name+'_'+phase,phase_cap)
    # Resolve durable responses without HTTP and block every ambiguous prior
    # dispatch before constructing any new transport, even outside this phase.
    recovery=DurableJudgeProvider(out,ledger,lambda body:(_ for _ in ()).throw(AssertionError('recovery attempted HTTP')),phase)
    for path in sorted((out/'requests').glob('*.json')):
        prior=json.loads(path.read_text())
        if prior['state']=='pending':
            ledger.block(prior['ledger_request_id'])
            prior.update(state='uncertain_retained',error={'type':'InterruptedRequest','message':'No durable response; automatic resend forbidden.'})
            atomic(path,prior)
        elif prior['state']=='response_received':
            messages=prior['request_payload']['input']
            recovery.complete(messages[0]['content'],messages[1]['content'])
    before=ledger.summary()
    if before['blocked'] or before['charged_or_reserved']+phase_cap>before['authorized']:
        raise ValueError('aggregate blocked or phase cap exceeds remaining authorization')
    phase_data={'phase':phase,'status':'running','started_at':time.time(),'view_keys':todo,
                'phase_cap':phase_cap,'workers':workers,'completed_before':len(calls)}
    frozen(phase_path,phase_data)
    if transport_factory is None:
        from jmon.llm import make_provider
        def transport_factory():
            provider=make_provider({'kind':'adaption',**CONFIG,'cost':{'allow_live':True,'max_calls':2*len(todo),
                                   'max_tokens':20000000,'max_credits':phase_cap}})
            return provider.http
    local=threading.local();errors=[];iterator=iter(todo)
    def one(key):
        if not hasattr(local,'provider'):local.provider=DurableJudgeProvider(out,ledger,transport_factory(),phase)
        return inspect_view(out,key,*jobs[key],local.provider)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending={}
            def submit():
                key=next(iterator,None)
                if key is not None:pending[pool.submit(one,key)]=key
            for _ in range(workers):submit()
            while pending:
                done,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    key=pending.pop(future)
                    try:calls[key]=future.result()
                    except Exception as exc:errors.append({'view_key':key,'type':type(exc).__name__,'error':str(exc)[:200]})
                checkpoint(out,assignments,expectations,jobs,calls)
                if not errors:
                    for _ in done:submit()
    except BaseException as exc:
        errors.append({'type':type(exc).__name__,'error':'Interrupted dispatcher; durable raw responses retained.'});raise
    finally:
        phase_data.update(status='failed' if errors else 'complete',finished_at=time.time(),errors=errors,
                          completed_after=len(calls),ledger=ledger.summary())
        atomic(phase_path,phase_data);checkpoint(out,assignments,expectations,jobs,calls)
    if errors:raise RuntimeError('judge phase failed; retained artifacts require audit: '+canonical(errors))
    return phase_data


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--generation-dir',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--live',action='store_true');parser.add_argument('--analyze',action='store_true')
    parser.add_argument('--phase',default='pilot');parser.add_argument('--phase-cap',type=float,default=.3)
    parser.add_argument('--new-view-limit',type=int,default=30);parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepared=prepare(args.generation_dir,out);plan,records,jobs,assignments,expectations,_=prepared
        calls=load_calls(out,jobs);status=checkpoint(out,assignments,expectations,jobs,calls)
        print(json.dumps({'plan':{k:plan[k] for k in ('episodes','case_clusters','assignments','expectations','unique_views')},'coverage':status},indent=2),flush=True)
        if args.live:
            result=run_phase(out,prepared,args.phase,args.phase_cap,args.new_view_limit,args.workers)
            print(json.dumps(result,indent=2),flush=True);calls=load_calls(out,jobs)
        if args.analyze:complete_analysis(out,records,assignments,expectations,jobs,calls)


if __name__=='__main__':main()
