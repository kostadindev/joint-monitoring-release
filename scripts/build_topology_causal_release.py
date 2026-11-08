"""Build/verify a complete local causal-development dataset; never publish/call models.

Explicit durable artifact allowlists. All source and observation replay uses the
bundled frozen runtime in an isolated subprocess with network calls disabled.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack,contextmanager
import copy
import fcntl
import hashlib
import json
import math
import os
import platform
from importlib.metadata import version
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile

GEN_FILES=('plan.json','tasks.json','suites.json','records.json','coverage.json')
MON_FILES=('plan.json','records.json','assignments.json','exact_random_plans.json','diagnostic_locations.json',
           'activation.json','annotation_correction.json','secondary_alias_actions.json',
           'dispatch_order.json','schema_preflight.json','coverage.json','views.json',
           'per_record_metrics.json','analysis_scope.json')
CARD='''# Local causal topology-monitoring development candidate

This private local candidate contains self-authored synthetic tasks, native actor
outputs, controlled private intervention instructions, and actual judge responses.
It is not a publication, held-out benchmark, or evidence of a universal winning
strategy. Every predeclared world and unknown is retained; no activation filter.
The frozen plans retain exact actor and judge model/settings; this extension uses
a 512-token actor cap and a uniform 2048-token judge cap across arms, separate from
the earlier 1024-token workload judge. No cross-configuration judge reuse is claimed.
The four underlying cases are crossed with three causal topologies. Each suite has
clean/A/B/AB native worlds and an observational honest-error AB twin; twins add no
execution or independent sample. Case-order counterbalancing does not estimate a
within-case order effect. A joint action flip does not imply individually invisible
false fragments; inspect clean/single-intervention correctness and source masks.

`allocator_inputs.jsonl` is the ONLY allocator input. Each row contains exactly an
anonymous integer-node graph plus k and s. No names, roles, text, case/record IDs,
labels, trust, private instructions or outcomes are provided. Row positions are
bookkeeping, never features. Seeds/algorithm settings are declared separately.
Both selection and grouping finish before acquisition. `judge_data/observations.jsonl`
is a trusted acquisition service's full store, NOT free allocator input; expose only
acquired channel histories. `evaluation/` holds mappings and labels, including the
positive/known-benign/unknown source partition. Unknown channels are not benign.
`private/generation/` contains full native authorities and private actor prompts;
`audit/monitor/` contains frozen judge plans, exact prompts, retries, raw responses
and benchmark metrics. Neither directory is an allocator feature source.

Full-network and deterministic matching references remain. Diagnostic views use
privileged potential locations and are labeled separately from topology-only
policies. Exact-random rows are expectations over disjoint allocations with matched
sizes, not policies reading all candidate groups. Native malformed/empty outputs,
noncompliance, unknown provenance, model errors, judge abstentions and infeasible
allocations remain explicit. Source truth covers structured facts and declared
computations; prose is unverified, and source origin does not establish intent.
Original native records and annotations remain immutable in `private/generation/`.
The judge evaluation uses a separately frozen strict-domain annotation correction:
noncanonical decision strings are UNKNOWN, even when a human recognizes a synonym.
`annotation_correction.json` binds the original artifacts, adapter and corrected
records. `secondary_alias_actions.json` is a separately labeled posthoc symmetric
action-only sensitivity; it never repairs malformed JSON or changes source labels.
Ordinary string `reason` metadata remains unverified prose and does not invalidate
the deterministic checker's supported structured facts.

Run `python tooling/build_topology_causal_release.py --verify CANDIDATE` to verify
file hashes, the access boundary, frozen generation replay, exact allocations and
judge prompts/configuration/raw-response lineage, metrics and actual ledger cost.
This needs the project's Python dependencies (recorded in manifest.environment, not credentials) and makes no model
calls. Frozen sources are overlaid into temporary runtime directories; original
plan bytes stay intact. Only the generation directory's physical location is
relocated during replay. Original directory strings are retained as provenance.

Actual spending is the filtered shared-ledger snapshot, with confirmed charges,
retained uncertainty and reservations separate. Native predecessor reuse and
judge-view reuse do not create new charges; never sum policy replay usage as spend.
The snapshot includes unreferenced durable attempts/errors as well as scored views.
Finalized durable artifacts are included; a lost transport response cannot be
reconstructed from an uncertain reservation. This candidate is local only and has
not been published. Dataset redistribution licensing remains undeclared.
'''


def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')
def write_rows(path,values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(canonical(v)+'\n' for v in values))
def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]

def safe(name):
    path=Path(name)
    if path.is_absolute() or '..' in path.parts:raise ValueError('unsafe artifact path')
    return path


@contextmanager
def stable_run(root):
    with (Path(root)/'.run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('run is still active')
        yield


def collect(root,kind):
    root=Path(root);plan=read(root/'plan.json');base=GEN_FILES if kind=='generation' else MON_FILES
    paths=list(base)
    for name in base:
        if not (root/name).is_file():raise ValueError('missing complete '+kind+' artifact: '+name)
    paths+=['source/'+str(safe(name)) for name in plan['source_sha256']]
    for name,want in plan['source_sha256'].items():
        if sha(root/'source'/safe(name))!=want:raise ValueError('frozen source hash mismatch')
    for path in sorted(root.glob('phase_*.json')):
        if not re.fullmatch(r'phase_[A-Za-z0-9_-]+\.json',path.name):raise ValueError('invalid phase artifact name')
        paths.append(path.name)
    if kind=='generation':
        suites=read(root/'suites.json')
        if len(suites)!=plan['suites'] or len(read(root/'records.json'))!=plan['planned_records']:raise ValueError('incomplete native generation')
        declared={'completed_suites/'+s['suite_id']+'.json' for s in suites}
        actual={str(p.relative_to(root)) for p in (root/'completed_suites').glob('*.json')}
        if actual!=declared:raise ValueError('incomplete or extra native suites')
        paths+=sorted(declared);directories=('requests',)
    else:
        coverage=read(root/'coverage.json')
        if not coverage.get('complete') or coverage['completed_views']!=plan['unique_views']:raise ValueError('incomplete judge views')
        directories=('requests','attempts','completed_views')
    for directory in directories:
        for path in sorted((root/directory).glob('*.json')):
            pattern=r'[0-9a-f]{64}_[01]\.json' if directory=='attempts' else r'[0-9a-f]{64}\.json'
            if not re.fullmatch(pattern,path.name):raise ValueError('invalid durable artifact name')
            paths.append(str(path.relative_to(root)))
    if len(paths)!=len(set(paths)):raise ValueError('duplicate artifact allowlist')
    return paths


def anonymous_graph(graph):
    names=sorted(graph['nodes'],key=canonical);random.Random(20261106).shuffle(names)
    restore=dict(enumerate(names));forward={name:i for i,name in restore.items()}
    return {'nodes':list(range(len(names))),'edges':sorted([[forward[a],forward[b]] for a,b in graph['edges']])},restore


def allocator_catalog(records,assignments):
    public=[];indices={};mapping=[]
    for record in records:
        graph,restore=anonymous_graph(record['graph']);budgets=set()
        for p in assignments:
            if p['episode_id']!=record['id']:continue
            if p['information_access']=='anonymous directed topology only':budgets.add((p['k'],p['s']))
            elif p['comparison_scope']!='privileged_location_diagnostic':raise ValueError('undeclared policy access boundary')
        refs=[]
        for k,s in sorted(budgets):
            row={'graph':graph,'k':k,'s':s};key=canonical(row)
            if key not in indices:indices[key]=len(public);public.append(row)
            refs.append({'k':k,'s':s,'allocator_row':indices[key]})
        mapping.append({'record_id':record['id'],'anonymization_seed':20261106,
                        'anonymous_to_named':[[n,restore[n]] for n in restore],'inputs':refs})
    return public,mapping


def split_records(records):
    observations=[];labels=[]
    message_keys=('id','sender','recipient','round','timestamp','content','trusted')
    for index,record in enumerate(records):
        oid='observation_'+str(index)
        observations.append({'observation_id':oid,'workflow':record['workflow'],
            'graph':{k:record['graph'][k] for k in ('nodes','edges')},
            'messages':[{k:m[k] for k in message_keys} for m in record['messages']]})
        labels.append({'record_id':record['id'],'observation_id':oid,
            'metadata':{k:v for k,v in record.items() if k not in ('id','workflow','graph','messages','agent_attempts')},
            'graph_attributes':{k:v for k,v in record['graph'].items() if k not in ('nodes','edges')},
            'message_attributes':{m['id']:{k:v for k,v in m.items() if k not in message_keys} for m in record['messages']}})
    return observations,labels


def strict_boundary(root,records,assignments):
    public=rows(root/'allocator_inputs.jsonl');wanted,mapping=allocator_catalog(records,assignments)
    if public!=wanted or rows(root/'evaluation/allocator_mappings.jsonl')!=mapping:raise ValueError('allocator graph/mapping mismatch or leaked fields')
    for row in public:
        if set(row)!={'graph','k','s'} or set(row['graph'])!={'nodes','edges'}:raise ValueError('allocator data boundary violated')
        g=row['graph']
        if g['nodes']!=list(range(len(g['nodes']))) or any(type(v) is not int for e in g['edges'] for v in e):raise ValueError('allocator nodes must be anonymous integers')
    obs,labs=split_records(records)
    if rows(root/'judge_data/observations.jsonl')!=obs or rows(root/'evaluation/labels.jsonl')!=labs:raise ValueError('public observation/evaluation split mismatch')
    return len(public)


def ledger_snapshot(generation,monitor):
    gp=read(generation/'plan.json');mp=read(monitor/'plan.json')
    if gp['ledger']!=mp['ledger']:raise ValueError('generation and monitor must share declared ledger')
    raw=Path(gp['ledger']).read_bytes();ledger=json.loads(raw);prefixes=(generation.name+'_',monitor.name+'_')
    if prefixes[0].startswith(prefixes[1]) or prefixes[1].startswith(prefixes[0]):raise ValueError('ambiguous generation/monitor ledger prefixes')
    calls={key:c for key,c in ledger['calls'].items() if c['batch'].startswith(prefixes)}
    return {'authorized_credits':ledger['authorized_credits'],'calls':calls,
            'batches':{k:v for k,v in ledger['batches'].items() if k.startswith(prefixes)},
            'source_ledger_sha256':hashlib.sha256(raw).hexdigest(),
            'shared_total_charged_or_reserved_at_snapshot':math.fsum(c['charge'] for c in ledger['calls'].values()),
            'scope':'Filtered actual phase charges; shared total is contextual and includes other experiments.'}


def verify_ledger(root,manifest):
    ledger=read(root/'audit/ledger_snapshot.json');linked=set();summary={}
    if ledger['authorized_credits']!=10.:raise ValueError('ledger authorization mismatch')
    prefixes=(manifest['generation_run_name']+'_',manifest['monitor_run_name']+'_')
    if prefixes[0].startswith(prefixes[1]) or prefixes[1].startswith(prefixes[0]):raise ValueError('ambiguous ledger prefixes')
    for charge in ledger['calls'].values():
        if not charge['batch'].startswith(prefixes):raise ValueError('unrelated ledger batch in candidate')
        if type(charge['charge']) not in (int,float) or not math.isfinite(charge['charge']) or charge['charge']<0:raise ValueError('invalid actual ledger charge')
        if charge['state'] not in ('settled','reserved') and not charge['state'].startswith('uncertain'):raise ValueError('unknown ledger disposition')
    for kind,folder in [('generation',root/'private/generation'),('monitor',root/'audit/monitor')]:
        prefix=manifest[kind+'_run_name']+'_';charged={k:c for k,c in ledger['calls'].items() if c['batch'].startswith(prefix)}
        plan=read(folder/'plan.json');config=plan['model_config'] if kind=='generation' else plan['judge_config']
        byte_cap=plan['max_input_utf8_bytes'] if kind=='generation' else plan['max_input_bytes']
        for path in (folder/'requests').glob('*.json'):
            request=read(path);key=request['ledger_request_id'];body=request['request_payload']
            inputs=body['input']
            if len(inputs)!=2 or [m['role'] for m in inputs]!=['system','user']:raise ValueError('unexpected raw request messages')
            system,prompt=[m['content'] for m in inputs]
            expected={'model':config['model'],'input':[{'role':'system','content':system},{'role':'user','content':prompt}],
                      'temperature':config['temperature'],'max_output_tokens':config['max_output_tokens'],
                      'agentic_memory':False,'tools':[],'tool_choice':'none','store':False,'stream':False,'background':False}
            nbytes=len(system.encode())+len(prompt.encode())
            if body!=expected or nbytes>byte_cap or request['planned_input_bytes']!=nbytes:raise ValueError('raw request model/input/config mismatch')
            if path.stem!=digest([config,system,prompt]) or request['call_key']!=digest([system,prompt]) or request['request_sha256']!=hashlib.sha256(json.dumps(body).encode()).hexdigest():raise ValueError('raw request hash/key mismatch')
            if 'provider_response' in request and request['provider_response_canonical_sha256']!=digest(request['provider_response']):raise ValueError('retained raw response digest mismatch')
            if key not in charged or charged[key]['batch']!=prefix+request['origin_phase']:raise ValueError('durable request missing exact ledger provenance')
            if key in linked:raise ValueError('one ledger reservation assigned to multiple durable requests')
            linked.add(key)
            if request['state']=='complete' and (charged[key]['state']!='settled' or not math.isclose(charged[key]['charge'],request['usage']['cost'],rel_tol=0,abs_tol=1e-12)):
                raise ValueError('completed raw request charge mismatch')
        summary[kind]={'confirmed_credits':math.fsum(c['charge'] for c in charged.values() if c['state']=='settled'),
                       'uncertain_retained_credits':math.fsum(c['charge'] for c in charged.values() if c['state'].startswith('uncertain')),
                       'active_reserved_credits':math.fsum(c['charge'] for c in charged.values() if c['state']=='reserved'),
                       'actual_ledger_requests':len(charged)}
    summary['unlinked_ledger_requests_retained']=sorted(set(ledger['calls'])-linked)
    return summary


def runtime(root,destination):
    """Overlay identical declared sources; conflicts cannot be silently resolved."""
    for folder in (root/'private/generation',root/'audit/monitor'):
        plan=read(folder/'plan.json')
        for name,want in plan['source_sha256'].items():
            source=folder/'source'/safe(name);target=destination/safe(name)
            if sha(source)!=want:raise ValueError('bundled frozen source hash mismatch')
            if target.exists() and sha(target)!=want:raise ValueError('generation/monitor source version conflict')
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)


def replay_worker(root,runtime_root):
    # Import only bundled execution sources. No dispatcher or provider is called.
    sys.path[:0]=[str(runtime_root/'scripts'),str(runtime_root/'src')]
    import urllib.request
    urllib.request.urlopen=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('network disabled during release verification'))
    import jmon.llm
    jmon.llm.make_provider=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('provider construction forbidden during replay'))
    import run_topology_causal_monitor as monitor
    import topology_causal_monitor as evaluation
    from topology_causal_workflows import digest as frozen_digest
    generation=root/'private/generation';judge=root/'audit/monitor';plan=read(judge/'plan.json')
    records,diagnostics,activation,metadata=monitor.load_generation(generation)
    declared=copy.deepcopy(plan['generation']);declared['directory']=metadata['directory']
    if metadata!=declared:raise ValueError('frozen generation bindings differ beyond physical directory relocation')
    if plan['judge_config']!=monitor.CONFIG or plan['system']!=evaluation.SYSTEM_TEXT:raise ValueError('frozen judge configuration mismatch')
    jobs,assignments,expectations=evaluation.plan_library(records,system=plan['system'],judge_config=plan['judge_config'],diagnostic_locations=diagnostics)
    comparisons={'records':records,'diagnostic_locations':diagnostics,'activation':activation,'assignments':assignments,
                 'annotation_correction':metadata['annotation_correction'],
                 'secondary_alias_actions':metadata['secondary_alias_actions'],
                 'exact_random_plans':expectations,'dispatch_order':monitor.dispatch_order(assignments,expectations,jobs),
                 'schema_preflight':evaluation.schema_preflight(jobs,system=plan['system'],judge_config=plan['judge_config'])}
    for name,value in comparisons.items():
        if canonical(read(judge/(name+'.json')))!=canonical(value):raise ValueError('frozen '+name+' replay mismatch')
    for value,key in [(records,'records_sha256'),(diagnostics,'diagnostic_locations_sha256'),(assignments,'assignments_sha256'),
                      (expectations,'expectations_sha256'),(sorted(jobs),'view_keys_sha256'),
                      (comparisons['dispatch_order'],'dispatch_order_sha256'),(comparisons['schema_preflight'],'schema_preflight_sha256')]:
        if frozen_digest(value)!=plan[key]:raise ValueError('monitor plan hash mismatch: '+key)
    calls=monitor.load_calls(judge,jobs)
    if set(calls)!=set(jobs):raise ValueError('incomplete planned judge coverage')
    if len(records)!=plan['episodes'] or len({r['case_id'] for r in records})!=plan['case_clusters']:raise ValueError('episode/cluster coverage mismatch')
    if len(assignments)!=plan['assignments'] or len(expectations)!=plan['expectations'] or len(jobs)!=plan['unique_views']:raise ValueError('frozen plan count mismatch')
    with tempfile.TemporaryDirectory(prefix='causal-analysis-replay-') as temp:
        metrics=monitor.complete_analysis(Path(temp),records,assignments,expectations,jobs,calls)
        if read(Path(temp)/'analysis_scope.json')!=read(judge/'analysis_scope.json'):raise ValueError('analysis scope replay mismatch')
    if canonical(metrics)!=canonical(read(judge/'per_record_metrics.json')):raise ValueError('unknown-aware metrics replay mismatch')
    status=monitor.coverage(assignments,expectations,jobs,calls)
    saved_status=read(judge/'coverage.json')
    if not math.isclose(status.pop('replay_credits'),saved_status.pop('replay_credits'),rel_tol=0,abs_tol=1e-10) or status!=saved_status:raise ValueError('judge coverage snapshot mismatch')
    if {c['view_key']:c for c in read(judge/'views.json')}!=calls:raise ValueError('judge views snapshot mismatch')
    # Every durable bounded attempt is preserved, including failed/unreferenced
    # attempts; completed attempts must agree with the exact per-view draw.
    saved_attempts={p.stem:read(p) for p in (judge/'attempts').glob('*.json')}
    for key,call in calls.items():
        for attempt in call['attempts']:
            if canonical(saved_attempts.get(key+'_'+str(attempt['attempt'])))!=canonical(attempt):raise ValueError('saved raw attempt differs from completed view')
    from run_topology_causal_workflows import read_completed_suite
    raw_records=[r for suite in read(generation/'suites.json')
                 for r in read_completed_suite(generation,suite)['records']]
    if canonical(read(generation/'records.json'))!=canonical(raw_records):raise ValueError('generation records snapshot differs from immutable suites')
    from topology_causal_annotations import observation_digest
    if len(raw_records)!=len(records) or any(a['id']!=b['id'] or observation_digest(a)!=observation_digest(b)
                                           for a,b in zip(raw_records,records)):
        raise ValueError('annotation correction changed native observations or requests')
    gen_plan=read(generation/'plan.json');suite_ids=[s['suite_id'] for s in read(generation/'suites.json')]
    if len(records)!=gen_plan['planned_records']:raise ValueError('filtered native worlds')
    expected_coverage={'planned_suites':gen_plan['suites'],'completed_suites':len(suite_ids),'completed_suite_ids':suite_ids,
                       'records':len(records),'complete':True,'unknown_outcomes_retained':True,'no_policy_performance_inferred':True}
    if read(generation/'coverage.json')!=expected_coverage:raise ValueError('generation coverage snapshot mismatch')
    print(json.dumps({'records':len(records),'case_clusters':plan['case_clusters'],'native_suites':len(activation),'judge_views':len(calls),
                      'metric_rows':len(metrics),'unknown_source_channels':sum(len(r['labels']['unknown_source_edges']) for r in records),
                      'judge_abstentions':sum('abstention' in c for c in calls.values()),'all_planned_worlds_retained':True,
                      'original_native_observations_preserved':True,
                      'annotation_correction_version':metadata['annotation_correction']['version'],
                      'secondary_alias_suites':len(metadata['secondary_alias_actions'])}))


def replay(root):
    with tempfile.TemporaryDirectory(prefix='causal-release-runtime-') as temp:
        directory=Path(temp);runtime(root,directory)
        env={'PATH':os.environ.get('PATH',''),'LANG':os.environ.get('LANG','C.UTF-8')}
        process=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--replay-worker',str(root.resolve()),'--runtime',str(directory)],
                               capture_output=True,text=True,env=env)
        if process.returncode:raise ValueError('offline frozen replay failed: '+process.stderr[-4000:])
        return json.loads(process.stdout)


def verify(directory):
    root=Path(directory);manifest=read(root/'manifest.json')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual!=set(manifest['files_sha256'])|{'manifest.json'}:raise ValueError('missing or unexpected candidate files')
    for name,want in manifest['files_sha256'].items():
        if sha(root/safe(name))!=want:raise ValueError('candidate hash mismatch: '+name)
    gen=root/'private/generation';mon=root/'audit/monitor'
    if collect(gen,'generation')!=manifest['generation_files'] or collect(mon,'monitor')!=manifest['monitor_files']:raise ValueError('candidate artifact allowlist mismatch')
    records=read(mon/'records.json');assignments=read(mon/'assignments.json')
    inputs=strict_boundary(root,records,assignments);costs=verify_ledger(root,manifest)
    result=replay(root)
    if result!=manifest['replay_summary']:raise ValueError('candidate coverage summary mismatch')
    if costs!=read(root/'audit/research_spend.json'):raise ValueError('actual spend summary mismatch')
    return {**result,'allocator_inputs':inputs,'actual_spend':costs,'verified':True}


def build(generation,monitor,destination):
    generation=Path(generation).resolve();monitor=Path(monitor).resolve();destination=Path(destination).resolve()
    if destination.exists():raise ValueError('candidate already exists; verify it or choose a new version')
    with ExitStack() as stack:
        stack.enter_context(stable_run(generation));stack.enter_context(stable_run(monitor))
        gen_files=collect(generation,'generation');mon_files=collect(monitor,'monitor')
        destination.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.causal-candidate-',dir=destination.parent) as temp:
            root=Path(temp)
            for source,names,folder in [(generation,gen_files,'private/generation'),(monitor,mon_files,'audit/monitor')]:
                for name in names:
                    target=root/folder/safe(name);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,target)
            records=read(monitor/'records.json');assignments=read(monitor/'assignments.json')
            public,mappings=allocator_catalog(records,assignments);observations,labels=split_records(records)
            write_rows(root/'allocator_inputs.jsonl',public);write_rows(root/'evaluation/allocator_mappings.jsonl',mappings)
            write_rows(root/'judge_data/observations.jsonl',observations);write_rows(root/'evaluation/labels.jsonl',labels)
            write(root/'audit/ledger_snapshot.json',ledger_snapshot(generation,monitor))
            manifest={'version':'causal-local-development-v0.1','scope':'Private local candidate; no publishing or held-out claim.',
                      'environment':{'python':platform.python_version(),'networkx':version('networkx'),'numpy':version('numpy')},
                      'generation_run_name':generation.name,'monitor_run_name':monitor.name,
                      'generation_files':gen_files,'monitor_files':mon_files}
            write(root/'audit/research_spend.json',verify_ledger(root,manifest))
            (root/'README.md').write_text(CARD)
            tooling=root/'tooling'/Path(__file__).name;tooling.parent.mkdir();shutil.copyfile(__file__,tooling)
            manifest['replay_summary']=replay(root)
            manifest['files_sha256']={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
            write(root/'manifest.json',manifest);verified=verify(root)
            os.rename(root,destination)
    return {'candidate':str(destination),**verified}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generation');parser.add_argument('--monitor');parser.add_argument('--out')
    parser.add_argument('--verify');parser.add_argument('--replay-worker');parser.add_argument('--runtime')
    args=parser.parse_args()
    if args.replay_worker:replay_worker(Path(args.replay_worker),Path(args.runtime));return
    if args.verify:result=verify(args.verify)
    else:
        if not all((args.generation,args.monitor,args.out)):parser.error('build requires --generation --monitor --out')
        result=build(args.generation,args.monitor,args.out)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
