"""Verify topology-only release boundaries, budgets, provenance and frozen replay.

The allocator-facing file contains only anonymous graph + k + s. This verifier is
a trusted evaluator and may join private mappings, acquisition data and labels.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile

from verify_workload_release import (canonical,digest,read_rows,unique,
    validate_dataset,prompt,RETRY,expected_weights)

ANALYSIS_FILES=('metrics.jsonl','summary.json','primary_contrast.json','development_ranking.json',
                'allocation_oracle.json','research_spend.json','reliability.json','ledger_snapshot.json')
INPUT_LOCATIONS={'episodes.jsonl':'evaluation/original_records.jsonl.gz',
 'assignments.jsonl':'evaluation/assignments.jsonl.gz','exact_random_plans.jsonl':'evaluation/exact_random_plans.jsonl.gz',
 'views.jsonl':'audit/views.jsonl.gz','plan.json':'evaluation/plan.json'}
ANON_SEED=20261106
FEATURES=('intercept','group_size','directed_two_path','reciprocal_pair','shared_sender',
 'shared_receiver','shared_endpoint','mean_sender_indegree','mean_sender_outdegree',
 'mean_recipient_indegree','mean_recipient_outdegree','mean_recipient_reachability',
 'recipient_reachability_overlap','node_count','edge_density')


def split_observation(rec):
    public={'id':rec['id'],'workflow':rec['workflow'],'graph':rec['graph'],
            'messages':[{k:m[k] for k in ('id','sender','recipient','round','timestamp','content','trusted')} for m in rec['messages']]}
    labels={'id':rec['id'],'metadata':{k:v for k,v in rec.items() if k not in public},
            'message_labels':{m['id']:m['hidden'] for m in rec['messages']}}
    return public,labels


def join_judge_data(observations,labels):
    obs=unique(observations,lambda r:r['id'],'judge observation');labs=unique(labels,lambda r:r['id'],'evaluation labels')
    if set(obs)!=set(labs):raise ValueError('judge/label coverage mismatch')
    records=[]
    for identity,row in obs.items():
        if not re.fullmatch(r'work_[0-9a-f]{20}',identity) or set(row)!={'id','workflow','graph','messages'}:raise ValueError('judge observation identity/schema mismatch')
        if any(set(m)!={'id','sender','recipient','round','timestamp','content','trusted'} for m in row['messages']):raise ValueError('evaluation label leaked into judge message')
        lab=labs[identity]
        if set(lab)!={'id','metadata','message_labels'} or set(lab['metadata'])&set(row):raise ValueError('evaluation metadata overwrites judge data')
        if set(lab['message_labels'])!={m['id'] for m in row['messages']}:raise ValueError('message label coverage mismatch')
        rec=dict(row);rec.update(lab['metadata']);rec['messages']=[dict(m,hidden=lab['message_labels'][m['id']]) for m in row['messages']]
        records.append(rec)
    return records


def strict_allocator_input(value):
    if type(value) is not dict or set(value)!={'graph','k','s'}:raise ValueError('allocator input must contain only anonymous graph, k and s')
    g=value['graph']
    if type(g) is not dict or set(g)!={'nodes','edges'}:raise ValueError('allocator graph attributes forbidden')
    nodes=g['nodes'];edges=g['edges']
    if type(nodes) is not list or any(type(n) is not int for n in nodes) or nodes!=list(range(len(nodes))):raise ValueError('anonymous integer nodes required')
    if type(edges) is not list:raise ValueError('edge list required')
    for e in edges:
        if type(e) is not list or len(e)!=2 or any(type(n) is not int or n not in nodes for n in e) or e[0]==e[1]:raise ValueError('invalid anonymous edge')
    if len({tuple(e) for e in edges})!=len(edges):raise ValueError('duplicate anonymous edge')
    if type(value['k']) is not int or not 0<=value['k']<=len(edges) or type(value['s']) is not int or value['s']<1:raise ValueError('invalid allocator budgets')
    return value


def anonymous_graph(named,seed=ANON_SEED):
    """Independent reproduction of the frozen adapter, reading adjacency only."""
    names=sorted(named['nodes'],key=canonical)
    if len(names)!=len(set(names)):raise ValueError('duplicate named node')
    random.Random(seed).shuffle(names);restore=dict(enumerate(names));forward={v:k for k,v in restore.items()}
    public={'nodes':list(range(len(names))),'edges':sorted([[forward[u],forward[v]] for u,v in named['edges']])}
    strict_allocator_input({'graph':public,'k':0,'s':1})
    return public,restore


def allocator_catalog(records,assignments):
    """Return public rows and evaluation-only endpoint/index mappings."""
    recs=unique(records,lambda r:r['id'],'episode');public=[];indices={};mapping=[]
    budgets={rid:set() for rid in recs}
    for a in assignments:
        if a['information_access']=='anonymous directed topology only':budgets[a['episode_id']].add((a['k'],a['s']))
        elif a['comparison_scope']!='privileged_location_oracle':raise ValueError('unknown allocator information boundary')
    for rid,rec in recs.items():
        g,restore=anonymous_graph(rec['graph']);rows=[]
        for k,s in sorted(budgets[rid]):
            value={'graph':g,'k':k,'s':s};key=canonical(value)
            if key not in indices:indices[key]=len(public);public.append(value)
            rows.append({'k':k,'s':s,'allocator_row':indices[key]})
        mapping.append({'episode_id':rid,'anonymization_seed':ANON_SEED,
                        'anonymous_to_named':[[n,restore[n]] for n in sorted(restore)],'inputs':rows})
    return public,mapping


def verify_allocator_catalog(records,assignments,public,mappings):
    for value in public:strict_allocator_input(value)
    if len({canonical(v) for v in public})!=len(public):raise ValueError('duplicate allocator input row')
    wanted,expected=allocator_catalog(records,assignments)
    if wanted!=public or expected!=mappings:raise ValueError('anonymous graph/index/mapping mismatch')
    return len(public)


def acquisition_prompts(records,mapping,k,s,selected,groups):
    """Trusted acquisition step after allocation; returns only selected judge views.

    This evaluator function may read named records/mappings. Neither its records,
    mappings nor returned message text may be passed back to an allocator.
    """
    rec=records[mapping['episode_id']];restore=dict(mapping['anonymous_to_named'])
    selected=list(map(tuple,selected));flat=[tuple(e) for group in groups for e in group]
    if mapping['anonymization_seed']!=ANON_SEED:raise ValueError('undeclared anonymization seed')
    public,expected_restore=anonymous_graph(rec['graph'],mapping['anonymization_seed'])
    if restore!=expected_restore or len(mapping['anonymous_to_named'])!=len(restore):raise ValueError('acquisition mapping mismatch')
    strict_allocator_input({'graph':public,'k':k,'s':s});universe=set(map(tuple,public['edges']))
    if len(selected)!=len(set(selected)) or not set(selected)<=universe or len(selected)>k:raise ValueError('invalid anonymous acquisition')
    if len(flat)!=len(set(flat)) or sorted(flat)!=sorted(selected) or any(not 1<=len(g)<=s for g in groups):raise ValueError('invalid anonymous partition')
    return [prompt(rec,[(restore[u],restore[v]) for u,v in group]) for group in groups]


def verify_reuse(calls,source_rows,plan):
    sources={p['path']:p['sha256'] for p in plan['reuse_sources']};bykey=unique(source_rows,lambda r:r['view_key'],'reused source record')
    expected={c['view_key']:c for c in calls if 'reuse_provenance' in c}
    if set(bykey)!=set(expected):raise ValueError('reused source coverage mismatch')
    for key,call in expected.items():
        p=call['reuse_provenance'];row=bykey[key]
        if set(p)!={'path','file_sha256','line','record_sha256'} or type(p['line']) is not int or p['line']<1:
            raise ValueError('invalid reused source coordinates')
        if p['path'] not in sources or sources[p['path']]!=p['file_sha256']:raise ValueError('undeclared reused source file')
        if row['provenance']!=p or row.get('incremental_credits')!=0:raise ValueError('reused provenance/cost mismatch')
        raw=row['raw_record_line']
        if not isinstance(raw,str) or digest(raw.encode())!=p['record_sha256']:raise ValueError('reused raw-record hash mismatch')
        original=json.loads(raw)
        if {k:v for k,v in call.items() if k!='reuse_provenance'}!=original:raise ValueError('reused source judgment altered')
    return len(expected)


def verify_attempt_journal(calls,journal):
    """Preserve even unfinished historical attempts without inventing outcomes."""
    indexed={c['view_key']:c for c in calls};observed=Counter(canonical(a) for a in journal)
    required=Counter(canonical(a) for c in calls if 'reuse_provenance' not in c for a in c['attempts'])
    if any(observed[a]<n for a,n in required.items()):raise ValueError('raw attempt journal missing final-view attempts')
    import math
    for item in journal:
        attempt=item.get('attempt')
        if type(attempt) is not int or attempt not in (0,1):raise ValueError('invalid journal attempt index')
        system=item['system']
        if attempt and not system.endswith(RETRY):raise ValueError('journal retry suffix missing')
        base=system[:-len(RETRY)] if attempt else system
        key=digest(canonical([base,item['prompt']]).encode())
        if key not in indexed:raise ValueError('unplanned raw journal view')
        call=indexed[key]
        if sorted(map(tuple,item['edges']))!=sorted(map(tuple,call['edges'])):raise ValueError('journal acquisition mismatch')
        if digest(canonical([system,item['prompt']]).encode())!=item['prompt_sha256']:raise ValueError('journal request hash mismatch')
        for field in ('model','temperature','max_output_tokens','planned_input_bytes'):
            if item.get(field)!=call[field]:raise ValueError('journal judge settings mismatch')
        if not isinstance(item['raw'],str):raise ValueError('journal raw response is not text')
        for field in ('input_tokens','output_tokens','cost'):
            v=item['usage'][field]
            if type(v) not in (int,float) or not math.isfinite(v) or v<0:raise ValueError('invalid journal usage')
    return {'raw_attempts':len(journal),'final_view_attempts':sum(required.values()),
            'additional_historical_attempts':sum((observed-required).values())}


def verify_model(artifact,runtime,records):
    model=artifact['model']
    if model!=runtime or set(runtime)!={'features','coef','alpha','supervision'} or tuple(runtime['features'])!=FEATURES:raise ValueError('non-topology model schema')
    if len(runtime['coef'])!=len(FEATURES) or runtime['alpha']!=.01 or runtime['supervision']!='numerical development rewards; topology-only deployment':raise ValueError('unexpected model parameters')
    import math
    if any(type(x) not in (int,float) or not math.isfinite(x) for x in runtime['coef']):raise ValueError('invalid topology model coefficient')
    training=artifact['training_cases']
    if len(training)!=len(set(training)) or len(training)!=5 or artifact['training_episodes']!=51:raise ValueError('unexpected training source identities')
    current={r['case_id'] for r in records}
    for rec in records:current.update(rec['labels'].get('case_ids',[]))
    if set(training)&current or artifact['workload_identity_check']['overlap']:raise ValueError('training/workload identity overlap')
    if not artifact['source_sha256'] or not artifact['input_sha256']:raise ValueError('missing model source provenance')
    if artifact['reward_definition']['false_positive_penalty']!=2 or artifact['reward_definition']['strict_threshold'] is not True:raise ValueError('training reward mismatch')
    return len(training)



def raw_file(path):
    return gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()


def allowed_path(name):
    """Explicit release allowlist, independent of the file-hash manifest."""
    p=Path(name)
    if p.is_absolute() or '..' in p.parts:return False
    fixed={'README.md','allocator_inputs.jsonl.gz','allocator_parameters/learned_model.json',
        'evaluation/allocator_index.jsonl.gz','evaluation/labels.jsonl.gz',
        'evaluation/learned_model.json','evaluation/protocol.md','evaluation/prompt_adoption.json',
        'judge_data/observations.jsonl.gz','audit/reused_source_records.jsonl.gz',
        'audit/judge_calls.jsonl.gz','audit/validation_quarantine.jsonl.gz','audit/coverage.json','audit/support_source_manifest.json','analysis/manifest.json','source/pyproject.toml'}
    fixed.update(INPUT_LOCATIONS.values())
    fixed.update('analysis/'+(n+'.gz' if n.endswith('.jsonl') else n) for n in ANALYSIS_FILES)
    fixed.update('tooling/'+n for n in ('build_topology_release.py','verify_topology_release.py','verify_workload_release.py'))
    if name in fixed:return True
    if len(p.parts)==2 and p.parts[0]=='audit' and re.fullmatch(r'phase_[A-Za-z0-9_.-]+\.json',p.name):return True
    if p.suffix!='.py' or any(part.startswith('.') for part in p.parts):return False
    if len(p.parts)==2 and p.parts[0]=='training_source':return True
    if p.parts[:2]==('source','scripts') and len(p.parts)==3:return True
    if p.parts[:3]==('source','src','jmon'):return True
    if p.parts[:3] in (('analysis','source','scripts'),('analysis','source','tests')) and len(p.parts)==4:return True
    return False


def verify_analysis_bindings(root,manifest,plan,calls,assignments,expectations):
    am=json.loads((root/'analysis/manifest.json').read_text())
    expected_coverage={'complete':True,'unique_views':len(calls),'assignments':len(assignments)+len(expectations),'case_clusters':12}
    if am['run_name']!=manifest['run_name'] or am['coverage']!=expected_coverage:raise ValueError('analysis coverage mismatch')
    if set(am['inputs'])!=set(INPUT_LOCATIONS) or set(am['output_sha256'])!=set(ANALYSIS_FILES):raise ValueError('analysis artifact contract mismatch')
    if am['inputs']!=manifest['source_input_sha256']:raise ValueError('analysis input declaration mismatch')
    for name,path in INPUT_LOCATIONS.items():
        if digest(raw_file(root/path))!=am['inputs'][name]:raise ValueError('analysis input hash mismatch')
    for name,want in am['output_sha256'].items():
        path=root/'analysis'/(name+'.gz' if name.endswith('.jsonl') else name)
        if digest(raw_file(path))!=want:raise ValueError('analysis output hash mismatch')
    if am['execution_source_sha256']!=plan['source_hashes']:raise ValueError('analysis execution source mismatch')
    for name,want in am['analysis_source_sha256'].items():
        if not allowed_path('analysis/source/'+name):raise ValueError('unsafe analysis source')
        if digest((root/'analysis/source'/name).read_bytes())!=want:raise ValueError('analysis source hash mismatch')
    for rows,key in [(assignments,'assignments_sha256'),(expectations,'expectations_sha256')]:
        if digest(canonical(rows).encode())!=plan[key]:raise ValueError('frozen allocation hash mismatch')
    if digest(canonical(sorted(c['view_key'] for c in calls)).encode())!=plan['view_keys_sha256']:raise ValueError('frozen view library mismatch')
    for a in assignments:
        if len(a['selected'])!=a['k']:raise ValueError('allocation does not spend declared edge budget')
    ledger=json.loads((root/'analysis/ledger_snapshot.json').read_text())
    charged=list(ledger['calls'].values());prefix=manifest['run_name']+'_'
    if any(not c['batch'].startswith(prefix) or c['state']=='reserved' for c in charged):raise ValueError('ledger scope or outstanding reservation mismatch')
    import math
    if any(type(c['charge']) not in (int,float) or not math.isfinite(c['charge']) or c['charge']<0 for c in charged):raise ValueError('invalid ledger charge')
    states=Counter(c['state'] for c in charged)
    if any(s!='settled' and not s.startswith('uncertain') for s in states):raise ValueError('unrecognized final ledger state')
    spend=json.loads((root/'analysis/research_spend.json').read_text())
    expected={'incremental_confirmed_credits':math.fsum(c['charge'] for c in charged if c['state']=='settled'),
        'incremental_uncertain_retained_credits':math.fsum(c['charge'] for c in charged if c['state'].startswith('uncertain')),
        'active_reservations':0,'incremental_ledger_request_states':dict(states),
        'reused_unique_views':sum('reuse_provenance' in c for c in calls),
        'new_unique_views':sum('reuse_provenance' not in c for c in calls),
        'reused_source_logical_replay_credits':math.fsum(c['usage']['cost'] for c in calls if 'reuse_provenance' in c),
        'new_view_logical_replay_credits':math.fsum(c['usage']['cost'] for c in calls if 'reuse_provenance' not in c)}
    if any(spend.get(k)!=v for k,v in expected.items()):raise ValueError('research spending mismatch')
    return am


def verify(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    if not re.fullmatch(r'topology_workload_[A-Za-z0-9_.-]+',manifest['run_name']):raise ValueError('unsafe or unexpected run name')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json'}
    # Nested analysis manifests are data and must remain hashed too.
    actual.update(str(p.relative_to(root)) for p in root.rglob('manifest.json') if p!=root/'manifest.json')
    if actual!=set(manifest['files_sha256']):raise ValueError('missing/unmanifested release files')
    for name,want in manifest['files_sha256'].items():
        path=Path(name)
        if not allowed_path(name) or any((root/Path(*path.parts[:i])).is_symlink() for i in range(1,len(path.parts)+1)):raise ValueError('unsafe release path')
        if digest((root/path).read_bytes())!=want:raise ValueError('release hash mismatch: '+name)
    labels=read_rows(root/'evaluation/labels.jsonl.gz')
    records=join_judge_data(read_rows(root/'judge_data/observations.jsonl.gz'),labels)
    assignments=read_rows(root/'evaluation/assignments.jsonl.gz');expectations=read_rows(root/'evaluation/exact_random_plans.jsonl.gz')
    calls=read_rows(root/'audit/views.jsonl.gz');metrics=read_rows(root/'analysis/metrics.jsonl.gz')
    plan=json.loads((root/'evaluation/plan.json').read_text());compatible=dict(plan,expected_random_assignments=plan['expectations'])
    counts=validate_dataset('workload',records,assignments,expectations,calls,metrics,compatible)
    coverage=json.loads((root/'audit/coverage.json').read_text())
    if coverage.get('complete') is not True or coverage.get('unique_planned_views')!=len(calls) or coverage.get('unique_scored_views')!=len(calls):raise ValueError('run coverage artifact mismatch')
    counts['allocator_inputs']=verify_allocator_catalog(records,assignments,read_rows(root/'allocator_inputs.jsonl.gz'),read_rows(root/'evaluation/allocator_index.jsonl.gz'))
    counts['reused_views']=verify_reuse(calls,read_rows(root/'audit/reused_source_records.jsonl.gz'),plan)
    counts['new_views']=len(calls)-counts['reused_views']
    journal=root/'audit/judge_calls.jsonl.gz'
    if counts['new_views'] and not journal.is_file():raise ValueError('missing new-call raw attempt journal')
    verify_attempt_journal(calls,read_rows(journal) if journal.is_file() else [])
    artifact=json.loads((root/'evaluation/learned_model.json').read_text());runtime=json.loads((root/'allocator_parameters/learned_model.json').read_text())
    counts['training_cases']=verify_model(artifact,runtime,records)
    originals=read_rows(root/'evaluation/original_records.jsonl.gz')
    if records!=originals or digest(canonical(records).encode())!=manifest['records_canonical_sha256']:raise ValueError('original record reconstruction mismatch')
    if digest(gzip.decompress((root/'evaluation/original_records.jsonl.gz').read_bytes()))!=plan['records_sha256']:raise ValueError('original records hash differs from frozen plan')
    for name,want in plan['source_hashes'].items():
        if digest((root/'source'/name).read_bytes())!=want:raise ValueError('frozen run source mismatch')
    support_archive=json.loads((root/'audit/support_source_manifest.json').read_text())
    for name,want in manifest['support_source_sha256'].items():
        if digest((root/'source'/name).read_bytes())!=want or support_archive['files_sha256'].get('source/'+name)!=want:
            raise ValueError('archived support source mismatch')
    for name,want in artifact['source_sha256'].items():
        if digest((root/'training_source'/name).read_bytes())!=want:raise ValueError('frozen model source mismatch')
    if digest((root/'evaluation/learned_model.json').read_bytes())!=plan['learned_model_sha256']:raise ValueError('model artifact differs from frozen plan')
    if digest((root/'evaluation/protocol.md').read_bytes())!=plan['protocol_sha256']:raise ValueError('protocol differs from frozen plan')
    if digest((root/'evaluation/prompt_adoption.json').read_bytes())!=plan['prompt_adoption_sha256']:raise ValueError('prompt adoption differs from frozen plan')
    verify_analysis_bindings(root,manifest,plan,calls,assignments,expectations)
    if counts!=manifest['counts']:raise ValueError('release count mismatch')
    return counts


def replay_worker(root):
    """Executed in a fresh process so no working-tree policy module can leak in."""
    root=Path(root).resolve();manifest=json.loads((root/'manifest.json').read_text())
    if not re.fullmatch(r'topology_workload_[A-Za-z0-9_.-]+',manifest['run_name']):raise ValueError('unsafe or unexpected run name')
    am=json.loads((root/'analysis/manifest.json').read_text())
    with tempfile.TemporaryDirectory(prefix='topology-release-replay-') as tmp:
        temp=Path(tmp);execution=temp/'execution';shutil.copytree(root/'source',execution)
        shutil.copytree(root/'analysis/source',execution,dirs_exist_ok=True)
        run=temp/manifest['run_name'];run.mkdir()
        for original,bundled in INPUT_LOCATIONS.items():(run/original).write_bytes(raw_file(root/bundled))
        for name in ('learned_model.json','protocol.md','prompt_adoption.json'):
            shutil.copyfile(root/'evaluation'/name,run/name)
        shutil.copytree(root/'source',run/'source')
        sys.path[:0]=[str(execution/'scripts'),str(execution/'src')]
        from topology_workload_plan import plan_library
        records=read_rows(run/'episodes.jsonl');plan=json.loads((run/'plan.json').read_text())
        artifact=json.loads((run/'learned_model.json').read_text())
        jobs,assignments,expectations=plan_library(records,plan['system'],artifact['model'])
        if canonical(assignments)!=canonical(read_rows(run/'assignments.jsonl')):raise ValueError('frozen graph allocation replay mismatch')
        if canonical(expectations)!=canonical(read_rows(run/'exact_random_plans.jsonl')):raise ValueError('frozen exact expectation replay mismatch')
        if digest(canonical(sorted(jobs)).encode())!=plan['view_keys_sha256']:raise ValueError('frozen planned view replay mismatch')
        from analyze_topology_workload import analyze_complete
        output=temp/'replayed_analysis'
        result=analyze_complete(run,output=output,ledger=root/'analysis/ledger_snapshot.json')
        for name,want in am['output_sha256'].items():
            if digest((output/name).read_bytes())!=want:raise ValueError('frozen analysis replay mismatch: '+name)
        if result!=am:raise ValueError('frozen analysis manifest replay mismatch')
        with (run/'views.jsonl').open() as stream:
            raw_attempts=sum(len(json.loads(line)['attempts']) for line in stream)
        return {'allocation_replay':True,'logical_views_reparsed':len(jobs),
                'raw_attempts_reparsed':raw_attempts,
                'analysis_outputs_reproduced':len(am['output_sha256']),'model_calls':0}


def replay(root):
    root=Path(root).resolve();verify(root)
    process=subprocess.run([sys.executable,str(root/'tooling/verify_topology_release.py'),str(root),'--_replay-worker'],
                           text=True,capture_output=True)
    if process.returncode:raise ValueError('offline replay failed: '+process.stderr[-5000:])
    return json.loads(process.stdout)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('release');p.add_argument('--replay',action='store_true')
    p.add_argument('--_replay-worker',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
    if a._replay_worker:result=replay_worker(a.release)
    else:
        result={'verification':verify(a.release)}
        if a.replay:result['offline_replay']=replay(a.release)
    print(json.dumps(result,indent=2))
