"""Complete local topology-only dataset candidate; no publishing or model calls."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import gzip
import json
from pathlib import Path
import shutil
import tempfile
from verify_workload_release import canonical,digest,read_rows,validate_dataset
from verify_topology_release import (allocator_catalog,split_observation,verify_allocator_catalog,
    verify_model,verify_reuse,verify_attempt_journal,verify)

SUPPORT_SCRIPTS=('situation_suite.py','record_allocation.py','record_pair_search.py','record_policy_extensions.py')
ANALYSIS_FILES=('metrics.jsonl','summary.json','primary_contrast.json','development_ranking.json',
                'allocation_oracle.json','research_spend.json','reliability.json','ledger_snapshot.json')
INPUT_LOCATIONS={'episodes.jsonl':'evaluation/original_records.jsonl.gz',
 'assignments.jsonl':'evaluation/assignments.jsonl.gz','exact_random_plans.jsonl':'evaluation/exact_random_plans.jsonl.gz',
 'views.jsonl':'audit/views.jsonl.gz','plan.json':'evaluation/plan.json'}
CARD='''# Anonymous-topology workload monitoring: local development release candidate

This is a local complete-run candidate, not a publication or held-out result. The
project code is MIT; dataset redistribution licensing remains undeclared. It contains
self-authored simulated workloads and recorded judge calls, not MESA or external raw
incident data, credentials or provider caches.

## Strict access boundary

`allocator_inputs.jsonl.gz` is the ONLY per-example allocator data. Every row is
exactly `{graph: {nodes: [integers], edges: [[integer, integer], ...]}, k, s}`. No
roles, trust, names, message content/lengths, case IDs, mappings, targets, labels or
judge scores are present. Row order is bookkeeping, not a feature. Fixed algorithm
parameters and seeds come from the declared benchmark configuration. The optional
runtime learned model in allocator_parameters/ contains only topology features and
fixed coefficients; its full supervision audit is evaluation-only.

Both selection and grouping must finish from this input. A trusted acquisition
service then maps the anonymous selections/groups to recorded channels and provides
their full message histories to the judge. `acquisition_prompts` in the verifier
shows this transition and enforces the acquired-edge budget and disjoint group cap.
Do not pass its named records, private mappings or returned text back to an allocator.

`judge_data/observations.jsonl.gz` is the acquisition service's full message store,
NOT free allocator input. It contains recorded names/workflow/graph and public
messages (including shown protected-reference markings), without hidden labels.
The service must expose only the chosen channels. `evaluation/` contains private
mappings/indexes, labels, conditions, original records, plans and allocation outputs.
`analysis/` and `audit/` are also evaluation-only: raw calls and scores are not policy
features. Original work_<20 hex digits> IDs are opaque bookkeeping and preserved
in these private/evaluator files to reproduce source hashes exactly.

## Scientific scope

All 288 records are development variants of twelve original target cases, crossing
4/16 cases per workload, clean/deliberate-false/honest-error conditions, audit-copy
alignment and history. These are message interventions on one shared skeleton,
not 288 independent attacks or distinct topology families. Error/attack twins share
public views and judgments. Labels identify factual origins, not malicious intent.
The original five-case training library is separate development; the 512-token
training judge interface differs from this 1,024-token/citation interface. Training
IDs, source hashes, weights and frozen model are retained, not relabeled as held out.

Ordinary policies partition acquired edges once within k and cap s. Exact matched
random rows describe expectations over uniform partitions of the same selection
and size multiset. Candidate groups are a shared research library, not all executed
by one allocation. Their rational inclusion weights preserve edge and call costs.
24 KB is a declared input operating limit; 60 KB full-context references are separate
allowance classes. Infeasible assignments and judge abstentions remain explicit.

Protected references, content-based rule checking and pooled joins are available to
detectors AFTER the graph-only allocation. They may not feed keys into grouping.
Privileged location/allocation oracles are evaluator diagnostics, never prospective
policies. Full-context and rule baselines remain even when they outperform the LLM.

## Reproduction and cost

`python tooling/verify_topology_release.py RELEASE_DIRECTORY` checks hashes, strict
allocator schemas and mappings, public-label separation, exact prompts, budgets,
expected weights, source metrics/baselines, source/model hashes and reused-call
provenance. Add `--replay` to regenerate frozen allocations and complete analysis
with bundled execution/analysis sources and ledger snapshot. Replay needs the
project dependencies and makes no model calls. It uses temporary files; original
plan bytes remain unchanged and the ledger is supplied as an explicit override.

Reused calls retain their historical usage as standalone replay cost, with exact
source-file/line/record hashes and raw source records. Their incremental cost is
zero. New-run research spend comes from its filtered actual ledger, including
transport losses; never sum per-policy replay costs or bill reused views again.
Whole source-file hashes are archival provenance; selected raw-line hashes are
independently verifiable here without redistributing the entire superseded library.

Run sources, post-run analysis sources and training sources are separated. The
execution bundle also includes explicitly hashed support Python modules needed by
historical imports. Four transitive script imports and support package modules are
bound to the earlier local v0.1 manifest, retained as audit/support_source_manifest.json;
these dependencies were not all listed in the new execution plan's source_hashes.
Verification establishes stated integrity/reproduction checks,
not universal benchmark realism, rationale correctness or deployment security.
'''


def compressed_bytes(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as stream,gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as zipped:zipped.write(data)

def write_rows(path,rows):compressed_bytes(path,''.join(canonical(r)+'\n' for r in rows).encode())

def raw_file(path):return gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()

def safe_relative(name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe source path')
    return p

@contextmanager
def stable_run(root):
    with (root/'.run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('topology experiment is still running')
        yield


def collect_reused(calls,plan):
    wanted={};output=[]
    declared={r['path']:r['sha256'] for r in plan['reuse_sources']}
    for call in calls:
        if 'reuse_provenance' not in call:continue
        p=call['reuse_provenance'];wanted.setdefault(p['path'],{})[p['line']]=call
    for path,lines in wanted.items():
        source=Path(path)
        if path not in declared or digest(source.read_bytes())!=declared[path]:raise ValueError('reused source file changed or unavailable')
        seen=set()
        with source.open() as stream:
            for number,line in enumerate(stream,1):
                if number not in lines:continue
                call=lines[number];seen.add(number)
                output.append({'view_key':call['view_key'],'provenance':call['reuse_provenance'],
                               'raw_record_line':line,'incremental_credits':0.})
        if seen!=set(lines):raise ValueError('reused source line missing')
    output.sort(key=lambda r:r['view_key']);verify_reuse(calls,output,plan)
    return output


def load_complete(root):
    root=Path(root);analysis=root/'analysis'
    required=('plan.json','episodes.jsonl','assignments.jsonl','exact_random_plans.jsonl','views.jsonl',
              'learned_model.json','protocol.md','prompt_adoption.json','coverage.json')
    for name in required:
        if not (root/name).is_file():raise ValueError('missing run artifact: '+name)
    for name in ANALYSIS_FILES+('manifest.json',):
        if not (analysis/name).is_file():raise ValueError('missing complete analysis: '+name)
    plan=json.loads((root/'plan.json').read_text());am=json.loads((analysis/'manifest.json').read_text())
    coverage=json.loads((root/'coverage.json').read_text())
    if coverage.get('complete') is not True or coverage.get('unique_planned_views')!=plan['unique_views'] or coverage.get('unique_scored_views')!=plan['unique_views']:
        raise ValueError('incomplete frozen run coverage')
    expected_coverage={'complete':True,'unique_views':plan['unique_views'],'assignments':plan['assignments']+plan['expectations'],'case_clusters':12}
    if am['coverage']!=expected_coverage:raise ValueError('analysis coverage mismatch')
    if am['run_name']!=root.name or am['coverage']['complete'] is not True:raise ValueError('wrong or incomplete analysis')
    if set(am['inputs'])!=set(INPUT_LOCATIONS) or set(am['output_sha256'])!=set(ANALYSIS_FILES):raise ValueError('analysis artifact contract mismatch')
    for name,want in am['inputs'].items():
        if digest((root/safe_relative(name)).read_bytes())!=want:raise ValueError('stale analysis input: '+name)
    for name,want in am['output_sha256'].items():
        if digest((analysis/safe_relative(name)).read_bytes())!=want:raise ValueError('analysis output hash mismatch')
    if am['execution_source_sha256']!=plan['source_hashes']:raise ValueError('analysis execution source declaration mismatch')
    for names,base in [(plan['source_hashes'],root/'source'),(am['analysis_source_sha256'],analysis/'source')]:
        for name,want in names.items():
            if digest((base/safe_relative(name)).read_bytes())!=want:raise ValueError('snapshot source hash mismatch')
    records=read_rows(root/'episodes.jsonl');assignments=read_rows(root/'assignments.jsonl');expectations=read_rows(root/'exact_random_plans.jsonl')
    calls=read_rows(root/'views.jsonl');metrics=read_rows(analysis/'metrics.jsonl')
    if digest((root/'episodes.jsonl').read_bytes())!=plan['records_sha256']:raise ValueError('frozen records mismatch')
    for rows,key in [(assignments,'assignments_sha256'),(expectations,'expectations_sha256')]:
        if digest(canonical(rows).encode())!=plan[key]:raise ValueError('frozen allocations mismatch')
    if digest(canonical(sorted(c['view_key'] for c in calls)).encode())!=plan['view_keys_sha256']:raise ValueError('incomplete or extra planned views')
    compatible=dict(plan,expected_random_assignments=plan['expectations'])
    counts=validate_dataset('workload',records,assignments,expectations,calls,metrics,compatible)
    public,mappings=allocator_catalog(records,assignments);counts['allocator_inputs']=verify_allocator_catalog(records,assignments,public,mappings)
    model=json.loads((root/'learned_model.json').read_text());counts['training_cases']=verify_model(model,model['model'],records)
    reused=collect_reused(calls,plan);counts['reused_views']=len(reused);counts['new_views']=len(calls)-len(reused)
    journal=root/'judge_calls.jsonl'
    if counts['new_views'] and not journal.is_file():raise ValueError('missing new-call raw attempt journal')
    verify_attempt_journal(calls,read_rows(journal) if journal.is_file() else [])
    if digest((root/'learned_model.json').read_bytes())!=plan['learned_model_sha256']:raise ValueError('frozen learned model changed')
    for name,key in [('protocol.md','protocol_sha256'),('prompt_adoption.json','prompt_adoption_sha256')]:
        if digest((root/name).read_bytes())!=plan[key]:raise ValueError('frozen protocol/adoption changed')
    ledger=json.loads((analysis/'ledger_snapshot.json').read_text());prefix=root.name+'_'
    if any(not c['batch'].startswith(prefix) or c['state']=='reserved' for c in ledger['calls'].values()):raise ValueError('ledger scope or active reservation mismatch')
    if any(json.loads(p.read_text())['status']=='running' for p in root.glob('phase_*.json')):raise ValueError('unterminated phase')
    return {'plan':plan,'analysis_manifest':am,'records':records,'assignments':assignments,'expectations':expectations,
            'calls':calls,'metrics':metrics,'public_inputs':public,'mappings':mappings,'model':model,'reused':reused,'counts':counts}


def _build_candidate(run,out,project_root,replay=True):
    run=Path(run);out=Path(out);project_root=Path(project_root);stage=Path(__file__).resolve().parents[1]
    if out.exists():raise ValueError('refusing to overwrite topology release destination')
    if not run.is_dir():raise ValueError('missing topology run')
    with stable_run(run):
        data=load_complete(run)
        # Resolve training snapshots and support modules before writing a candidate.
        training={}
        for name,want in data['model']['source_sha256'].items():
            if Path(name).name!=name or not name.endswith('.py'):raise ValueError('unsafe training source name')
            choices=[stage/'scripts'/name,run/'source/scripts'/name]
            source=next((p for p in choices if p.is_file() and digest(p.read_bytes())==want),None)
            if source is None:raise ValueError('frozen training source unavailable: '+name)
            training[name]=source
        if not (project_root/'src/jmon').is_dir():raise ValueError('support package unavailable')
        archive=stage/'results/keep/record_development_v0_1'
        archived_manifest=json.loads((archive/'manifest.json').read_text())
        supports={}
        for name in SUPPORT_SCRIPTS:
            path='source/scripts/'+name;want=archived_manifest['files_sha256'][path]
            if digest((archive/path).read_bytes())!=want:raise ValueError('archived support source mismatch')
            supports['scripts/'+name]=(archive/path,want)
        for source in sorted((project_root/'src/jmon').rglob('*.py')):
            rel=str(source.relative_to(project_root));want=archived_manifest['files_sha256']['source/'+rel]
            if digest(source.read_bytes())!=want:raise ValueError('support package differs from archived version: '+rel)
            if rel not in data['plan']['source_hashes']:supports[rel]=(source,want)
        out.mkdir(parents=True)
        write_rows(out/'allocator_inputs.jsonl.gz',data['public_inputs'])
        write_rows(out/'evaluation/allocator_index.jsonl.gz',data['mappings'])
        pairs=[split_observation(r) for r in data['records']]
        write_rows(out/'judge_data/observations.jsonl.gz',(p for p,l in pairs))
        write_rows(out/'evaluation/labels.jsonl.gz',(l for p,l in pairs))
        for source,dest in INPUT_LOCATIONS.items():
            target=out/dest;raw=(run/source).read_bytes()
            if target.suffix=='.gz':compressed_bytes(target,raw)
            else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        for name in ('learned_model.json','protocol.md','prompt_adoption.json'):
            shutil.copyfile(run/name,out/'evaluation'/name)
        target=out/'allocator_parameters/learned_model.json';target.parent.mkdir()
        target.write_text(json.dumps(data['model']['model'],indent=2)+'\n')
        write_rows(out/'audit/reused_source_records.jsonl.gz',data['reused'])
        if (run/'judge_calls.jsonl').exists():compressed_bytes(out/'audit/judge_calls.jsonl.gz',(run/'judge_calls.jsonl').read_bytes())
        if (run/'validation_quarantine.jsonl').exists():compressed_bytes(out/'audit/validation_quarantine.jsonl.gz',(run/'validation_quarantine.jsonl').read_bytes())
        shutil.copyfile(run/'coverage.json',out/'audit/coverage.json')
        for source in run.glob('phase_*.json'):shutil.copyfile(source,out/'audit'/source.name)
        for name in ANALYSIS_FILES+('manifest.json',):
            source=run/'analysis'/name;target=out/'analysis'/(name+'.gz' if name.endswith('.jsonl') else name)
            if target.suffix=='.gz':compressed_bytes(target,source.read_bytes())
            else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        # Support modules are explicitly .py only; frozen files overlay them below.
        support=[]
        for source in sorted((project_root/'src/jmon').rglob('*.py')):
            if source.is_symlink():raise ValueError('support symlink forbidden')
            rel=source.relative_to(project_root);target=out/'source'/rel
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);support.append(str(rel))
        shutil.copyfile(project_root/'pyproject.toml',out/'source/pyproject.toml')
        shutil.copyfile(archive/'manifest.json',out/'audit/support_source_manifest.json')
        for name,(source,want) in supports.items():
            target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        for name in data['plan']['source_hashes']:
            target=out/'source'/safe_relative(name);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(run/'source'/name,target)
        for name in data['analysis_manifest']['analysis_source_sha256']:
            target=out/'analysis/source'/safe_relative(name);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(run/'analysis/source'/name,target)
        for name,source in training.items():
            target=out/'training_source'/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target)
        for name in ('build_topology_release.py','verify_topology_release.py','verify_workload_release.py'):
            target=out/'tooling'/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(stage/'scripts'/name,target)
        (out/'README.md').write_text(CARD)
        manifest={'version':'topology-development-0.1','scope':'Complete local topology-only development candidate; no public release or held-out claim.',
            'run_name':run.name,'counts':data['counts'],'records_canonical_sha256':digest(canonical(data['records']).encode()),
            'source_input_sha256':data['analysis_manifest']['inputs'],'support_python_files':support,
            'support_source_sha256':{n:h for n,(p,h) in supports.items()},
            'support_source_note':'These transitive import dependencies match the earlier record_development_v0_1 release manifest. They were not additional entries in this run plan source_hashes.',
            'files_sha256':{str(p.relative_to(out)):digest(p.read_bytes()) for p in sorted(out.rglob('*')) if p.is_file()}}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        result={'verification':verify(out)}
        if replay:
            from verify_topology_release import replay as do_replay
            result['offline_replay']=do_replay(out)
        return result


def build(run,out,project_root,replay=True):
    """Commit a local candidate only after every requested verification passes."""
    out=Path(out)
    if out.exists():raise ValueError('refusing to overwrite topology release destination')
    # An incomplete run must not create even the destination's parent tree.
    run=Path(run)
    if not run.is_dir():raise ValueError('missing topology run')
    with stable_run(run):load_complete(run)
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.topology-release-',dir=out.parent) as temp:
        staged=Path(temp)/'candidate'
        result=_build_candidate(run,staged,project_root,replay=replay)
        if out.exists():raise ValueError('destination appeared during verification')
        staged.rename(out)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='results/topology_workload_development_v1')
    p.add_argument('--out',default='results/keep/topology_workload_development_v0_1')
    p.add_argument('--project-root',default='/Users/kostadindev/repos/joint-monitoring');a=p.parse_args()
    print(json.dumps(build(a.run,a.out,a.project_root),indent=2))
