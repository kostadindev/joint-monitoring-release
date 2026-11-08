"""Build a complete, local development release; never publishes or calls models.

Fail before creating output when either experiment is incomplete. Explicit file
allowlists exclude credentials, caches, MESA and downloaded external artifacts.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack,contextmanager
import copy
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from verify_workload_release import canonical,digest,read_rows,validate_dataset,verify,MESSAGE_FIELDS

REQUIRED={
 'diagnostic':('plan.json','summary.json','analysis_coverage.json','adoption_decision.json','prompt_comparison.json','analysis_source_manifest.json'),
 'workload':('plan.json','summary.json','coverage.json','primary_contrast.json','development_ranking.json','allocation_oracle.json','research_spend.json')}
ROWS={
 'diagnostic':('assignments.jsonl','metrics.jsonl','rule_metrics.jsonl','prompt_pairs.jsonl'),
 'workload':('assignments.jsonl','exact_random_plans.jsonl','metrics.jsonl')}
CARD='''# Shared-channel workload development release candidate v0.2

Local candidate only; this command does not publish. No held-out claims. The
project code license is MIT; dataset redistribution licensing remains undeclared.
No MESA logs, external trace raw data, provider caches or credentials are included.

The diagnostic and workload are deterministic message interventions with twelve
original target-case clusters each. Histories, routing, rendering, causes and
allocation seeds are repeated conditions, not independent attacks. Honest-error
and deliberate-false worlds share exact public observations and model judgments.
Scores identify factual false origins, not malicious intent or compromised actors.

## Public observations and evaluation material

Each dataset has observations.jsonl.gz containing only opaque episode IDs, public
workflow, graph, and delivered-message fields. Evaluation labels, target-case IDs,
conditions, task hashes, hidden message annotations, group assignments and outcome
metrics are isolated under evaluation/. Everything under evaluation/ and audit/ is
for scoring/reproduction, never prospective policy input. Selected channels retain
all their messages. A protected reference consumes an inspected edge.

All raw model outputs, bounded retry attempts, abstentions, exact public prompts,
usage, and phase outcomes are in audit/. Shared public views are scored once; their
research cost must not be added repeatedly across policy assignments. Standalone
policy replay cost is a different quantity. No server-side model revision is available.

## Constraints and exact random baseline

Ordinary groups form a disjoint partition within k inspected edges and cap s.
24 KB is a declared operating input allowance, not a model context-window claim.
Explicit relaxed full-context references use 60 KB and are not matched-cost winners.
Infeasible assignments remain with undefined LLM outcomes; rule matching can remain
feasible. Valid scores with invalid auxiliary citations are retained without retry.

Exact random records are expectations over uniform matched-size partitions, not
one policy executing every candidate group. Candidate group inclusion weights sum
to the expected number of calls; each acquired edge has marginal inclusion one.
For k=7 and sizes [1,2,2,2], 28 scored subsets represent the expectation of four
calls over 105 partitions. Metrics and replay costs use those weights. Precision
is a ratio of expected counts, not the mean of partition precision. Acquisition
seeds and correlated worlds are averaged within original case clusters.

## Reproduction and verification

Run `python tooling/verify_workload_release.py RELEASE_DIRECTORY` (standard library
only). It checks all file hashes, exact public prompt reconstruction, shared-view
coverage, actual retry/usage accounting, ordinary acquisition/group/input budgets,
exact random weights and marginals, one-to-one metric coverage, source counts and
within-group/pooled deterministic record-matching counts. It checks frozen source
hashes. It does not certify correctness of every free-text rationale or label.

Add `--replay` to re-run the frozen workload metrics, summaries, primary contrast,
development ranking and allocation oracle, and compare exact outputs. Replay needs
the project dependencies (for example the repository Python environment), uses only
bundled observations/labels/calls, writes temporary files, and makes no model calls.
Research-spend accounting is preserved separately; replay does not invent a ledger.

The frozen experiment sources are under each dataset's source/. The diagnostic's
post-run analysis snapshot is separate from its run snapshot. Shared project Python
modules and packaging metadata are included under support/ for those historical
scripts' imports. Use the bundled observations plus evaluation labels to reconstruct
full records for offline analysis; the verification implementation demonstrates the
join. Public messages reproduce judge prompts without using any evaluation labels.
Earlier v0.1 original-workflow data remain a separate candidate; v0.2 does not
retroactively replace its frozen analysis.
'''


def opaque(kind,identity):return 'ep_'+digest((kind+'|'+identity).encode())[:20]


def split_record(kind,rec):
    """Only fields actually available to the monitor are public."""
    public={'id':opaque(kind,rec['id']),'workflow':rec['workflow'],'graph':copy.deepcopy(rec['graph']),
            'messages':[{k:copy.deepcopy(m[k]) for k in MESSAGE_FIELDS} for m in rec['messages']]}
    label={'id':public['id'],'original_id':rec['id'],
           'metadata':{k:copy.deepcopy(v) for k,v in rec.items() if k not in ('id','workflow','graph','messages')},
           'message_labels':{m['id']:copy.deepcopy(m['hidden']) for m in rec['messages']}}
    return public,label


def remap(value,ids):
    if isinstance(value,dict):return {k:(ids[v] if k in ('id','episode_id') and isinstance(v,str) and v in ids else remap(v,ids)) for k,v in value.items()}
    if isinstance(value,list):return [remap(v,ids) for v in value]
    return value


def write_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as stream,gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as zipped:
        for row in rows:zipped.write((canonical(row)+'\n').encode())


@contextmanager
def stable_source(root):
    # Prevent a running dispatcher from mutating inputs while creating a release.
    lock_path=root/'.run.lock'
    if lock_path.exists():
        with lock_path.open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ValueError('experiment still running: '+str(root))
            yield
    else:yield


def load_complete(kind,root):
    for name in REQUIRED[kind]+ROWS[kind]+('episodes.jsonl','views.jsonl','judge_calls.jsonl'):
        if not (root/name).is_file():raise ValueError('missing completed artifact: '+str(root/name))
    plan=json.loads((root/'plan.json').read_text());records=read_rows(root/'episodes.jsonl')
    assignments=read_rows(root/'assignments.jsonl');expectations=read_rows(root/'exact_random_plans.jsonl') if kind=='workload' else []
    calls=read_rows(root/'views.jsonl');metrics=read_rows(root/'metrics.jsonl')
    if kind=='workload':
        if digest(canonical(assignments).encode())!=plan['assignments_sha256'] or digest(canonical(expectations).encode())!=plan['expected_random_sha256']:raise ValueError('frozen assignment hashes changed')
        if digest(canonical(sorted(c['view_key'] for c in calls)).encode())!=plan['view_keys_sha256']:raise ValueError('frozen view set changed')
        spend=json.loads((root/'research_spend.json').read_text())
        if spend['active_reservations']!=0:raise ValueError('outstanding workload reservations')
        if any(json.loads(p.read_text())['status']=='running' for p in root.glob('phase_*.json')):raise ValueError('unterminated workload phase')
    elif digest(canonical(records).encode())!=plan['records_sha256']:raise ValueError('frozen diagnostic records changed')
    for name,want in plan['source_hashes'].items():
        if Path(name).name!=name or digest((root/'source'/name).read_bytes())!=want:raise ValueError('frozen source mismatch: '+name)
    counts=validate_dataset(kind,records,assignments,expectations,calls,metrics,plan)
    return {'records':records,'assignments':assignments,'expectations':expectations,'calls':calls,'metrics':metrics,'plan':plan,'counts':counts}


def build(diagnostic,workload,out,project_root):
    sources={'diagnostic':Path(diagnostic),'workload':Path(workload)};out=Path(out);project_root=Path(project_root)
    if out.exists():raise ValueError('refusing to overwrite release destination')
    with ExitStack() as stack:
        for source in sources.values():stack.enter_context(stable_source(source))
        loaded={kind:load_complete(kind,source) for kind,source in sources.items()}
        if not (project_root/'src/jmon').is_dir() or not (project_root/'pyproject.toml').is_file():raise ValueError('project support source unavailable')
        # Prevalidation completed; an interrupted build remains visibly incomplete
        # without a manifest and is never mistaken for a successful release.
        out.mkdir(parents=True);input_hashes={}
        for kind,source in sources.items():
            data=loaded[kind];base=out/kind;ids={r['id']:opaque(kind,r['id']) for r in data['records']}
            pairs=[split_record(kind,r) for r in data['records']]
            write_rows(base/'observations.jsonl.gz',(p for p,l in pairs));write_rows(base/'evaluation/labels.jsonl.gz',(l for p,l in pairs))
            for name in ROWS[kind]:write_rows(base/'evaluation'/(name+'.gz'),(remap(r,ids) for r in read_rows(source/name)))
            for name in REQUIRED[kind]:
                (base/'evaluation').mkdir(exist_ok=True);(base/'evaluation'/name).write_text(json.dumps(remap(json.loads((source/name).read_text()),ids),indent=2)+'\n')
            for name in ('views.jsonl','judge_calls.jsonl'):
                write_rows(base/'audit'/(name+'.gz'),(remap(r,ids) for r in read_rows(source/name)))
            metadata=[p for pattern in ('meta*.json','timeout_recovery*.json','phase_*.json') for p in source.glob(pattern)]
            for p in metadata:
                (base/'audit'/p.name).write_text(json.dumps(remap(json.loads(p.read_text()),ids),indent=2)+'\n')
            for name in data['plan']['source_hashes']:
                (base/'source').mkdir(exist_ok=True);shutil.copyfile(source/'source'/name,base/'source'/name)
            if kind=='diagnostic':
                analysis=json.loads((source/'analysis_source_manifest.json').read_text())
                for name,want in analysis.items():
                    original=source/'analysis_source'/name
                    if Path(name).name!=name or digest(original.read_bytes())!=want:raise ValueError('diagnostic analysis snapshot changed')
                    target=base/'analysis_source'/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(original,target)
            input_hashes[kind]={name:digest((source/name).read_bytes()) for name in ('episodes.jsonl','assignments.jsonl','views.jsonl','metrics.jsonl','plan.json')}
        supplemental=sources['workload']/'supplemental_contrast_audit.json'
        if supplemental.exists():
            extra=out/'workload/supplemental';extra.mkdir()
            shutil.copyfile(supplemental,extra/supplemental.name)
            stage=Path(__file__).resolve().parents[1]
            for rel in ('scripts/audit_workload_contrasts.py','research/workload_contrast_addendum.md'):
                source=stage/rel
                if not source.is_file():raise ValueError('missing supplemental source/protocol: '+rel)
                shutil.copyfile(source,extra/source.name)
            (extra/'README.md').write_text('Post-launch supplemental contrasts; separate from the frozen primary analysis.\n')
        for original in sorted((project_root/'src/jmon').rglob('*.py')):
            if original.is_symlink():raise ValueError('symlink in support source')
            target=out/'support'/original.relative_to(project_root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,target)
        shutil.copyfile(project_root/'pyproject.toml',out/'support/pyproject.toml')
        for name in ('build_workload_release.py','verify_workload_release.py'):
            target=out/'tooling'/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(Path(__file__).parent/name,target)
        (out/'README.md').write_text(CARD)
        manifest={'version':'0.2','scope':'Complete local development candidate, not a public or held-out release.',
            'post_launch_supplemental_contrasts':supplemental.exists(),
            'counts':{kind:data['counts'] for kind,data in loaded.items()},
            'records_canonical_sha256':{kind:digest(canonical(data['records']).encode()) for kind,data in loaded.items()},'source_input_sha256':input_hashes,
            'files_sha256':{str(p.relative_to(out)):digest(p.read_bytes()) for p in sorted(out.rglob('*')) if p.is_file()}}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        return verify(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--diagnostic',default='results/record_audit_diagnostics_v1')
    p.add_argument('--workload',default='results/record_workload_development_v1')
    p.add_argument('--out',default='results/keep/record_workload_development_v0_2')
    p.add_argument('--project-root',default=str(Path(__file__).resolve().parents[1]));a=p.parse_args()
    print(json.dumps(build(a.diagnostic,a.workload,a.out,a.project_root),indent=2))
