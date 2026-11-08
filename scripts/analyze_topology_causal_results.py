"""Offline complete-only native development report; frozen analysis stays intact.

No provider construction, model calls, allocation changes or partial-run ranking.
All seeds average inside four content-case clusters. Intent twins stay separate.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import importlib
from pathlib import Path
from record_workflow import canonical

ARMS=('directed_path','exact_uniform_matched_random')
METHODS=('judge','structured_within','structured_pooled')


def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def mean(values):
    values=[v for v in values if v is not None]
    return math.fsum(values)/len(values) if values else None

def ratio(a,b):return a/b if b else None


def require_complete(plan,records,assignments,expectations,calls,metrics):
    required={key for row in assignments+expectations if row['feasible'] for key in row['view_keys']}
    if len(required)!=plan['unique_views'] or digest(sorted(required))!=plan['view_keys_sha256'] or set(calls)!=required:
        raise ValueError('complete exact planned view coverage required; no partial policy report')
    record_ids={r['id'] for r in records}
    if len(record_ids)!=len(records) or len(records)!=plan['episodes']:raise ValueError('record coverage mismatch')
    expected={(r['episode_id'],r['policy']):r for r in assignments+expectations}
    actual={(r['episode_id'],r['policy']):r for r in metrics}
    if len(expected)!=len(assignments)+len(expectations) or len(actual)!=len(metrics) or set(expected)!=set(actual):raise ValueError('complete exact per-record metrics required')
    if any(p['episode_id'] not in record_ids for p in expected.values()):raise ValueError('unbound assignment record')
    for key,p in expected.items():
        row=actual[key]
        if row['status']!=('scored' if p['feasible'] else 'infeasible') or row['metrics_available']!=p['feasible']:
            raise ValueError('unscored or improperly excluded allocation')
    from topology_causal_monitor import source_masks
    for r in records:source_masks(r)


def evaluated(row,method):
    if method=='judge':return row
    field={'structured_within':'record_matching_within_source_metrics','structured_pooled':'record_matching_pooled_source_metrics'}[method]
    return {**row,**row[field]}


def one_case(rows,method):
    """Pool seed numerators/denominators inside a case; do not inflate n cases."""
    values=[evaluated(r,method) for r in rows]
    def total(k):return math.fsum(r[k] for r in values)
    def avg(k):return mean([r[k] for r in values])
    truth=total('n_truth');benign=total('selected_known_benign');network=total('n_known_benign')
    tp=total('tp');fp=total('fp');ap=total('abstained_known_positive');ab=total('abstained_known_benign')
    result={'case_id':rows[0]['case_id'],'seed_rows':len(rows),'known_positive_coverage':ratio(total('selected_truth'),truth),
        'source_recall':ratio(tp,truth),'source_recall_upper':ratio(tp+ap,truth),
        'selected_known_benign_fpr':ratio(fp,benign),'selected_known_benign_fpr_upper':ratio(fp+ab,benign),
        'network_known_benign_fpr':ratio(fp,network),'network_known_benign_fpr_upper':ratio(fp+ab,network),
        'selected_unknown_fraction':ratio(total('selected_unknown'),total('k')),
        'network_unknown_fraction':ratio(total('n_unknown'),total('n_truth')+total('n_known_benign')+total('n_unknown')),
        'decision_unassessed_fraction':ratio(total('abstained_channels'),total('k')),
        'known_label_utility':avg('tp')-2*avg('fp'),
        'known_label_utility_lower':avg('tp')-2*(avg('fp')+avg('abstained_known_benign')),
        'known_label_utility_upper':avg('tp')+avg('abstained_known_positive')-2*avg('fp'),
        'unknown_label_pessimistic_utility':avg('tp')-2*(avg('fp')+avg('flagged_unknown')+avg('abstained_known_benign')+avg('abstained_unknown')),
        'unknown_label_optimistic_utility':avg('tp')+avg('flagged_unknown')+avg('abstained_known_positive')+avg('abstained_unknown')-2*avg('fp')}
    for key in ('tp','fp','flagged_unknown','abstained_known_positive','abstained_known_benign','abstained_unknown','source_any_recovery','source_all_recovery','source_both_recovery','source_any_recovery_upper','source_all_recovery_upper','source_both_recovery_upper'):
        result['mean_'+key]=avg(key)
    # Rule unassessed channels are a partial proof domain, not model failures.
    for key in ('input_tokens','output_tokens','replay_credits','requests','judge_calls','abstention_calls'):
        result['mean_'+key]=avg(key) if method=='judge' else 0.
    return result


def summary_rows(metrics,plans):
    buckets=defaultdict(list)
    fields=('environment','condition','selector','grouper','k','s','input_byte_allowance')
    for r in metrics:buckets[tuple(r[k] for k in fields)].append(r)
    output=[]
    for key,rows in sorted(buckets.items()):
        scope={plans[(r['episode_id'],r['policy'])]['comparison_scope'] for r in rows}
        if len(scope)!=1:raise ValueError('mixed policy access scope')
        case_ids=sorted({r['case_id'] for r in rows})
        for method in METHODS:
            common={**dict(zip(fields,key)),'method':method,'scope':next(iter(scope)),'case_clusters':len(case_ids),
                'assignment_rows':len(rows),'seed_values':sorted({r['seed'] for r in rows}),
                'infeasible_rows':sum(r['status']=='infeasible' for r in rows),'all_allocations_feasible':all(r['status']=='scored' for r in rows)}
            if not common['all_allocations_feasible']:
                output.append({**common,'metrics_defined':False,'per_case':[]});continue
            cases=[one_case([r for r in rows if r['case_id']==cid],method) for cid in case_ids]
            values={k:mean([r[k] for r in cases]) for k in cases[0] if k not in ('case_id','seed_rows')}
            output.append({**common,'metrics_defined':True,**values,'per_case':cases,
                'both_recovery_applicable_cases':sum(r['mean_source_both_recovery'] is not None for r in cases),
                'interpretation':'Equal content-case means after seeds average within case. Both applies only when exactly two certified false-origin channels exist; all/any are separate. Unknown labels are neither benign nor missed positives.'})
    return output


def grouping_contrasts(metrics,plans):
    selected=[r for r in metrics if r['selector']=='structural_path_pairs' and r['grouper'] in ARMS and r['k']==7 and r['s']==2]
    index={}
    for r in selected:
        key=(r['environment'],r['case_id'],r['condition'],r['seed'],r['grouper'])
        if key in index:raise ValueError('duplicate grouping comparison')
        index[key]=r
    bases={k[:-1] for k in index};paired=defaultdict(list)
    for base in sorted(bases):
        left,right=[index.get(base+(arm,)) for arm in ARMS]
        if left is None or right is None:raise ValueError('missing paired grouping arm')
        a,b=[plans[(r['episode_id'],r['policy'])] for r in (left,right)]
        if sorted(map(tuple,a['selected']))!=sorted(map(tuple,b['selected'])) or sorted(map(len,a['groups']))!=sorted(b['group_sizes']):raise ValueError('grouping comparison changes acquisition or group sizes')
        if left['status']!='scored' or right['status']!='scored':raise ValueError('infeasible isolated grouping comparison')
        if not math.isclose(left['judge_calls'],right['judge_calls'],abs_tol=1e-9):raise ValueError('grouping calls unmatched')
        paired[base[:3]].append((left,right))
    per_case=[]
    for (topology,cid,condition),pairs in sorted(paired.items()):
        for method in METHODS:
            a=one_case([p[0] for p in pairs],method);b=one_case([p[1] for p in pairs],method)
            fields=('source_recall','selected_known_benign_fpr','network_known_benign_fpr','mean_flagged_unknown','mean_source_any_recovery','mean_source_both_recovery','mean_source_all_recovery','known_label_utility','unknown_label_pessimistic_utility','mean_judge_calls','mean_requests','mean_replay_credits')
            delta={k:a[k]-b[k] if a[k] is not None and b[k] is not None else None for k in fields}
            per_case.append({'environment':topology,'case_id':cid,'condition':condition,'method':method,'path_minus_exact_random':delta})
    grouped=defaultdict(list)
    for row in per_case:grouped[(row['environment'],row['condition'],row['method'])].append(row)
    aggregate=[]
    for (topology,condition,method),rows in sorted(grouped.items()):
        deltas={}
        for key in rows[0]['path_minus_exact_random']:
            values=[r['path_minus_exact_random'][key] for r in rows if r['path_minus_exact_random'][key] is not None]
            deltas[key]={'mean':mean(values),'positive_cases':sum(v>1e-10 for v in values),'negative_cases':sum(v< -1e-10 for v in values),'zero_cases':sum(abs(v)<=1e-10 for v in values),'applicable_cases':len(values)}
        aggregate.append({'environment':topology,'condition':condition,'method':method,'deltas':deltas})
    return {'scope':'Fixed acquisition and group sizes; exact uniform partitions conditional on saved judge responses. Descriptive four-case signs, not a p-value. Twins stay separate.', 'per_case':per_case,'aggregate':aggregate}


def cost_audit(directory,ledger):
    """Unique raw HTTP requests, never summed across policies or intent twins."""
    rows=[]
    for path in sorted((Path(directory)/'requests').glob('*.json')):
        request=json.loads(path.read_text());rid=request['ledger_request_id'];entry=ledger['calls'][rid]
        if request['state']=='complete' and (entry['state']!='settled' or not math.isclose(entry['charge'],request['usage']['cost'],abs_tol=1e-12)):
            raise ValueError('raw completed request and actual ledger charge disagree')
        rows.append({'request_sha256':sha(path),'ledger_request_id':rid,'state':entry['state'],'charge':entry['charge'],'batch':entry['batch']})
    if len({r['ledger_request_id'] for r in rows})!=len(rows):raise ValueError('duplicate request charge ID')
    return {'unique_http_requests':len(rows),'actual_charged_or_reserved':math.fsum(r['charge'] for r in rows),
        'settled_cost':math.fsum(r['charge'] for r in rows if r['state']=='settled'),
        'uncertain_or_reserved_cost':math.fsum(r['charge'] for r in rows if r['state']!='settled'),'requests':rows}


def reliability(calls,cap):
    groups=defaultdict(list)
    for c in calls.values():groups[len(c['edges'])].append(c)
    output=[]
    for size,rows in sorted(groups.items()):
        attempts=[a for c in rows for a in c['attempts']]
        output.append({'group_size':size,'unique_views':len(rows),'abstentions':sum('abstention' in c for c in rows),
            'format_retries':sum(c['request_count']-1 for c in rows),'attempts':len(attempts),
            'output_at_cap_attempts':sum(a['usage']['output_tokens']>=cap for a in attempts),
            'input_tokens':sum(c['usage']['input_tokens'] for c in rows),'output_tokens':sum(c['usage']['output_tokens'] for c in rows),
            'logical_replay_cost':math.fsum(c['usage']['cost'] for c in rows)})
    return output


def report(data):
    def fmt(v):return '—' if v is None else f'{100*v:.1f}%'
    lines=['# Complete native development comparison','',
        'All feasible planned views and every allocation metric passed the completeness gate and raw-call validation. The separate native judge uses the same uniform 2,048-token cap for all its arms; the earlier main workload used 1,024 and is not a matched judge comparison. Four task cases are the clusters; seeds average within case. Topology variants and observational honest-error twins are correlated. This is development evidence, not a held-out winner.',
        '', '## Main graph-only comparison at k=7, s=2','',
        'The table uses combined (AB) worlds for source recovery and paired clean worlds for false alarms. Unknown-label channels are excluded from known-benign denominators and reported separately. Both/all count certified origin channels, not necessarily distinct actors. Rows are not ranked across different constraints.', '',
        '| Topology | Selector/grouping | Source coverage | Recall | Any | Both | All | AB benign FPR | Clean benign FPR | Unknown selected | Calls |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    rows=data['summary'];lookup={(r['environment'],r['condition'],r['selector'],r['grouper'],r['k'],r['s'],r['method']):r for r in rows}
    for r in rows:
        if r['condition']!='combined' or r['method']!='judge' or r['k']!=7 or r['s']!=2 or not r['scope'].startswith('prospective_') or not r['metrics_defined']:continue
        clean=lookup[(r['environment'],'clean',r['selector'],r['grouper'],7,2,'judge')]
        cells=[r['environment'],r['selector']+'/'+r['grouper']]+[fmt(r[k]) for k in ('known_positive_coverage','source_recall','mean_source_any_recovery','mean_source_both_recovery','mean_source_all_recovery','selected_known_benign_fpr')]+[fmt(clean['selected_known_benign_fpr']),fmt(r['selected_unknown_fraction']),f"{r['mean_judge_calls']:.2f}"]
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', '## Structured controls and larger-context references','',
        'The structured source checker is deliberately partial: it needs a matching protected receipt and acquired incoming-neighborhood evidence. Derived computations, unsupported outputs and prose remain unassessed. Its unassessed channels are not correct negative predictions. The pooled version may join all seven acquired channels and therefore has a different cross-group processing assumption.', '',
        '| Topology | Arm | k/s | Evaluator | Recall | Both | Benign FPR | Unassessed selected |',
        '|---|---|---|---|---:|---:|---:|---:|']
    for r in rows:
        if r['condition']!='combined' or not r['metrics_defined']:continue
        show=(r['selector']=='structural_path_pairs' and r['grouper']=='directed_path' and r['method']!='judge') or (r['scope']=='full_network_reference') or (r['selector']=='structural_path_pairs' and r['grouper'] in ('singleton','full_selected') and r['method']=='judge')
        if show:lines.append('| '+' | '.join([r['environment'],r['selector']+'/'+r['grouper'],f"{r['k']}/{r['s']}",r['method'],fmt(r['source_recall']),fmt(r['mean_source_both_recovery']),fmt(r['selected_known_benign_fpr']),fmt(r['decision_unassessed_fraction'])])+' |')
    lines += ['', 'Privileged potential-location diagnostics are stored in `privileged_diagnostics.json` separately. They are not graph-only prospective competitors. All conditions, seeds, bounds and within-case values are retained in `summary.json`.', '',
        '## Isolated grouping contrast','',
        'The path-minus-exact-random contrasts fix the same selected edges, group sizes and logical call count. Nonlinear any/both/all probabilities come from complete-partition enumeration in the frozen analyzer. Positive recall deltas favor paths; negative false-alarm deltas favor paths. Four-case signs are descriptive and have no general-population p-value.', '',
        '| Topology | Condition | Recall delta | Known-benign FPR delta | Both delta |', '|---|---|---:|---:|---:|']
    for r in data['grouping']['aggregate']:
        if r['method']=='judge' and r['condition'] in ('clean','combined'):
            d=r['deltas'];lines.append('| '+' | '.join([r['environment'],r['condition'],fmt(d['source_recall']['mean']),fmt(d['selected_known_benign_fpr']['mean']),fmt(d['mean_source_both_recovery']['mean'])])+' |')
    costs=data['costs'];lines += ['', '## Actual resource use','',
        f"Judge acquisition: {costs['monitor']['unique_http_requests']} unique HTTP requests; {costs['monitor']['settled_cost']:.8f} settled credits; {costs['monitor']['uncertain_or_reserved_cost']:.8f} uncertain/reserved credits. Native generation: {costs['generation']['settled_cost']:.8f} settled credits. Policy-level cost rows are expected replay costs and must not be summed as actual spending.", '',
        'Known-label utility excludes uncertain truth; the JSON also includes conservative unknown-label utility ranges and bounded-abstention resolution ranges. These are not confidence intervals. Source detection is distinct from the strict/alias action activation audit: unknown primary action labels and confessional intervention cues remain limitations. No result here establishes stealth, intent discrimination or LLM necessity.', '']
    return '\n'.join(lines)


def analyze_complete(directory,output,*,ledger_path=None,generation_directory=None):
    from topology_causal_monitor import analyze,plan_library
    from run_topology_causal_monitor import load_calls
    directory=Path(directory);output=Path(output)
    # Check completion artifacts before reading/aggregating any outcome rows.
    plan=json.loads((directory/'plan.json').read_text());coverage=json.loads((directory/'coverage.json').read_text())
    if coverage.get('complete') is not True or coverage.get('completed_views')!=plan['unique_views'] or not (directory/'per_record_metrics.json').exists():raise ValueError('run incomplete; no partial native comparison')
    for name in ('topology_causal_monitor','run_topology_causal_monitor','topology_causal_annotations'):
        relative='scripts/'+name+'.py'
        if sha(importlib.import_module(name).__file__)!=plan['source_sha256'][relative]:raise ValueError('frozen analysis dependency changed: '+name)
    records=json.loads((directory/'records.json').read_text());assignments=json.loads((directory/'assignments.json').read_text());expectations=json.loads((directory/'exact_random_plans.json').read_text());metrics=json.loads((directory/'per_record_metrics.json').read_text())
    for name,value in [('records',records),('assignments',assignments),('expectations',expectations)]:
        if digest(value)!=plan[name+'_sha256']:raise ValueError('frozen '+name+' hash mismatch')
    diagnostics=json.loads((directory/'diagnostic_locations.json').read_text())
    jobs,a,e=plan_library(records,system=plan['system'],judge_config=plan['judge_config'],diagnostic_locations=diagnostics)
    if canonical(a)!=canonical(assignments) or canonical(e)!=canonical(expectations):raise ValueError('frozen planner replay mismatch')
    calls=load_calls(directory,jobs);require_complete(plan,records,assignments,expectations,calls,metrics)
    if canonical(analyze(records,assignments,expectations,calls))!=canonical(metrics):raise ValueError('saved native metrics differ from exact replay')
    plans={(r['episode_id'],r['policy']):r for r in assignments+expectations};summary=summary_rows(metrics,plans);grouping=grouping_contrasts(metrics,plans)
    ledger=json.loads(Path(ledger_path or plan['ledger']).read_text());costs={'monitor':cost_audit(directory,ledger),'generation':cost_audit(generation_directory or plan['generation']['directory'],ledger)}
    data={'scope':'Complete native development only; four content clusters, correlated topology/seed/intent twins, no partial ranking.',
        'summary':summary,'grouping':grouping,'costs':costs,'reliability':reliability(calls,plan['judge_config']['max_output_tokens'])}
    output.mkdir(parents=True,exist_ok=True)
    artifacts={'summary.json':{'scope':data['scope'],'rows':summary},'grouping_contrasts.json':grouping,'actual_costs.json':costs,'unique_view_reliability.json':data['reliability'],
        'privileged_diagnostics.json':[r for r in summary if r['scope']=='privileged_location_diagnostic']}
    for name,value in artifacts.items():(output/name).write_text(json.dumps(value,indent=2)+'\n')
    (output/'report.md').write_text(report(data))
    manifest={'scope':data['scope'],'complete_planned_views':len(calls),'metric_rows':len(metrics),'case_clusters':plan['case_clusters'],
        'inputs_sha256':{name:sha(directory/name) for name in ('plan.json','coverage.json','records.json','assignments.json','exact_random_plans.json','per_record_metrics.json','activation.json','secondary_alias_actions.json')},
        'analysis_source_sha256':sha(__file__),'frozen_source_sha256':plan['source_sha256'],
        'outputs_sha256':{name:sha(output/name) for name in list(artifacts)+['report.md']}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest


def from_release(candidate,output):
    """Verify the immutable local candidate, then replay with only bundled data.

    No original staging directory or external ledger is consulted. The candidate
    plans are never rewritten; only physical generation/ledger lookup is relocated.
    """
    root=Path(candidate).resolve();output=Path(output).resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    for name,want in manifest['files_sha256'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or sha(root/relative)!=want:raise ValueError('candidate artifact hash/path mismatch: '+name)
    env={'PATH':os.environ.get('PATH',''),'LANG':os.environ.get('LANG','C.UTF-8'),'PYTHONDONTWRITEBYTECODE':'1'}
    verifier=root/'tooling/build_topology_causal_release.py'
    verified=subprocess.run([sys.executable,str(verifier),'--verify',str(root)],capture_output=True,text=True,env=env)
    if verified.returncode:raise ValueError('bundled candidate verifier rejected replay: '+verified.stderr[-3000:])
    if json.loads(verified.stdout).get('verified') is not True:raise ValueError('candidate verifier did not confirm validity')
    with tempfile.TemporaryDirectory(prefix='native-review-release-') as temp:
        runtime=Path(temp)
        for folder in (root/'private/generation',root/'audit/monitor'):
            plan=json.loads((folder/'plan.json').read_text())
            for name,want in plan['source_sha256'].items():
                relative=Path(name)
                if relative.is_absolute() or '..' in relative.parts:raise ValueError('unsafe frozen source path')
                source=folder/'source'/relative;destination=runtime/relative
                if sha(source)!=want or destination.exists() and sha(destination)!=want:raise ValueError('conflicting frozen runtime source')
                destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
        code=("import sys,runpy,urllib.request;sys.path[:0]=sys.argv[1:3];"
              "urllib.request.urlopen=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('offline replay forbids network'));"
              "import jmon.llm;jmon.llm.make_provider=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('offline replay forbids provider'));"
              "sys.argv=sys.argv[3:];runpy.run_path(sys.argv[0],run_name='__main__')")
        command=[sys.executable,'-c',code,str(runtime/'scripts'),str(runtime/'src'),str(Path(__file__).resolve()),
                 '--run',str(root/'audit/monitor'),'--out',str(output),'--ledger-snapshot',str(root/'audit/ledger_snapshot.json'),
                 '--generation',str(root/'private/generation')]
        completed=subprocess.run(command,capture_output=True,text=True,env=env)
        if completed.returncode:raise ValueError('offline report replay failed: '+completed.stderr[-3000:])
    return json.loads((output/'manifest.json').read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);p.add_argument('--from-release',action='store_true')
    p.add_argument('--ledger-snapshot');p.add_argument('--generation');a=p.parse_args()
    result=from_release(a.run,a.out) if a.from_release else analyze_complete(a.run,a.out,ledger_path=a.ledger_snapshot,generation_directory=a.generation)
    print(json.dumps({k:result[k] for k in ('complete_planned_views','metric_rows','case_clusters','analysis_source_sha256')},indent=2))
if __name__=='__main__':main()
