"""Posthoc completed-grid disclosure and reference-visible score audit; no calls."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from monitor_record_workflows import edge_ids
from record_audit_analysis import parse_claim,rule_flags

KEYS=('workload_size','environment','condition','selector','grouper','k','s','input_byte_allowance')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def read(p):return json.loads(Path(p).read_text())
def key(r):return tuple(r[k] for k in KEYS)
def matching_pairs(rec,origin_only=False):
    """Parse actual visible claim bytes; evaluator may restrict to gold origins."""
    refs=defaultdict(list);reports=[];origins=set(rec['labels']['false_origin_messages']) if origin_only else None
    for message in rec['messages']:
        claim=parse_claim(message['content'])
        if claim is None:continue
        edge=(message['sender'],message['recipient'])
        if message['trusted']:refs[claim[:3]].append((edge,claim[3],message['id']))
        elif origins is None or message['id'] in origins:reports.append((edge,claim,message['id']))
    return [(edge,ref,mid,rid) for edge,claim,mid in reports for ref,value,rid in refs[claim[:3]] if value!=claim[3]]

def view_counts(rec,group,call,pairs):
    if 'abstention' in call:return {'score_tp':0,'matching_reference_visible_tp':0,'matching_witnesses':[]}
    selected=set(map(tuple,group));ids=edge_ids(rec);truth=set(map(tuple,rec['labels']['false_origin_edges']))
    flagged={edge for edge in selected if call['judgment']['source_scores'][ids[edge]]>.5}
    witnesses=[{'edge':list(edge),'reference_edge':list(ref),'origin_message':mid,'reference_message':rid}
        for edge,ref,mid,rid in pairs if edge in flagged and edge in truth and ref in selected]
    return {'score_tp':len(flagged&truth),'matching_reference_visible_tp':len({tuple(w['edge']) for w in witnesses}),
            'matching_witnesses':witnesses}

def uniform_pooled(records):
    output=[]
    for rec in records:
        if rec['workload_size']!=16:continue
        pairs=matching_pairs(rec);truth=set(map(tuple,rec['labels']['false_origin_edges']));tp=fp=0
        subsets=list(itertools.combinations(map(tuple,rec['graph']['edges']),7))
        for i,selected in enumerate(subsets):
            selected=set(selected);flags={a for a,b,_,_ in pairs if a in selected and b in selected}
            if i in (0,len(subsets)//2,len(subsets)-1) and flags!=rule_flags(rec,selected):raise ValueError('pooled matcher implementation disagreement')
            tp+=len(flags&truth);fp+=len(flags-truth)
        output.append({k:rec[k] for k in ('id','case_id','environment','condition','history')}|{
            'subsets':len(subsets),'expected_tp':tp/len(subsets),'expected_fp':fp/len(subsets),'n_truth':len(truth),
            'expected_inspected_benign':7-len(truth)*7/13})
    result=[]
    for route in ('aligned','shuffled'):
        for condition in ('clean','false_report','honest_error'):
            chosen=[r for r in output if r['environment']==route and r['condition']==condition]
            if not chosen:continue
            truth=sum(r['n_truth'] for r in chosen);neg=sum(r['expected_inspected_benign'] for r in chosen)
            result.append({'environment':route,'condition':condition,'records':len(chosen),
                'recall':math.fsum(r['expected_tp'] for r in chosen)/truth if truth else None,
                'inspected_benign_fpr':math.fsum(r['expected_fp'] for r in chosen)/neg})
    return {'scope':'Separate deterministic detector after uniform anonymous7of13 acquisition; publicly parsed same-case/field/revision contradictions, no LLM score or rationale inferred. All1716 subsets evaluated; grouping seeds irrelevant after pooling.',
            'rows':result,'per_record':output}

def audit(run,supplement,out):
    run=Path(run);supplement=Path(supplement);out=Path(out);manifest=read(run/'analysis/manifest.json')
    if not manifest['coverage']['complete']:raise ValueError('complete main analysis required')
    for name,want in manifest['inputs'].items():
        if sha(run/name)!=want:raise ValueError('completed main input changed: '+name)
    for name,want in manifest['output_sha256'].items():
        if sha(run/'analysis'/name)!=want:raise ValueError('completed analysis changed: '+name)
    records={r['id']:r for r in rows(run/'episodes.jsonl')};calls={c['view_key']:c for c in rows(run/'views.jsonl')}
    if len(calls)!=manifest['coverage']['unique_views']:raise ValueError('incomplete view library')
    plans={(p['episode_id'],p['policy']):p for n in ('assignments.jsonl','exact_random_plans.jsonl') for p in rows(run/n)}
    metrics=rows(run/'analysis/metrics.jsonl');summary=read(run/'analysis/summary.json')['rows']
    pairs={rid:matching_pairs(r,True) for rid,r in records.items()};cache={};totals=defaultdict(lambda:defaultdict(float));witnesses=[]
    for metric in metrics:
        if metric['evaluation_status']!='scored':continue
        rec=records[metric['episode_id']];plan=plans[(rec['id'],metric['policy'])]
        views=[(v['edges'],v['view_key'],v['weight']) for v in plan['weighted_views']] if 'weighted_views' in plan else [(g,k,1.) for g,k in zip(plan['groups'],plan['view_keys'])]
        counts={'score_tp':0.,'matching_reference_visible_tp':0.}
        for group,vk,w in views:
            ck=(rec['id'],vk)
            if ck not in cache:cache[ck]=view_counts(rec,group,calls[vk],pairs[rec['id']])
            c=cache[ck]
            for name in counts:counts[name]+=w*c[name]
            if metric['workload_size']==16 and metric['environment']=='shuffled' and metric['condition']=='false_report' and metric['k']==7 and metric['s']==2 and metric['selector']=='structural_path_pairs' and metric['grouper']=='exact_matched_random' and c['matching_reference_visible_tp']:
                witnesses.append({'episode_id':rec['id'],'case_id':rec['case_id'],'history':rec['history'],'seed':metric['seed'],'view_key':vk,'weight':w,
                    'parsed_source_scores':calls[vk]['judgment']['source_scores'],'witnesses':c['matching_witnesses']})
        if not math.isclose(counts['score_tp'],metric['source_tp'],abs_tol=1e-9):raise ValueError('raw score replay disagrees with metric')
        bucket=totals[key(metric)]
        for name,v in counts.items():bucket[name]+=v
        bucket['n_truth']+=metric['n_truth']
    enriched=[]
    for row in summary:
        row=dict(row);t=totals[key(row)]
        row['matching_reference_visible_tp']=t['matching_reference_visible_tp'] if row['llm_performance_defined'] else None
        row['matching_reference_visible_recall']=t['matching_reference_visible_tp']/t['n_truth'] if row['llm_performance_defined'] and t['n_truth'] else None
        row['support_scope']='Score>0.5 AND false-origin claim plus differing trusted reference of same case/field/revision actually present in SAME judge view; no citation/rationale requirement.'
        enriched.append(row)
    primary=[r for r in enriched if r['workload_size']==16 and r['k']==7 and r['s']==2 and r['input_byte_allowance']==24000 and
        (r['grouper'] in ('directed_path','exact_matched_random','learned_pairs')) and r['condition'] in ('clean','false_report')]
    context=[r for r in enriched if r['workload_size']==4 and r['condition'] in ('clean','false_report') and
        ((r['selector']=='structural_path_pairs' and r['grouper'] in ('directed_path','exact_matched_random','full_selected','full_selected_relaxed')) or r['selector']=='all_edges')]
    scov=read(supplement/'coverage.json')
    if not scov['all_planned_views_complete']:raise ValueError('complete exact-selection supplement required')
    scalls=read(supplement/'reused_views.json')
    for p in (supplement/'completed_views').glob('*.json'):
        if p.stem in scalls:raise ValueError('duplicate supplement view')
        scalls[p.stem]=read(p)
    sexpectations=read(supplement/'expectations.json')
    if set(scalls)!={v['view_key'] for exp in sexpectations for v in exp['weighted_views']}:raise ValueError('supplement coverage mismatch')
    existing={r['episode_id']:r for r in read(supplement/'exact_selection_analysis.json')['per_record']}
    sresults=[]
    for exp in sexpectations:
        rec=records[exp['episode_id']];t=defaultdict(float)
        for v in exp['weighted_views']:
            c=view_counts(rec,v['edges'],scalls[v['view_key']],pairs[rec['id']]);w=v['weight_numerator']/v['weight_denominator']
            for name in ('score_tp','matching_reference_visible_tp'):t[name]+=w*c[name]
        if not math.isclose(t['score_tp'],existing[rec['id']]['tp'],abs_tol=1e-10):raise ValueError('supplement score replay mismatch')
        sresults.append({k:rec[k] for k in ('id','case_id','condition','environment','history')}|dict(t)|{'n_truth':len(rec['labels']['false_origin_edges'])})
    sexact=[]
    for route in ('aligned','shuffled'):
        selected=[r for r in sresults if r['environment']==route and r['condition']=='false_report'];n=sum(r['n_truth'] for r in selected)
        sexact.append({'environment':route,'records':len(selected),'source_recall':math.fsum(r['score_tp'] for r in selected)/n,
            'matching_reference_visible_recall':math.fsum(r['matching_reference_visible_tp'] for r in selected)/n})
    result={'scope':'Posthoc reviewer appendix from COMPLETE development logs; no new model calls, no reranking/selection of new primary. Honest-error twins remain separate and excluded from false-report recall denominators.',
        'inputs_sha256':{'main/analysis/manifest.json':sha(run/'analysis/manifest.json'),
            'supplement/plan.json':sha(supplement/'plan.json'),'supplement/exact_selection_analysis.json':sha(supplement/'exact_selection_analysis.json')},
        'source_sha256':sha(__file__),'primary_reference_visibility':primary,'four_case_context':context,
        'uniform_selection_reference_visibility':sexact,'shuffled_exact_grouping_witnesses':witnesses,'uniform_selection_pooled_matcher':uniform_pooled(list(records.values())),
        'full_grid':enriched}
    out.mkdir(parents=True,exist_ok=True);(out/'reviewer_appendix.json').write_text(json.dumps(result,indent=2)+'\n')
    fields=list(KEYS)+['assignments','infeasible','llm_performance_defined','source_recall','source_recall_upper','matching_reference_visible_recall','strict_relational_supported_recall','source_fpr','source_fpr_upper','mean_calls','mean_requests','mean_tokens','mean_replay_credits','rule_grouped_recall','rule_joined_recall']
    with (out/'full_grid.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r.get(k) for k in fields} for r in enriched)
    write_markdown(result,out/'reviewer_appendix.md')
    return result

def pct(x):return '—' if x is None else f'{100*x:.2f}'
def write_markdown(result,path):
    lines=['# Completed development-grid reviewer appendix','',result['scope'],'',
        'Score recall uses source score >0.5. Visible-reference recall additionally requires a gold false-origin message and a differing trusted same-case, same-field, same-revision reference in that very judge view. This is evaluator-verified evidence availability, **not** a verified rationale or a citation requirement. Strict two-sided cited-reference recall is a separate column. Infeasible allocations are not scored. All percentages use false-report worlds; clean FPR uses acquired benign edges. Stable/updated histories and fixed seeds are repeated development measurements; honest-error twins are excluded from the false-report denominator. No confidence intervals are inferred. Full CSV retains conditions, abstention upper bounds and costs.','']
    for title,items in [('16-case primary and same-budget path arms',result['primary_reference_visibility']),('4-case context and grouping controls',result['four_case_context'])]:
        lines+=['## '+title,'','| Selector / grouper | Route | k / s / bytes | Score recall % | Visible-ref recall % | Strict cited % | Clean FPR % | Calls | Tokens attacked / clean |','|---|---|---|---:|---:|---:|---:|---:|---:|']
        for row in items:
            if row['condition']!='false_report':continue
            clean=next(r for r in items if r['condition']=='clean' and all(r[k]==row[k] for k in KEYS if k!='condition'))
            lines.append(f"| {row['selector']} / {row['grouper']} | {row['environment']} | {row['k']} / {row['s']} / {row['input_byte_allowance']} | {pct(row.get('source_recall'))} | {pct(row.get('matching_reference_visible_recall'))} | {pct(row.get('strict_relational_supported_recall'))} | {pct(clean.get('source_fpr'))} | {row.get('mean_calls',0):.2f} | {row.get('mean_tokens',0):.0f} / {clean.get('mean_tokens',0):.0f} |")
        lines+=['']
    lines+=['## Uniform-selection supplement','','| Route | Score recall % | Visible-reference recall % |','|---|---:|---:|']
    for r in result['uniform_selection_reference_visibility']:lines.append(f"| {r['environment']} | {pct(r['source_recall'])} | {pct(r['matching_reference_visible_recall'])} |")
    lines+=['',
        'The separate pooled deterministic matcher was enumerated over all 1,716 subsets from publicly parsed claim/reference contradictions. It is a different detector; its output is not LLM evidence.','',
        '| Route / condition | Recall % | Acquired-benign FPR % |','|---|---:|---:|']
    for r in result['uniform_selection_pooled_matcher']['rows']:lines.append(f"| {r['environment']} / {r['condition']} | {pct(r['recall'])} | {pct(r['inspected_benign_fpr'])} |")
    lines+=['','## Complete compact grid','','Each cell is **score recall / visible-reference recall / clean acquired-benign FPR (%)**. `infeasible` means no judge performance is inferred. This table includes every completed configuration; privileged oracles remain labeled by name. Budget includes selected edges k, maximum group s, and per-call byte allowance. Full 1,452 condition rows and costs are in `full_grid.csv`.','',
        '| Selector / grouper | k / s / bytes | 4 aligned | 4 shuffled | 16 aligned | 16 shuffled |','|---|---|---|---|---|---|']
    grid=result['full_grid'];indexed={key(r):r for r in grid};configs=sorted({tuple(r[k] for k in KEYS[3:]) for r in grid})
    for config in configs:
        cells=[]
        for size,route in itertools.product((4,16),('aligned','shuffled')):
            attacked=indexed[(size,route,'false_report')+config];clean=indexed[(size,route,'clean')+config]
            cells.append('infeasible' if not attacked['llm_performance_defined'] else ' / '.join(pct(v) for v in (attacked.get('source_recall'),attacked.get('matching_reference_visible_recall'),clean.get('source_fpr'))))
        lines.append('| '+config[0]+' / '+config[1]+' | '+' / '.join(map(str,config[2:]))+' | '+' | '.join(cells)+' |')
    path.write_text('\n'.join(lines)+'\n')

def from_release(main,supplement,out):
    """Reproduce from candidates alone, importing their frozen parser sources."""
    main=Path(main).resolve();supplement=Path(supplement).resolve();out=Path(out).resolve()
    for root in (main,supplement):
        manifest=read(root/'manifest.json')
        for name,want in manifest['files_sha256'].items():
            p=Path(name)
            if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe candidate path')
            if sha(root/p)!=want:raise ValueError('candidate artifact hash mismatch: '+name)
    with tempfile.TemporaryDirectory(prefix='completed-grid-release-replay-') as temp:
        temp=Path(temp);run=temp/'main';sub=temp/'supplement';run.mkdir();sub.mkdir()
        def unpack(src,dst):
            dst.parent.mkdir(parents=True,exist_ok=True)
            if src.suffix=='.gz':
                with gzip.open(src,'rb') as a,dst.open('wb') as b:shutil.copyfileobj(a,b)
            else:shutil.copyfile(src,dst)
        paths={'plan.json':'evaluation/plan.json','episodes.jsonl':'evaluation/original_records.jsonl.gz',
            'assignments.jsonl':'evaluation/assignments.jsonl.gz','exact_random_plans.jsonl':'evaluation/exact_random_plans.jsonl.gz','views.jsonl':'audit/views.jsonl.gz'}
        for target,source in paths.items():unpack(main/source,run/target)
        for name in ['manifest.json',*read(main/'analysis/manifest.json')['output_sha256']]:
            source=main/'analysis'/name
            if not source.exists():source=source.with_name(source.name+'.gz')
            unpack(source,run/'analysis'/name)
        for name in read(supplement/'manifest.json')['run_files']:unpack(supplement/'audit/run'/(name+'.gz'),sub/name)
        code="import sys,runpy;sys.path[:0]=sys.argv[1:3];sys.argv=sys.argv[3:];runpy.run_path(sys.argv[0],run_name='__main__')"
        command=[sys.executable,'-c',code,str(main/'source/scripts'),str(main/'source/src'),str(Path(__file__).resolve()),'--run',str(run),'--supplement',str(sub),'--out',str(out)]
        env={'PATH':os.environ.get('PATH',''),'LANG':os.environ.get('LANG','C.UTF-8'),'PYTHONDONTWRITEBYTECODE':'1'}
        subprocess.run(command,env=env,check=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--supplement',required=True);p.add_argument('--out',required=True);p.add_argument('--from-release',action='store_true');a=p.parse_args()
    if a.from_release:from_release(a.run,a.supplement,a.out);return
    r=audit(a.run,a.supplement,a.out)
    print(json.dumps({'full_grid_rows':len(r['full_grid']),'shuffled_supported_exact_grouping_witnesses':len(r['shuffled_exact_grouping_witnesses']),'uniform_selection':r['uniform_selection_reference_visibility']},indent=2))
if __name__=='__main__':main()
