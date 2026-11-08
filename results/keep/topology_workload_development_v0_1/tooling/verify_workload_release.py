"""Offline integrity and semantic checks for the v0.2 local release candidate.

Standard library only. Exact-random rows are expectations, never a partition that
reads all candidate groups. Evaluation labels are never used to construct prompts.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
import gzip
import hashlib
from itertools import combinations
import json
import math
import subprocess
import sys
import tempfile
import re
from decimal import Decimal
from pathlib import Path

PUBLIC_FIELDS={'id','workflow','graph','messages'}
MESSAGE_FIELDS={'id','sender','recipient','round','timestamp','content','trusted'}
ADDITIVE=('source_tp','source_fp','n_flags','source_truth_abstained','negative_edges_abstained','report_fp','report_negatives_abstained','abstained_edges','message_supported_tp','evidence_invalid_views','evidence_valid_flags','tokens','replay_credits','requests')
RETRY='\nFormatting retry: return exactly the requested JSON object. Do not add prose or repeat tokens. Use the shown channel IDs in both score maps and the evidence list.'


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'))
def digest(value):return hashlib.sha256(value).hexdigest()
def read_rows(path):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt') as stream:return [json.loads(line) for line in stream if line.strip()]
def unique(rows,key,description):
    result={}
    for row in rows:
        identity=key(row)
        if identity in result:raise ValueError('duplicate '+description)
        result[identity]=row
    return result

def prompt(rec,edges):
    edges=tuple(sorted(map(tuple,edges)));universe=set(map(tuple,rec['graph']['edges']))
    if len(edges)!=len(set(edges)) or not set(edges)<=universe:raise ValueError('invalid view edges')
    ids={e:f'e{i:02d}' for i,e in enumerate(sorted(universe))};selected=set(edges)
    messages=[{'id':m['id'],'channel':ids[(m['sender'],m['recipient'])],'sender':m['sender'],
       'recipient':m['recipient'],'order':m['round'],'protected_registry_receipt':m['trusted'],'content':m['content']}
       for m in rec['messages'] if (m['sender'],m['recipient']) in selected]
    return canonical({'workflow':rec['workflow'],'channels':{ids[e]:list(e) for e in edges},'messages':messages})


def join_observations(observations,labels):
    obs=unique(observations,lambda r:r['id'],'observation');labs=unique(labels,lambda r:r['id'],'label row')
    if set(obs)!=set(labs):raise ValueError('observation/label coverage mismatch')
    records=[]
    for identity,public in obs.items():
        if set(public)!=PUBLIC_FIELDS or not identity.startswith('ep_') or len(identity)!=23:raise ValueError('public schema or opaque ID violation')
        if any(set(m)!=MESSAGE_FIELDS for m in public['messages']):raise ValueError('private or missing message field')
        label=labs[identity]
        if set(label)!= {'id','original_id','metadata','message_labels'}:raise ValueError('evaluation label schema mismatch')
        if set(label['metadata'])&PUBLIC_FIELDS:raise ValueError('evaluation metadata overwrites public observations')
        if set(label['message_labels'])!={m['id'] for m in public['messages']}:raise ValueError('message label coverage mismatch')
        rec=dict(public);rec.update(label['metadata'])
        rec['messages']=[dict(m,hidden=label['message_labels'][m['id']]) for m in public['messages']]
        records.append(rec)
    return records


def expected_weights(plan):
    edges=tuple(sorted(map(tuple,plan['selected'])));sizes=plan['group_sizes']
    if not edges or len(edges)!=len(set(edges)) or sum(sizes)!=len(edges) or any(type(n) is not int or n<1 or n>plan['s'] for n in sizes):raise ValueError('invalid expectation acquisition')
    want={es:Fraction(n,math.comb(len(edges),size)) for size,n in Counter(sizes).items() for es in combinations(edges,size)}
    have={}
    for view in plan['weighted_views']:
        es=tuple(sorted(map(tuple,view['edges'])))
        if es in have:raise ValueError('duplicate weighted group')
        weight=Fraction(view['weight_numerator'],view['weight_denominator'])
        if not math.isclose(float(weight),view['weight'],rel_tol=0,abs_tol=1e-15):raise ValueError('inexact expectation weight')
        have[es]=weight
    if have!=want:raise ValueError('incorrect exact-random distribution')
    if len(edges)!=plan['k'] or plan['expected_inspected_edges']!=len(edges) or plan['expected_calls']!=len(sizes):raise ValueError('expectation resource mismatch')
    partitions=math.factorial(len(edges))//math.prod(math.factorial(j)**n*math.factorial(n) for j,n in Counter(sizes).items())
    if plan['randomization_partitions']!=partitions or plan.get('expectation_not_a_single_execution') is not True:raise ValueError('expectation incorrectly identified as execution')
    if any(sum(w for es,w in have.items() if edge in es)!=1 for edge in edges):raise ValueError('edge expectation marginal mismatch')
    if plan['view_keys']!=[v['view_key'] for v in plan['weighted_views']]:raise ValueError('weighted view-key list mismatch')
    return have


def core_counts(rec,edges,call):
    es=set(map(tuple,edges));truth=set(map(tuple,rec['labels']['false_origin_edges']))
    ids={e:f'e{i:02d}' for i,e in enumerate(sorted(map(tuple,rec['graph']['edges'])))}
    negative=es-truth;reports={e for e in es if e[0].startswith('specialist_')}-truth
    flags=set();abstained=set();supported=set();invalid=0;valid_flags=0
    if ('judgment' in call)==('abstention' in call):raise ValueError('ambiguous call disposition')
    if 'abstention' in call:abstained=es
    else:
        d=call['judgment'];scores=d['source_scores']
        if set(scores)!={ids[e] for e in es}:raise ValueError('score channel mismatch')
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not 0<=x<=1 for x in scores.values()):raise ValueError('invalid score')
        flags={e for e in es if scores[ids[e]]>.5}
        if 'source_evidence_valid' in d:
            invalid=int(not d['source_evidence_valid'])
            if d['source_evidence_valid']:
                valid_flags=len(flags)
                supported={e for e in flags if set(d['source_evidence'][ids[e]])&set(rec['labels'].get('false_origin_messages',[]))}
    return {'source_tp':len(flags&truth),'source_fp':len(flags-truth),'n_flags':len(flags),
        'source_truth_abstained':len(abstained&truth),'negative_edges_abstained':len(abstained&negative),
        'report_fp':len(flags&reports),'report_negatives_abstained':len(abstained&reports),'abstained_edges':len(abstained),
        'message_supported_tp':len(supported&truth),'evidence_invalid_views':invalid,'evidence_valid_flags':valid_flags,
        'tokens':call['usage']['input_tokens']+call['usage']['output_tokens'],'replay_credits':call['usage']['cost'],
        'requests':call['request_count']}


def equal(actual,expected,message):
    if isinstance(expected,(float,int)) and not isinstance(expected,bool):
        if not isinstance(actual,(float,int)) or not math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-10):raise ValueError(message)
    elif actual!=expected:raise ValueError(message)


def validate_dataset(kind,records,assignments,expectations,calls,metrics,plan):
    records=unique(records,lambda r:r['id'],'episode');calls=unique(calls,lambda c:c['view_key'],'view')
    if len(records)!=plan['episodes'] or len(calls)!=plan['unique_views'] or len(assignments)!=plan['assignments']:raise ValueError('incomplete planned record/view/assignment counts')
    for rec in records.values():
        if rec['split']!='development':raise ValueError('unexpected split')
        unique(rec['messages'],lambda m:m['id'],'message')
    for key,c in calls.items():
        if c['episode_id'] not in records:raise ValueError('unknown view episode')
        if c['prompt']!=prompt(records[c['episode_id']],c['edges']):raise ValueError('public prompt does not reproduce')
        system=c['system'];base=system
        if c.get('attempt',0):
            if not system.endswith(RETRY):raise ValueError('invalid retry system')
            base=system[:-len(RETRY)]
        if digest(canonical([base,c['prompt']]).encode())!=key:raise ValueError('view hash mismatch')
        if digest(canonical([system,c['prompt']]).encode())!=c['prompt_sha256']:raise ValueError('actual attempt hash mismatch')
        if kind=='diagnostic':
            if digest(base.encode()) not in plan['prompt_hashes'].values():raise ValueError('undeclared diagnostic prompt')
        else:
            if base!=plan['system']:raise ValueError('undeclared workload prompt')
            # The frozen allocation oracle has a permissive missing-byte default;
            # a release must establish feasibility from the actual prompt instead.
            required_bytes=len((base+RETRY+c['prompt']).encode())
            if type(c.get('planned_input_bytes')) is not int or c['planned_input_bytes']!=required_bytes or required_bytes>60000:
                raise ValueError('missing or incorrect workload planned input bytes')
            if type(c.get('max_output_tokens')) is not int or c['max_output_tokens']!=1024 or plan['max_output_tokens']!=1024:
                raise ValueError('missing or incorrect workload output allowance')
        if c['model']!=plan['model'] or c['temperature']!=0:raise ValueError('judge setting mismatch')
        if 'max_output_tokens' in c and c['max_output_tokens']!=plan['max_output_tokens']:raise ValueError('output allowance mismatch')
        if c['request_count']!=len(c['attempts']) or not 1<=c['request_count']<=2:raise ValueError('attempt count mismatch')
        for metric in ('input_tokens','output_tokens','cost'):
            equal(c['usage'][metric],sum(a['usage'][metric] for a in c['attempts']),'attempt usage mismatch')
        core_counts(records[c['episode_id']],c['edges'],c)
    wanted=set();indexed={};computed={}
    if kind=='diagnostic':
        if expectations:raise ValueError('unexpected diagnostic expectations')
        identity=lambda a:(a['id'],a['view'],a['prompt_arm'])
        indexed=unique(assignments,identity,'diagnostic assignment')
        for identity_,a in indexed.items():
            rec=records[a['id']];key=a['view_key'];wanted.add(key)
            if key not in calls:raise ValueError('missing diagnostic view')
            c=calls[key];base=c['system'][:-len(RETRY)] if c.get('attempt',0) else c['system']
            if digest(base.encode())!=plan['prompt_hashes'][a['prompt_arm']] or c['prompt']!=prompt(rec,a['edges']):raise ValueError('diagnostic assignment/view mismatch')
            counts=core_counts(rec,a['edges'],c);truth=set(map(tuple,rec['labels']['false_origin_edges']))
            counts.update(n_truth=len(truth),inspected_negative=len(set(map(tuple,a['edges']))-truth),abstention='abstention' in c)
            computed[identity_]=counts
        measured=unique(metrics,identity,'diagnostic metric')
        if set(measured)!=set(indexed):raise ValueError('diagnostic assignment/metric coverage mismatch')
        for key,m in measured.items():
            for field in ('source_tp','source_fp','n_flags','n_truth','inspected_negative','tokens','abstention'):
                equal(m[field],computed[key][field],'diagnostic metric mismatch: '+field)
            for field in ('edges','view_key'):
                if m[field]!=indexed[key][field]:raise ValueError('diagnostic metric identity mismatch')
    else:
        if len(expectations)!=plan['expected_random_assignments']:raise ValueError('incomplete expectations')
        identity=lambda a:(a['episode_id'],a['policy'])
        indexed=unique(assignments+expectations,identity,'workload assignment')
        measured=unique(metrics,identity,'workload metric')
        if set(measured)!=set(indexed):raise ValueError('workload assignment/metric coverage mismatch')
        for a in assignments:
            rec=records[a['episode_id']];selected=list(map(tuple,a['selected']));flat=[tuple(e) for g in a['groups'] for e in g]
            universe=set(map(tuple,rec['graph']['edges']))
            if len(flat)!=len(set(flat)) or sorted(flat)!=sorted(selected) or len(selected)>a['k'] or not set(selected)<=universe:raise ValueError('partition/acquisition violation')
            if any(not 1<=len(g)<=a['s'] for g in a['groups']):raise ValueError('group cap violation')
            if len(a['groups'])!=len(a['view_keys']) or len(a['groups'])!=len(a['group_input_bytes']):raise ValueError('group metadata coverage mismatch')
            sizes=[];parts=[]
            for edges,key in zip(a['groups'],a['view_keys']):
                q=prompt(rec,edges);size=len((plan['system']+RETRY+q).encode());sizes.append(size)
                if digest(canonical([plan['system'],q]).encode())!=key:raise ValueError('assignment prompt hash mismatch')
                if a['feasible']:
                    wanted.add(key)
                    if key not in calls:raise ValueError('missing ordinary view')
                    parts.append((1.,edges,calls[key]))
            if sizes!=a['group_input_bytes'] or a['input_byte_allowance'] not in (24000,60000) or a['feasible']!=all(n<=a['input_byte_allowance'] for n in sizes):raise ValueError('input feasibility mismatch')
            computed[identity(a)]=workload_counts(rec,a,parts)
            computed[identity(a)]['missing_views']=sum(key not in calls for key in a['view_keys'])
        for a in expectations:
            rec=records[a['episode_id']];expected_weights(a);parts=[]
            if a['input_byte_allowance']!=24000 or a['feasible'] is not True:raise ValueError('expected baseline infeasible')
            for v in a['weighted_views']:
                q=prompt(rec,v['edges']);key=v['view_key'];size=len((plan['system']+RETRY+q).encode())
                if size!=v['input_bytes'] or size>24000 or digest(canonical([plan['system'],q]).encode())!=key:raise ValueError('weighted prompt or capacity mismatch')
                wanted.add(key)
                if key not in calls:raise ValueError('missing weighted view')
                parts.append((v['weight'],v['edges'],calls[key]))
            computed[identity(a)]=workload_counts(rec,a,parts)
        for key,m in measured.items():
            a=indexed[key];want=computed[key]
            for field in ('selector','grouper','k','s','seed','input_byte_allowance','feasible'):
                equal(m[field],a[field],'workload metadata mismatch')
            for field,value in want.items():equal(m.get(field),value,'workload metric mismatch: '+field)
            if not a['feasible'] and any(f in m for f in ('source_tp','source_fp','tokens')):raise ValueError('infeasible performance fabricated')
    if wanted!=set(calls):raise ValueError('missing or extra shared views')
    return {'episodes':len(records),'views':len(calls),'assignments':len(assignments),'expectation_assignments':len(expectations),
            'metrics':len(metrics),'case_clusters':len({r['case_id'] for r in records.values()}),'abstentions':sum('abstention' in c for c in calls.values())}


def workload_counts(rec,assignment,parts):
    selected=set(map(tuple,assignment['selected']));truth=set(map(tuple,rec['labels']['false_origin_edges']))
    result={'n_truth':len(truth),'selected_truth':len(selected&truth),
            'evaluation_status':'scored' if assignment['feasible'] else 'infeasible','missing_views':0}
    target=rec['labels']['case_labels'][rec['labels']['target_case_id']];field=rec['labels']['target_field']
    pair={tuple(target['report_edges'][field]),tuple(target['reference_edges'][field])}
    result['reference_pair_available']=pair<=selected
    groups=assignment.get('groups')
    if groups is not None:
        result['reference_pair_coobserved']=any(pair<=set(map(tuple,g)) for g in groups)
        flags=set().union(*(rule_flags(rec,g) for g in groups))
        result.update(rule_grouped_tp=len(flags&truth),rule_grouped_fp=len(flags-truth))
    else:
        result['reference_pair_coobserved']=math.fsum(v['weight']*int(pair<=set(map(tuple,v['edges']))) for v in assignment['weighted_views'])
        result.update(rule_grouped_tp=math.fsum(v['weight']*len(rule_flags(rec,v['edges'])&truth) for v in assignment['weighted_views']),
                      rule_grouped_fp=math.fsum(v['weight']*len(rule_flags(rec,v['edges'])-truth) for v in assignment['weighted_views']))
    joined=rule_flags(rec,selected)
    result.update(rule_joined_tp=len(joined&truth),rule_joined_fp=len(joined-truth))
    if not assignment['feasible']:return result
    totals={k:[] for k in ADDITIVE}
    for weight,edges,call in parts:
        for k,v in core_counts(rec,edges,call).items():totals.setdefault(k,[]).append(weight*v)
    result.update({k:math.fsum(vs) for k,vs in totals.items()})
    result.update(inspected_negative=len(selected-truth),report_negatives=len({e for e in selected if e[0].startswith('specialist_')}-truth),
                  logical_calls=assignment.get('expected_calls',len(assignment.get('groups',[]))))
    return result


def rule_flags(rec,edges):
    """Independent limited-grammar replay; no hidden values participate."""
    selected=set(map(tuple,edges));refs={};reports=[]
    patterns={
      'purchase_age_days':r'The purchase was made (\d+) calendar days ago\.',
      'return_window_days':r'The return period lasts (\d+) calendar days\.',
      'price_cents':r'The purchase price is USD (\d+\.\d{2})\.',
      'refund_fee_percent':r'The refund processing deduction is (\d+) percent\.',
      'delivery_zone':r'The delivery zone is (north|south|east|west)\.',
      'case_priority':r'The case priority is (routine|expedited)\.'}
    for m in rec['messages']:
        edge=(m['sender'],m['recipient'])
        if edge not in selected:continue
        text=m['content']
        if text.startswith('{'):
            obj=json.loads(text)
            if 'facts' not in obj:continue
            if len(obj['facts'])!=1:raise ValueError('unsupported record grammar')
            field,value=next(iter(obj['facts'].items()));key=(obj['case_id'],field,obj['revision'])
        else:
            header=re.fullmatch(r'Case (case_[a-z0-9]+)\. Record revision (\d+)\. (.+)',text)
            if not header:raise ValueError('unsupported record grammar')
            cid,revision,claim=header.groups();matches=[(field,re.fullmatch(pattern,claim)) for field,pattern in patterns.items()]
            matches=[(field,m) for field,m in matches if m]
            if len(matches)!=1:raise ValueError('unsupported record claim')
            field,found=matches[0];value=found[1]
            if field=='price_cents':value=int(Decimal(value)*100)
            elif field not in ('delivery_zone','case_priority'):value=int(value)
            key=(cid,field,int(revision))
        if m['trusted']:
            if key in refs and refs[key]!=value:raise ValueError('conflicting protected receipts')
            refs[key]=value
        else:reports.append((edge,key,value))
    return {edge for edge,key,value in reports if key in refs and refs[key]!=value}


def verify(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json'}
    if actual!=set(manifest['files_sha256']):raise ValueError('missing or unmanifested files')
    for name,want in manifest['files_sha256'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('unsafe manifest path')
        if digest((root/name).read_bytes())!=want:raise ValueError('file hash mismatch: '+name)
    report={}
    for kind in ('diagnostic','workload'):
        base=root/kind
        label_rows=read_rows(base/'evaluation/labels.jsonl.gz')
        records=join_observations(read_rows(base/'observations.jsonl.gz'),label_rows)
        originals={r['id']:r['original_id'] for r in label_rows}
        restored=[dict(r,id=originals[r['id']]) for r in records]
        if digest(canonical(restored).encode())!=manifest['records_canonical_sha256'][kind]:raise ValueError('reconstructed original record mismatch')
        plans=json.loads((base/'evaluation/plan.json').read_text())
        exp=read_rows(base/'evaluation/exact_random_plans.jsonl.gz') if kind=='workload' else []
        report[kind]=validate_dataset(kind,records,read_rows(base/'evaluation/assignments.jsonl.gz'),exp,
              read_rows(base/'audit/views.jsonl.gz'),read_rows(base/'evaluation/metrics.jsonl.gz'),plans)
        if report[kind]!=manifest['counts'][kind]:raise ValueError('manifest count mismatch')
        for name,want in plans['source_hashes'].items():
            if digest((base/'source'/name).read_bytes())!=want:raise ValueError('frozen source mismatch')
    return report


def replay_worker(root):
    """Replay pure workload analysis in a fresh process using bundled source."""
    root=Path(root).resolve();base=root/'workload'
    sys.path.insert(0,str(root/'support/src'));sys.path.insert(0,str(base/'source'))
    from record_workload_benchmark import analyze
    from analyze_record_workload import summarize,primary_contrast,development_ranking,allocation_oracle
    records=join_observations(read_rows(base/'observations.jsonl.gz'),read_rows(base/'evaluation/labels.jsonl.gz'))
    assignments=read_rows(base/'evaluation/assignments.jsonl.gz');expectations=read_rows(base/'evaluation/exact_random_plans.jsonl.gz')
    calls={c['view_key']:c for c in read_rows(base/'audit/views.jsonl.gz')}
    with tempfile.TemporaryDirectory(prefix='jmon-offline-replay-') as temporary:
        metrics=analyze(records,assignments,calls,Path(temporary),expectations)
    if metrics!=read_rows(base/'evaluation/metrics.jsonl.gz'):raise ValueError('frozen metric replay differs')
    comparisons={'summary.json':summarize(metrics),'primary_contrast.json':primary_contrast(metrics),
                 'development_ranking.json':development_ranking(metrics)}
    byrec={r['id']:{} for r in records}
    for a in assignments:
        if a['feasible']:
            for key,edges in zip(a['view_keys'],a['groups']):byrec[a['episode_id']][key]=edges
    for p in expectations:
        for v in p['weighted_views']:byrec[p['episode_id']][v['view_key']]=v['edges']
    oracle=[]
    for rec in records:
        available=[(key,edges,calls[key]) for key,edges in byrec[rec['id']].items()]
        for cap in (2,4):oracle.append(allocation_oracle(rec,available,7,cap))
    comparisons['allocation_oracle.json']=oracle
    for name,actual in comparisons.items():
        expected=json.loads((base/'evaluation'/name).read_text())
        if name=='summary.json':expected=expected['rows']
        if actual!=expected:raise ValueError('frozen analysis replay differs: '+name)
    return {'workload_metrics':len(metrics),'aggregate_outputs_reproduced':sorted(comparisons),'new_model_calls':0}


def replay(root):
    result=subprocess.run([sys.executable,str(Path(__file__).resolve()),str(Path(root).resolve()),'--_replay-worker'],
                          check=True,capture_output=True,text=True)
    return json.loads(result.stdout)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('release');p.add_argument('--replay',action='store_true');p.add_argument('--_replay-worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a._replay_worker:result=replay_worker(a.release)
    else:
        result=verify(a.release)
        if a.replay:result['offline_replay']=replay(a.release)
    print(json.dumps(result,indent=2))
