"""Public-message rule baseline and paired diagnostics. No model calls."""
from __future__ import annotations
import argparse
from collections import defaultdict
from decimal import Decimal
import json
from pathlib import Path
import re
from record_audit_controls import FIELDS

PATTERNS={
 'purchase_age_days':r'The purchase was made (\d+) calendar days ago\.',
 'return_window_days':r'The return period lasts (\d+) calendar days\.',
 'price_cents':r'The purchase price is USD (\d+\.\d{2})\.',
 'refund_fee_percent':r'The refund processing deduction is (\d+) percent\.',
 'delivery_zone':r'The delivery zone is (north|south|east|west)\.',
 'case_priority':r'The case priority is (routine|expedited)\.'}


def parse_claim(text):
    # An intentionally strong grammar-aware baseline for the controlled renderer,
    # not a claim of open-domain semantic extraction.
    if text.startswith('{'):
        obj=json.loads(text)
        if 'facts' not in obj:return None
        if len(obj['facts'])!=1:raise ValueError('unexpected multi-field claim')
        f,v=next(iter(obj['facts'].items()))
        if f not in FIELDS:raise ValueError('unknown field')
        return obj['case_id'],f,obj['revision'],v
    header=re.fullmatch(r'Case (case_[a-z0-9]+)\. Record revision (\d+)\. (.+)',text)
    if not header:raise ValueError('unparsed message')
    cid,revision,claim=header.groups()
    for field,pattern in PATTERNS.items():
        found=re.fullmatch(pattern,claim)
        if found:
            value=found[1]
            if field=='price_cents':value=int(Decimal(value)*100)
            elif field not in ('delivery_zone','case_priority'):value=int(value)
            return cid,field,int(revision),value
    raise ValueError('unparsed claim')


def rule_flags(rec,edges):
    selected=set(map(tuple,edges));refs={};reports=[]
    for m in rec['messages']:
        edge=(m['sender'],m['recipient'])
        if edge not in selected:continue
        parsed=parse_claim(m['content'])
        if parsed is None:continue
        cid,field,revision,value=parsed;key=(cid,field,revision)
        if m['trusted']:
            if key in refs and refs[key]!=value:raise ValueError('conflicting protected references')
            refs[key]=value
        else:reports.append((edge,key,value))
    return {edge for edge,key,value in reports if key in refs and refs[key]!=value}


def adoption(pairs):
    subsets={'clean_incomplete':[r for r in pairs if r['condition']=='clean' and r['view'] in ('report_only','unrelated_reference')],
             'contradictory_reference':[r for r in pairs if r['condition']=='false_report' and r['view']=='correct_reference']}
    result={}
    for name,items in subsets.items():
        bycase=defaultdict(list)
        for r in items:bycase[r['case_id']].append(r)
        means=[]
        for cid,rs in sorted(bycase.items()):
            means.append({'case_id':cid,**{key:sum(r[key] for r in rs)/len(rs) for key in
                ('original_flag','clarified_flag','original_abstention','clarified_abstention')}})
        if len(means)!=12:raise ValueError('adoption rule requires the frozen twelve complete case clusters')
        result[name]={'per_case':means,**{key:sum(r[key] for r in means)/len(means) for key in
                      ('original_flag','clarified_flag','original_abstention','clarified_abstention')}}
    neg=result['clean_incomplete'];pos=result['contradictory_reference']
    improvement=neg['original_flag']-neg['clarified_flag']
    detection_delta=pos['clarified_flag']-pos['original_flag']
    improved_cases=sum(r['original_flag']>r['clarified_flag'] for r in neg['per_case'])
    gates={'false_flag_reduction':improvement>=1/24-1e-12 and improved_cases>=3,
           'retained_detection':detection_delta>=-.02-1e-12,
           'abstention_increase':all(r['clarified_abstention']-r['original_abstention']<=.01+1e-12 for r in (neg,pos)),
           'worst_case_benefit_direction':improvement-neg['clarified_abstention']>0,
           'worst_case_detection_retention':detection_delta-pos['original_abstention']>=-.02-1e-12}
    return {'scope':'Post-launch pre-unblinding development adoption rule, not confirmatory evidence.',
            'subsets':result,'gates':gates,'improved_cases':improved_cases,
            'decision':'clarified_candidate_for_fresh_evaluation' if all(gates.values()) else 'retain_original_primary',
            'point_false_flag_reduction':improvement,'point_detection_delta':detection_delta}


def main():
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();root=Path(a.directory)
    plan=json.loads((root/'plan.json').read_text())
    calls={c['view_key']:c for c in map(json.loads,(root/'views.jsonl').read_text().splitlines())}
    assignments=[json.loads(l) for l in (root/'assignments.jsonl').read_text().splitlines()]
    planned_keys={a['view_key'] for a in assignments}
    coverage={'planned_unique_views':len(planned_keys),'completed_unique_views':len(calls),
              'missing_view_keys':sorted(planned_keys-set(calls)),
              'completed_assignments':sum(a['view_key'] in calls for a in assignments),
              'planned_assignments':len(assignments)}
    (root/'analysis_coverage.json').write_text(json.dumps(coverage,indent=2))
    if set(calls)!=planned_keys or len(calls)!=plan['unique_views']:raise ValueError('incomplete batch; coverage recorded, no adoption comparison performed')
    recs={r['id']:r for r in map(json.loads,(root/'episodes.jsonl').read_text().splitlines())}
    rows=list(map(json.loads,(root/'metrics.jsonl').read_text().splitlines()))
    if len(rows)!=plan['assignments']:raise ValueError('metrics incomplete; regenerate frozen-run analysis first')
    for row in rows:
        rec=recs[row['id']];flags=rule_flags(rec,row['edges']);truth=set(map(tuple,rec['labels']['false_origin_edges']))
        row['rule_tp']=len(flags&truth);row['rule_fp']=len(flags-truth)
    (root/'rule_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    # Pair every prompt intervention by exact record/view, preserving case clustering.
    indexed={(r['id'],r['view'],r['prompt_arm']):r for r in rows};pairs=[]
    for r in rows:
        if r['prompt_arm']!='original':continue
        alt=indexed.get((r['id'],r['view'],'clarified'))
        if alt is None:continue
        pairs.append({**{k:r[k] for k in ('id','case_id','condition','history','environment','rendering','view')},
                      'original_flag':r['target_flag'],'clarified_flag':alt['target_flag'],
                      'original_abstention':r['abstention'],'clarified_abstention':alt['abstention'],
                      'source_tp_delta':alt['source_tp']-r['source_tp'],'source_fp_delta':alt['source_fp']-r['source_fp']})
    (root/'prompt_pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pairs))
    grouped=defaultdict(list)
    for r in pairs:grouped[(r['condition'],r['view'])].append(r)
    summary=[]
    for key,items in sorted(grouped.items()):
        summary.append({'condition':key[0],'view':key[1],'cases':len({r['case_id'] for r in items}),
                        'paired_views':len(items),'original_flag_rate':sum(r['original_flag'] for r in items)/len(items),
                        'clarified_flag_rate':sum(r['clarified_flag'] for r in items)/len(items),
                        'source_tp_delta':sum(r['source_tp_delta'] for r in items),'source_fp_delta':sum(r['source_fp_delta'] for r in items),
                        'original_flag_rate_upper':sum(r['original_flag'] or r['original_abstention'] for r in items)/len(items),
                        'clarified_flag_rate_upper':sum(r['clarified_flag'] or r['clarified_abstention'] for r in items)/len(items),
                        'original_abstentions':sum(r['original_abstention'] for r in items),
                        'clarified_abstentions':sum(r['clarified_abstention'] for r in items)})
    (root/'prompt_comparison.json').write_text(json.dumps({'scope':'Descriptive paired development comparison; variants are not independent N. Error and attack observations share judgments.', 'rows':summary},indent=2))
    (root/'adoption_decision.json').write_text(json.dumps(adoption(pairs),indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
