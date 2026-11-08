"""Post-launch supplemental workload audit; never modifies frozen core analyses.

Requires complete planned-view coverage. Replays primary rows in a temporary
folder; all new outputs are separately named and exploratory.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import tempfile
from record_workflow import canonical
from record_audit_analysis import parse_claim
from monitor_record_workflows import edge_ids

ARMS=('provenance','exact_matched_random')
ROUTES=('aligned','shuffled')
CONDITIONS=('clean','false_report')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):return [json.loads(l) for l in Path(path).read_text().splitlines()]
def mean(values):return math.fsum(values)/len(values) if values else None


def require_coverage(plan,assignments,expectations,call_rows):
    expected={key for a in assignments if a['feasible'] for key in a['view_keys']}
    expected.update(key for a in expectations for key in a['view_keys'])
    calls={c['view_key']:c for c in call_rows}
    if len(calls)!=len(call_rows):raise ValueError('duplicate completed view keys')
    if set(calls)!=expected or len(calls)!=plan['unique_views']:raise ValueError('incomplete or extra workload views; supplemental contrasts withheld')
    return calls


def background_overlap(records):
    bycase=defaultdict(set)
    for rec in records:
        ids=set(rec['labels']['case_ids'])
        if len(ids)!=rec['workload_size'] or rec['case_id'] not in ids:raise ValueError('invalid workload case IDs')
        bycase[rec['case_id']].update(ids)
    shared=[{'left':a,'right':b,'shared_case_ids':sorted(bycase[a]&bycase[b])}
            for a,b in itertools.combinations(sorted(bycase),2) if bycase[a]&bycase[b]]
    return {'target_clusters':len(bycase),'distinct_cases':len(set().union(*bycase.values())),
            'cases_per_target_cluster':{k:len(v) for k,v in sorted(bycase.items())},
            'cross_cluster_overlaps':shared,'no_case_id_overlap_across_target_clusters':not shared,
            'scope':'Case-identity overlap only. A shared synthetic generator/schema still limits generalization.'}


def primary(row):
    return (row['workload_size']==16 and row['k']==7 and row['s']==2 and
            row['input_byte_allowance']==24000 and row['selector']=='provenance_bundles' and
            row['grouper'] in ARMS and row['condition'] in CONDITIONS)


def bounds(operational,lower,upper):
    if lower>operational+1e-10 or upper<operational-1e-10:raise ValueError('invalid abstention bounds')
    return {'operational':operational,'lower':lower,'upper':upper}


def difference(left,right):
    return bounds(left['operational']-right['operational'],left['lower']-right['upper'],left['upper']-right['lower'])


def condition_arm(rs):
    def avg(key):return mean([r[key] for r in rs])
    tp=avg('source_tp');fp=avg('source_fp');unknown_tp=avg('source_truth_abstained');unknown_fp=avg('negative_edges_abstained')
    rates={}
    for name,numerator,denominator,unresolved in (('selected_fpr','source_fp','inspected_negative','negative_edges_abstained'),
            ('network_fpr','source_fp','network_negative','negative_edges_abstained'),
            ('report_fpr','report_fp','report_negatives','report_negatives_abstained')):
        n=math.fsum(r[denominator] for r in rs)
        if n<=0:raise ValueError('primary FPR denominator absent')
        point=math.fsum(r[numerator] for r in rs)/n
        rates[name]=bounds(point,point,point+math.fsum(r[unresolved] for r in rs)/n)
    flagged=avg('flagged_views');valid=avg('flagged_views_with_valid_citations')
    return {**rates,'flagged_view_citation_validity_rate':valid/flagged if flagged else None,'assignments':len(rs),'tp':bounds(tp,tp,tp+unknown_tp),'fp':bounds(fp,fp,fp+unknown_fp),
            'utility':bounds(tp-2*fp,tp-2*(fp+unknown_fp),tp+unknown_tp-2*fp),
            'truth_edges_abstained':unknown_tp,'negative_edges_abstained':unknown_fp,
            'target_coverage':avg('selected_truth'),'reference_pair_available':avg('reference_pair_available'),
            'reference_pair_coobserved':avg('reference_pair_coobserved'),
            'rule_grouped_tp':avg('rule_grouped_tp'),'rule_grouped_fp':avg('rule_grouped_fp'),
            'rule_joined_tp':avg('rule_joined_tp'),'rule_joined_fp':avg('rule_joined_fp'),
            'weaker_message_supported_tp':avg('message_supported_tp'),
            'strict_relational_supported_tp':avg('strict_relational_supported_tp'),
            'flagged_views':avg('flagged_views'),'flagged_views_with_valid_citations':avg('flagged_views_with_valid_citations')}


def sign_summary(values):
    def sign(x):return 0 if abs(x)<1e-10 else 1 if x>0 else -1
    return {'cases':len(values),'negative':sum(sign(x)==-1 for x in values),'zero':sum(sign(x)==0 for x in values),
            'positive':sum(sign(x)==1 for x in values),'mean':mean(values),
            'scope':'Descriptive case signs; not a hypothesis test or independent replication count.'}


def paired_contrasts(metrics,case_ids):
    selected=[r for r in metrics if primary(r)]
    if any(r['evaluation_status']!='scored' for r in selected):raise ValueError('primary assignment incomplete or infeasible')
    if {r['case_id'] for r in selected}!=set(case_ids):raise ValueError('missing primary case cluster')
    indexed={}
    for r in selected:
        key=(r['case_id'],r['environment'],r['condition'],r['grouper'],r['history'],r['seed'])
        if key in indexed:raise ValueError('duplicate primary assignment')
        if r['n_truth']!=(1 if r['condition']=='false_report' else 0):raise ValueError('unexpected source-truth multiplicity')
        indexed[key]=r
    wanted=set(itertools.product(sorted(case_ids),ROUTES,CONDITIONS,ARMS,('stable','updated'),(0,1,2)))
    if set(indexed)!=wanted:raise ValueError('primary factorial arm coverage mismatch')
    output={}
    for route in ROUTES:
        cases=[]
        for cid in sorted(case_ids):
            arms={}
            for arm in ARMS:
                bycondition={condition:condition_arm([indexed[(cid,route,condition,arm,h,seed)] for h in ('stable','updated') for seed in (0,1,2)]) for condition in CONDITIONS}
                class_utilities=[bycondition[c]['utility'] for c in CONDITIONS]
                bycondition['equal_class_utility']={k:mean([u[k] for u in class_utilities]) for k in ('operational','lower','upper')}
                arms[arm]=bycondition
            a,b=[arms[x] for x in ARMS]
            # Same selections, histories and opportunities must remain paired.
            for condition in CONDITIONS:
                for h,seed in itertools.product(('stable','updated'),(0,1,2)):
                    l=indexed[(cid,route,condition,ARMS[0],h,seed)];r=indexed[(cid,route,condition,ARMS[1],h,seed)]
                    if l['selected_truth']!=r['selected_truth'] or l['reference_pair_available']!=r['reference_pair_available']:raise ValueError('unmatched primary acquisition')
            deltas={'recall':difference(a['false_report']['tp'],b['false_report']['tp']),
                    'clean_fp':difference(a['clean']['fp'],b['clean']['fp']),
                    'attacked_fp':difference(a['false_report']['fp'],b['false_report']['fp']),
                    'equal_class_utility':difference(a['equal_class_utility'],b['equal_class_utility'])}
            for cond,label in (('clean','clean'),('false_report','attacked')):
                for name in ('selected_fpr','network_fpr','report_fpr'):
                    deltas[label+'_'+name]=difference(a[cond][name],b[cond][name])
            mechanism={k:a['false_report'][k]-b['false_report'][k] for k in ('target_coverage','reference_pair_available','reference_pair_coobserved','rule_grouped_tp','rule_grouped_fp','rule_joined_tp','rule_joined_fp','weaker_message_supported_tp','strict_relational_supported_tp')}
            mechanism['llm_minus_rule_recall_delta']=deltas['recall']['operational']-mechanism['rule_grouped_tp']
            cases.append({'case_id':cid,'arms':arms,'adjacency_minus_exact_random':deltas,'mechanism_deltas':mechanism})
        keys=('recall','clean_fp','attacked_fp','equal_class_utility')+tuple(c+'_'+m for c in ('clean','attacked') for m in ('selected_fpr','network_fpr','report_fpr'))
        output[route]={'per_case':cases,'deltas':{k:{**sign_summary([c['adjacency_minus_exact_random'][k]['operational'] for c in cases]),
                        'mean_lower':mean([c['adjacency_minus_exact_random'][k]['lower'] for c in cases]),
                        'mean_upper':mean([c['adjacency_minus_exact_random'][k]['upper'] for c in cases]),
                        'strictly_positive_under_all_abstention_resolutions':sum(c['adjacency_minus_exact_random'][k]['lower']>1e-10 for c in cases),
                        'strictly_negative_under_all_abstention_resolutions':sum(c['adjacency_minus_exact_random'][k]['upper']< -1e-10 for c in cases)} for k in keys}}
    aligned={c['case_id']:c for c in output['aligned']['per_case']};shuffled={c['case_id']:c for c in output['shuffled']['per_case']}
    output['unconditional_alignment_interaction']={}
    for key in keys:
        values=[{'case_id':c,**difference(aligned[c]['adjacency_minus_exact_random'][key],shuffled[c]['adjacency_minus_exact_random'][key])} for c in sorted(case_ids)]
        output['unconditional_alignment_interaction'][key]={**sign_summary([v['operational'] for v in values]),'per_case':values,
            'mean_lower':mean([v['lower'] for v in values]),'mean_upper':mean([v['upper'] for v in values])}
    output['interpretation']='Positive recall/utility deltas favor adjacency; negative FP deltas favor adjacency. Bounds address missing judge decisions, not sampling uncertainty. Routing changes evidence availability; interpret interaction alongside rule/co-observation diagnostics, not as a pure judge effect.'
    return output


def citation_counts(rec,group,call):
    if 'abstention' in call:return {'strict_relational_supported_tp':0,'flagged_views':0,'flagged_views_with_valid_citations':0}
    ids=edge_ids(rec);selected=set(map(tuple,group));truth=set(map(tuple,rec['labels']['false_origin_edges']))
    d=call['judgment'];flags={e for e in selected if d['source_scores'][ids[e]]>.5}
    valid=d['source_evidence_valid'];supported=set();messages={m['id']:m for m in rec['messages']}
    if valid:
        for edge in flags&truth:
            cited=d['source_evidence'][ids[edge]]
            # Enforce shown, not merely existing, references independently.
            if any(mid not in messages or (messages[mid]['sender'],messages[mid]['recipient']) not in selected for mid in cited):raise ValueError('valid citation points outside shown group')
            for mid in set(cited)&set(rec['labels']['false_origin_messages']):
                m=messages[mid]
                if (m['sender'],m['recipient'])!=edge or m['trusted']:continue
                claim=parse_claim(m['content'])
                if claim is None:continue
                for ref_id in cited:
                    reference=messages[ref_id]
                    if not reference['trusted']:continue
                    ref=parse_claim(reference['content'])
                    if ref is not None and ref[:3]==claim[:3] and ref[3]!=claim[3]:supported.add(edge)
    return {'strict_relational_supported_tp':len(supported),'flagged_views':int(bool(flags)),
            'flagged_views_with_valid_citations':int(bool(flags) and valid)}


def enrich_citations(metrics,records,assignments,expectations,calls):
    indexed={r['id']:r for r in records};plans={(p['episode_id'],p['policy']):p for p in assignments+expectations}
    for row in metrics:
        p=plans[(row['episode_id'],row['policy'])];rec=indexed[row['episode_id']]
        row['network_negative']=len(rec['graph']['edges'])-row['n_truth']
        views=[(v['edges'],v['view_key'],v['weight']) for v in p['weighted_views']] if 'weighted_views' in p else [(g,k,1.) for g,k in zip(p['groups'],p['view_keys'])]
        counts=[(citation_counts(rec,g,calls[key]),weight) for g,key,weight in views]
        row.update({name:math.fsum(c[name]*w for c,w in counts) for name in ('strict_relational_supported_tp','flagged_views','flagged_views_with_valid_citations')})
    return metrics


def view_reliability(calls):
    buckets=defaultdict(list)
    for call in calls.values():buckets[len(call['edges'])].append(call)
    output=[]
    for size,group in sorted(buckets.items()):
        attempts=[]
        for call in group:
            if len(call['attempts'])!=call['request_count']:raise ValueError('attempt accounting mismatch')
            if ('judgment' in call)==('abstention' in call):raise ValueError('ambiguous view disposition')
            attempts.extend(call['attempts'])
        for attempt in attempts:
            if type(attempt.get('max_output_tokens')) is not int or attempt['max_output_tokens']<1:raise ValueError('missing output cap metadata')
        n=len(group);abstentions=sum('abstention' in c for c in group);retried=sum(c['request_count']>1 for c in group)
        flagged=[c for c in group if 'judgment' in c and any(v>.5 for v in c['judgment']['source_scores'].values())]
        output.append({'group_size':size,'unique_views':n,'judged_views':n-abstentions,'abstained_views':abstentions,
            'abstention_rate':abstentions/n,'retried_views':retried,'retry_rate':retried/n,
            'format_retry_attempts':sum(c['request_count']-1 for c in group),'total_attempts':len(attempts),
            'attempts_at_or_above_output_cap':sum(a['usage']['output_tokens']>=a['max_output_tokens'] for a in attempts),
            'output_caps':sorted({a['max_output_tokens'] for a in attempts}),
            'logical_input_tokens':sum(a['usage']['input_tokens'] for a in attempts),
            'logical_output_tokens':sum(a['usage']['output_tokens'] for a in attempts),
            'logical_replay_credits':math.fsum(a['usage']['cost'] for a in attempts),
            'flagged_views':len(flagged),'flagged_views_with_valid_citations':sum(c['judgment']['source_evidence_valid'] for c in flagged)})
    return {'rows':output,'scope':'Unique completed judge views, not policy-weighted assignments. Counts include schema retries; output-at-cap is a truncation diagnostic, not proof of causation. Logical replay charges may include cached responses and exclude uncertain transport failures; use the shared ledger for actual research spend.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('directory');args=p.parse_args();root=Path(args.directory)
    plan=json.loads((root/'plan.json').read_text());records=rows(root/'episodes.jsonl')
    assignments=rows(root/'assignments.jsonl');expectations=rows(root/'exact_random_plans.jsonl')
    if hashlib.sha256(canonical(assignments).encode()).hexdigest()!=plan['assignments_sha256']:raise ValueError('assignment hash mismatch')
    if hashlib.sha256(canonical(expectations).encode()).hexdigest()!=plan['expected_random_sha256']:raise ValueError('exact-random plan hash mismatch')
    if len(assignments)!=plan['assignments'] or len(expectations)!=plan['expected_random_assignments']:raise ValueError('assignment count mismatch')
    for name,want in plan['source_hashes'].items():
        if sha(root/'source'/name)!=want or sha(Path(__file__).parent/name)!=want:raise ValueError('frozen source mismatch: '+name)
    calls=require_coverage(plan,assignments,expectations,rows(root/'views.jsonl'))
    if hashlib.sha256(canonical(sorted(calls)).encode()).hexdigest()!=plan['view_keys_sha256']:raise ValueError('view library hash mismatch')
    case_ids={r['case_id'] for r in records}
    if len(case_ids)!=plan['case_clusters']:raise ValueError('case count mismatch')
    byid={r['id']:r for r in records}
    def choose(a):return primary({**a,**{k:byid[a['episode_id']][k] for k in ('workload_size','condition')}})
    chosen=[a for a in assignments if choose(a)];expected=[a for a in expectations if choose(a)]
    from record_workload_benchmark import analyze
    # Reuse exactly the frozen operational metrics; writes go only to temporary storage.
    with tempfile.TemporaryDirectory(prefix='jmon-supplement-') as temp:
        metrics=analyze(records,chosen,calls,Path(temp),expected)
    enrich_citations(metrics,records,chosen,expected,calls)
    result={'scope':'Post-launch exploratory supplemental audit added following independent review, before full aggregate unblinding. Original frozen analysis remains unchanged.',
            'coverage':{'complete':True,'unique_views':len(calls),'primary_rows':len(metrics)},
            'source_sha256':sha(__file__),'addendum_sha256':sha(Path(__file__).resolve().parents[1]/'research/workload_contrast_addendum.md'),
            'input_sha256':{name:sha(root/name) for name in ('plan.json','episodes.jsonl','assignments.jsonl','exact_random_plans.jsonl','views.jsonl')},
            'background_overlap':background_overlap(records),'unique_view_reliability':view_reliability(calls),'contrasts':paired_contrasts(metrics,case_ids),
            'citation_scope':'Stricter supplementary TP requires cited actual false-origin message and cited trusted reference with same case, field and revision but a different value, both shown. This is corroboration under the known grammar, not full rationale correctness.'}
    (root/'supplemental_contrast_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':str(root/'supplemental_contrast_audit.json'),'unique_views':len(calls),'primary_rows':len(metrics),'case_clusters':len(case_ids)},indent=2))

if __name__=='__main__':main()
