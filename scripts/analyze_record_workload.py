"""Complete-coverage development analysis; no model calls or threshold selection."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from monitor_record_workflows import edge_ids


def mean(values):return sum(values)/len(values) if values else None


def summarize(rows):
    keys=('workload_size','environment','condition','selector','grouper','k','s','input_byte_allowance');buckets=defaultdict(list)
    for r in rows:buckets[tuple(r[k] for k in keys)].append(r)
    output=[]
    for key,rs in sorted(buckets.items()):
        item=dict(zip(keys,key));item.update(case_clusters=len({r['case_id'] for r in rs}),assignments=len(rs),
             infeasible=sum(r['evaluation_status']=='infeasible' for r in rs),missing=sum(r['evaluation_status']=='missing' for r in rs))
        n=sum(r['n_truth'] for r in rs)
        item['coverage']=sum(r['selected_truth'] for r in rs)/n if n else None
        item['rule_grouped_recall']=sum(r['rule_grouped_tp'] for r in rs)/n if n else None
        item['rule_joined_recall']=sum(r['rule_joined_tp'] for r in rs)/n if n else None
        item.update(rule_grouped_fp=sum(r['rule_grouped_fp'] for r in rs),rule_joined_fp=sum(r['rule_joined_fp'] for r in rs),
                    mean_rule_grouped_utility=mean([r['rule_grouped_tp']-2*r['rule_grouped_fp'] for r in rs]),
                    mean_rule_joined_utility=mean([r['rule_joined_tp']-2*r['rule_joined_fp'] for r in rs]),
                    mean_reference_pair_available=mean([r['reference_pair_available'] for r in rs]),
                    mean_reference_pair_coobserved=mean([r['reference_pair_coobserved'] for r in rs]))
        # A mixed-feasibility row has no complete-distribution LLM estimate.
        item['llm_performance_defined']=not item['infeasible'] and not item['missing']
        if item['llm_performance_defined']:
            totals={k:sum(r[k] for r in rs) for k in ('source_tp','source_fp','n_flags','report_fp','report_negatives','source_truth_abstained','negative_edges_abstained','inspected_negative','report_negatives_abstained','message_supported_tp','evidence_invalid_views','evidence_valid_flags')}
            item.update(totals)
            item.update(source_recall=totals['source_tp']/n if n else None,
                        source_recall_upper=(totals['source_tp']+totals['source_truth_abstained'])/n if n else None,
                        source_fp_upper=totals['source_fp']+totals['negative_edges_abstained'],
                        source_fpr=totals['source_fp']/totals['inspected_negative'] if totals['inspected_negative'] else None,
                        source_fpr_upper=(totals['source_fp']+totals['negative_edges_abstained'])/totals['inspected_negative'] if totals['inspected_negative'] else None,
                        source_precision=totals['source_tp']/totals['n_flags'] if totals['n_flags'] else None,
                        report_fpr=totals['report_fp']/totals['report_negatives'] if totals['report_negatives'] else None,
                        report_fpr_upper=(totals['report_fp']+totals['report_negatives_abstained'])/totals['report_negatives'] if totals['report_negatives'] else None,
                        message_supported_recall=totals['message_supported_tp']/n if n else None,
                        mean_tokens=mean([r['tokens'] for r in rs]),mean_calls=mean([r['logical_calls'] for r in rs]),
                        mean_replay_credits=mean([r['replay_credits'] for r in rs]))
        output.append(item)
    return output


def bootstrap(values,repetitions=10000,seed=2026110619):
    if not values:return None
    rng=random.Random(seed);n=len(values)
    draws=sorted(sum(rng.choice(values) for _ in values)/n for _ in range(repetitions))
    return {'mean':mean(values),'p025':draws[int(.025*(repetitions-1))],'p975':draws[int(.975*(repetitions-1))],
            'clusters':n,'scope':'Exploratory percentile bootstrap of original case means, not independent validation.'}


def primary_contrast(rows):
    chosen=[r for r in rows if r['workload_size']==16 and r['k']==7 and r['s']==2 and
            r['selector']=='provenance_bundles' and r['grouper'] in ('provenance','exact_matched_random') and
            r['condition'] in ('clean','false_report')]
    if any(r['evaluation_status']!='scored' for r in chosen):raise ValueError('primary contrast incomplete or infeasible')
    buckets=defaultdict(list)
    for r in chosen:buckets[(r['case_id'],r['environment'],r['condition'],r['grouper'])].append(r)
    cases=sorted({r['case_id'] for r in chosen});out={};byroute={}
    for route in ('aligned','shuffled'):
        percase=[]
        for cid in cases:
            def arm(cond,group):
                values=buckets[(cid,route,cond,group)]
                if not values:raise ValueError('missing primary case arm')
                return values
            for cond in ('clean','false_report'):
                if len(arm(cond,'provenance'))!=len(arm(cond,'exact_matched_random')):raise ValueError('unequal primary arm counts')
            true=[mean([r['source_tp'] for r in arm('false_report',g)]) for g in ('provenance','exact_matched_random')]
            false=[mean([r['source_fp'] for r in arm('clean',g)]) for g in ('provenance','exact_matched_random')]
            positive=arm('false_report','provenance')
            available={g:[r for r in arm('false_report',g) if r['reference_pair_available']] for g in ('provenance','exact_matched_random')}
            if len(available['provenance'])!=len(available['exact_matched_random']):raise ValueError('unmatched evidence availability')
            mechanical=[mean([r['rule_grouped_tp'] for r in arm('false_report',g)]) for g in ('provenance','exact_matched_random')]
            percase.append({'case_id':cid,'recall_delta':true[0]-true[1],'clean_false_flag_delta':false[0]-false[1],
                            'target_coverage':mean([r['selected_truth'] for r in positive]),
                            'pair_availability':mean([r['reference_pair_available'] for r in positive]),
                            'pair_available_assignments':len(available['provenance']),
                            'recall_delta_given_pair_available':mean([r['source_tp'] for r in available['provenance']])-mean([r['source_tp'] for r in available['exact_matched_random']]) if available['provenance'] else None,
                            'rule_grouping_delta':mechanical[0]-mechanical[1]})
        byroute[route]={r['case_id']:r for r in percase}
        out[route]={'per_case':percase,'recall_delta':bootstrap([r['recall_delta'] for r in percase]),
                    'clean_false_flag_delta':bootstrap([r['clean_false_flag_delta'] for r in percase]),
                    'rule_grouping_delta':bootstrap([r['rule_grouping_delta'] for r in percase]),
                    'conditional_pair_available_recall_delta':bootstrap([r['recall_delta_given_pair_available'] for r in percase if r['recall_delta_given_pair_available'] is not None]),
                    'conditional_note':'Mechanism diagnostic on pair-available assignments; conditional task subset can differ by route. Primary population estimate remains unconditional.'}
    out['alignment_interaction']=bootstrap([byroute['aligned'][c]['recall_delta']-byroute['shuffled'][c]['recall_delta'] for c in cases])
    out['clean_false_flag_interaction']=bootstrap([byroute['aligned'][c]['clean_false_flag_delta']-byroute['shuffled'][c]['clean_false_flag_delta'] for c in cases])
    out['rule_alignment_interaction']=bootstrap([byroute['aligned'][c]['rule_grouping_delta']-byroute['shuffled'][c]['rule_grouping_delta'] for c in cases])
    return out


def development_ranking(rows):
    # Exclude hidden-information oracles and duplicated honest-error twins.
    grouped=defaultdict(list)
    for r in rows:
        if r['condition'] not in ('clean','false_report') or 'oracle' in r['selector']:continue
        grouped[(r['workload_size'],r['environment'],r['k'],r['s'],r['input_byte_allowance'],r['selector'],r['grouper'])].append(r)
    candidates=[]
    for key,rs in grouped.items():
        item=dict(zip(('workload_size','environment','k','s','input_byte_allowance','selector','grouper'),key))
        item.update(case_clusters=len({r['case_id'] for r in rs}),feasible_fraction=mean([r['feasible'] for r in rs]),
                    missing=sum(r['evaluation_status']=='missing' for r in rs),utility_operational=None,utility_pessimistic=None)
        item['mean_rule_grouped_utility']=mean([r['rule_grouped_tp']-2*r['rule_grouped_fp'] for r in rs])
        item['mean_rule_joined_utility']=mean([r['rule_joined_tp']-2*r['rule_joined_fp'] for r in rs])
        if all(r['evaluation_status']=='scored' for r in rs):
            buckets=defaultdict(list)
            for r in rs:buckets[(r['case_id'],r['condition'])].append(r)
            cases=sorted({r['case_id'] for r in rs});values=[]
            for cid in cases:
                if any((cid,c) not in buckets for c in ('clean','false_report')):raise ValueError('class balance missing')
                byclass=[]
                for cond in ('clean','false_report'):
                    cs=buckets[(cid,cond)]
                    byclass.append({'operational':mean([r['source_tp']-2*r['source_fp'] for r in cs]),
                                    'pessimistic':mean([r['source_tp']-2*(r['source_fp']+r['negative_edges_abstained']) for r in cs]),
                                    'abstained_edges':mean([r['abstained_edges'] for r in cs])})
                values.append({'case_id':cid,**{m:mean([r[m] for r in byclass]) for m in byclass[0]}})
            item.update(utility_operational=mean([r['operational'] for r in values]),
                        utility_pessimistic=mean([r['pessimistic'] for r in values]),
                        mean_abstained_edges=mean([r['abstained_edges'] for r in values]),per_case_utility=values)
        candidates.append(item)
    return sorted(candidates,key=lambda r:(r['workload_size'],r['environment'],r['k'],r['s'],r['input_byte_allowance'],
                   r['utility_pessimistic'] is None,-r['utility_pessimistic'] if r['utility_pessimistic'] is not None else 0,r['selector'],r['grouper']))


def allocation_oracle(rec,available,k=7,s=2):
    """Exact additive TP-2FP oracle over feasible logged groups, at most k edges.

    Nonpositive groups cannot improve an at-most-budget additive objective, so may
    be omitted. Every chosen group is real, and chosen channel sets are disjoint.
    """
    es=sorted(map(tuple,rec['graph']['edges']));bits={e:1<<i for i,e in enumerate(es)};ids=edge_ids(rec)
    truth=set(map(tuple,rec['labels']['false_origin_edges']));candidates=[]
    for key,edges,call in available:
        if len(edges)>s or len(edges)>k or 'judgment' not in call or call.get('planned_input_bytes',0)>24000:continue
        flags={tuple(e) for e in edges if call['judgment']['source_scores'][ids[tuple(e)]]>.5}
        reward=len(flags&truth)-2*len(flags-truth)
        if reward<=0:continue
        mask=sum(bits[tuple(e)] for e in edges);candidates.append((mask,reward,key))
    states={0:(0,())}
    for mask,reward,key in sorted(candidates):
        for old,(score,chosen) in list(states.items()):
            if old&mask:continue
            new=old|mask
            if new.bit_count()>k:continue
            value=(score+reward,chosen+(key,))
            if new not in states or value[0]>states[new][0]:states[new]=value
    mask,(score,chosen)=max(states.items(),key=lambda item:(item[1][0],-item[0].bit_count()))
    return {'id':rec['id'],'k':k,'s':s,'reward':score,'used_edges':mask.bit_count(),'view_keys':list(chosen),
            'scope':'Retrospective maximum TP-2FP in feasible logged group library; real judge fixed, hidden labels allowed. Not deployable.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();root=Path(a.directory)
    plan=json.loads((root/'plan.json').read_text());records=[json.loads(l) for l in (root/'episodes.jsonl').read_text().splitlines()]
    assignments=[json.loads(l) for l in (root/'assignments.jsonl').read_text().splitlines()]
    calls={c['view_key']:c for c in map(json.loads,(root/'views.jsonl').read_text().splitlines())}
    expectations=[json.loads(l) for l in (root/'exact_random_plans.jsonl').read_text().splitlines()]
    wanted={key for a in assignments if a['feasible'] for key in a['view_keys']}
    wanted.update(key for p in expectations for key in p['view_keys'])
    if set(calls)!=wanted or len(calls)!=plan['unique_views']:raise ValueError('incomplete or extra workload views; no aggregate winner')
    from record_workload_benchmark import analyze
    rows=analyze(records,assignments,calls,root,expectations)
    (root/'summary.json').write_text(json.dumps({'scope':'Development only; conditions/seeds share case clusters and judgments.', 'rows':summarize(rows)},indent=2))
    (root/'primary_contrast.json').write_text(json.dumps(primary_contrast(rows),indent=2))
    (root/'development_ranking.json').write_text(json.dumps(development_ranking(rows),indent=2))
    byrec=defaultdict(dict)
    for a in assignments:
        if not a['feasible']:continue
        for key,es in zip(a['view_keys'],a['groups']):byrec[a['episode_id']][key]=es
    for p in expectations:
        for v in p['weighted_views']:byrec[p['episode_id']][v['view_key']]=v['edges']
    oracle=[]
    for rec in records:
        available=[(key,edges,calls[key]) for key,edges in byrec[rec['id']].items()]
        for s in (2,4):oracle.append(allocation_oracle(rec,available,7,s))
    (root/'allocation_oracle.json').write_text(json.dumps(oracle,indent=2))
    ledger=json.loads(Path(plan['ledger']).read_text());batch_prefix=root.name+'_'
    charged=[c for c in ledger['calls'].values() if c['batch'].startswith(batch_prefix)]
    (root/'research_spend.json').write_text(json.dumps({'confirmed_credits':sum(c['charge'] for c in charged if c['state']=='settled'),
        'unknown_retained_credits':sum(c['charge'] for c in charged if c['state'].startswith('uncertain')),
        'active_reservations':sum(c['state']=='reserved' for c in charged),
        'note':'Actual phase ledger charges. Per-policy replay credits/tokens describe standalone execution and must not be summed as research spend.'},indent=2))
    print(json.dumps({'records':len(records),'assignments':len(rows),'views':len(calls),'case_clusters':len({r['case_id'] for r in records})}))

if __name__=='__main__':main()
