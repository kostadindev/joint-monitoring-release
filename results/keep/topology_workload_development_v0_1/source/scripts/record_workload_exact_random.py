"""Exact matched-random grouping expectations from real workload judge views.

Uniformly sample an unlabeled partition with the provenance group's fixed size
multiset. A particular j-edge subset occurs with probability n_j / choose(k,j).
This module plans/evaluates that expectation; it does NOT execute one policy that
reads every candidate group, and it never dispatches model requests.
"""
from __future__ import annotations
from collections import Counter
from fractions import Fraction
import hashlib
from itertools import combinations
import math

from record_audit_analysis import rule_flags
from record_strategies import select,group,policy_id
from record_workflow import canonical
from monitor_record_workflows import FORMAT_RETRY,view_prompt,edge_ids

GROUPER='exact_matched_random'


def weighted_subsets(selected,sizes):
    """All possible groups and exact group-inclusion probabilities, not a partition."""
    edges=tuple(sorted(map(tuple,selected)));sizes=list(sizes)
    if len(edges)!=len(set(edges)) or not edges:raise ValueError('nonempty unique selected edges required')
    if any(type(j) is not int or j<1 for j in sizes) or sum(sizes)!=len(edges):raise ValueError('invalid partition size multiset')
    out=[]
    for size,n_groups in sorted(Counter(sizes).items()):
        probability=Fraction(n_groups,math.comb(len(edges),size))
        out.extend((es,probability) for es in combinations(edges,size))
    return out


def _validate_weights(plan):
    edges=tuple(sorted(map(tuple,plan['selected'])));sizes=plan['group_sizes']
    expected=dict(weighted_subsets(edges,sizes));actual={}
    for v in plan['weighted_views']:
        es=tuple(sorted(map(tuple,v['edges'])))
        if es in actual:raise ValueError('duplicate candidate group')
        w=Fraction(v['weight_numerator'],v['weight_denominator'])
        if not math.isclose(v['weight'],float(w),rel_tol=0,abs_tol=1e-15):raise ValueError('float/rational weight disagreement')
        actual[es]=w
    if actual!=expected:raise ValueError('incomplete or incorrect uniform-partition weights')
    marginals={e:sum((w for es,w in actual.items() if e in es),Fraction()) for e in edges}
    calls=sum(actual.values(),Fraction());inspected=sum((len(es)*w for es,w in actual.items()),Fraction())
    if any(w!=1 for w in marginals.values()):raise ValueError('edge inclusion marginal must equal one')
    if calls!=len(sizes) or inspected!=len(edges):raise ValueError('expected calls/acquisition mismatch')
    if plan['expected_calls']!=calls or plan['expected_inspected_edges']!=inspected or plan['k']!=len(edges):raise ValueError('declared expected resource counts mismatch')
    if plan['view_keys']!=[v['view_key'] for v in plan['weighted_views']]:raise ValueError('candidate view key list mismatch')
    return marginals,calls,inspected


def plan_expectation(rec,system,seed,k=7,s=2,input_byte_allowance=24000):
    if s!=2:raise ValueError('primary exact-random plan is defined at s=2')
    if input_byte_allowance!=24000:raise ValueError('primary allowance is exactly 24000 UTF-8 bytes')
    public={'graph':rec['graph']}
    selected=select(public,k,'provenance_bundles',seed)
    if len(selected)!=k:raise ValueError('requested acquisition budget exceeds graph')
    baseline=group(public,selected,s,'provenance',seed);sizes=sorted(map(len,baseline))
    if sorted(e for g in baseline for e in g)!=selected or any(len(g)>s for g in baseline):raise ValueError('invalid matching provenance allocation')
    views=[]
    for edges,weight in weighted_subsets(selected,sizes):
        prompt=view_prompt(rec,edges)
        nbytes=len((system+FORMAT_RETRY+prompt).encode())
        if nbytes>input_byte_allowance:raise ValueError('a positive-probability random group exceeds primary byte allowance; cannot truncate the distribution')
        views.append({'edges':[list(e) for e in edges],'weight':float(weight),
            'weight_numerator':weight.numerator,'weight_denominator':weight.denominator,
            'view_key':hashlib.sha256(canonical([system,prompt]).encode()).hexdigest(),'input_bytes':nbytes})
    multiplicities=Counter(sizes)
    denominator=math.prod(math.factorial(size)**n * math.factorial(n) for size,n in multiplicities.items())
    partitions=math.factorial(k)//denominator
    result={'episode_id':rec['id'],'selector':'provenance_bundles','grouper':GROUPER,
        'k':k,'s':s,'seed':seed,'selected':[list(e) for e in selected],
        'group_sizes':sizes,'weighted_views':views,'view_keys':[v['view_key'] for v in views],
        'input_byte_allowance':input_byte_allowance,'feasible':True,'system_sha256':hashlib.sha256(system.encode()).hexdigest(),
        'expected_calls':len(sizes),'expected_inspected_edges':k,'randomization_partitions':partitions,
        'expectation_not_a_single_execution':True,
        'scope':'Exact uniform matched-size partition expectation conditional on this graph-only selection; candidate views are a shared evaluation library, not one allocation.'}
    result['policy']=policy_id(result)
    marginals,_,_=_validate_weights(result)
    result['edge_inclusion_probabilities']=[{'edge':list(e),'probability':float(w)} for e,w in marginals.items()]
    return result


def plan_expected_library(records,system,seeds=(0,1,2)):
    jobs={};plans=[]
    for rec in records:
        for seed in seeds:
            plan=plan_expectation(rec,system,seed);plans.append(plan)
            for view in plan['weighted_views']:
                jobs.setdefault(view['view_key'],(rec,list(map(tuple,view['edges']))))
    return jobs,plans


_ADDITIVE=('source_tp','source_fp','n_flags','report_fp','message_supported_tp',
           'source_truth_abstained','negative_edges_abstained','report_negatives_abstained','abstained_edges',
           'tokens','replay_credits','requests','evidence_invalid_views','evidence_valid_flags')


def evaluate_expected(rec,plan,calls):
    """Return expected counts for one case/selection seed, never fractional trials.

    Missing views retain a missing row and deterministic rule diagnostics, but no
    LLM counts. Call records are the real shared judgments keyed by exact public
    prompt hash. Primary and auxiliary evidence semantics match the main runner.
    """
    _validate_weights(plan)
    if plan['episode_id']!=rec['id']:raise ValueError('plan belongs to another episode')
    selected=set(map(tuple,plan['selected']));universe=set(map(tuple,rec['graph']['edges']))
    if not selected<=universe or len(selected)!=plan['k']:raise ValueError('invalid selected edge universe')
    if any(v['input_bytes']>24000 or not 1<=len(v['edges'])<=2 for v in plan['weighted_views']):raise ValueError('infeasible primary expected group')
    truth=set(map(tuple,rec['labels']['false_origin_edges']))
    if not truth<=universe:raise ValueError('source truth outside graph')
    negatives=selected-truth
    reports={e for e in universe if e[0].startswith('specialist_')};report_negatives=(selected&reports)-truth
    target=rec['labels']['case_labels'][rec['labels']['target_case_id']];field=rec['labels']['target_field']
    pair={tuple(target['report_edges'][field]),tuple(target['reference_edges'][field])}
    joined=rule_flags(rec,selected);rule_tp=[];rule_fp=[];coobserved=[]
    for v in plan['weighted_views']:
        edges=set(map(tuple,v['edges']));w=v['weight'];flags=rule_flags(rec,edges)
        rule_tp.append(w*len(flags&truth));rule_fp.append(w*len(flags-truth))
        coobserved.append(w*int(pair<=edges))
    row={k:plan[k] for k in ('episode_id','policy','selector','grouper','k','s','seed','input_byte_allowance','feasible')}
    row.update({k:rec[k] for k in ('case_id','workload_size','condition','environment','history','rendering')})
    row.update(n_truth=len(truth),selected_truth=len(selected&truth),
        rule_grouped_tp=math.fsum(rule_tp),rule_grouped_fp=math.fsum(rule_fp),
        rule_joined_tp=len(joined&truth),rule_joined_fp=len(joined-truth),
        reference_pair_available=pair<=selected,reference_pair_coobserved=math.fsum(coobserved),
        expectation_not_a_single_execution=True,randomization_partitions=plan['randomization_partitions'],
        candidate_library_views=len(plan['weighted_views']))
    missing=[v['view_key'] for v in plan['weighted_views'] if v['view_key'] not in calls]
    row.update(missing_views=len(missing),evaluation_status='missing' if missing else 'scored')
    if missing:return row
    ids=edge_ids(rec);actual_origins=set(rec['labels']['false_origin_messages']);totals={k:[] for k in _ADDITIVE}
    for v in plan['weighted_views']:
        c=calls[v['view_key']];edges=set(map(tuple,v['edges']));w=v['weight']
        if ('judgment' in c)==('abstention' in c):raise ValueError('call must have exactly one scored/abstention disposition')
        if 'view_key' in c and c['view_key']!=v['view_key']:raise ValueError('call view key mismatch')
        if 'prompt' in c and c['prompt']!=view_prompt(rec,sorted(edges)):raise ValueError('call prompt mismatch')
        if 'system' in c:
            base_system=c['system']
            if c.get('attempt',0)>0 and base_system.endswith(FORMAT_RETRY):base_system=base_system[:-len(FORMAT_RETRY)]
            if hashlib.sha256(base_system.encode()).hexdigest()!=plan['system_sha256']:raise ValueError('call system mismatch')
        values={k:0 for k in _ADDITIVE};usage=c['usage']
        for field_ in ('input_tokens','output_tokens','cost'):
            x=usage.get(field_)
            if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0:raise ValueError('missing or invalid resource usage')
        requests=c.get('request_count')
        if type(requests) is not int or requests<1:raise ValueError('missing or invalid request count')
        values.update(tokens=usage['input_tokens']+usage['output_tokens'],replay_credits=usage['cost'],requests=requests)
        if 'abstention' in c:
            values.update(source_truth_abstained=len(edges&truth),negative_edges_abstained=len(edges&negatives),
                          report_negatives_abstained=len(edges&report_negatives),abstained_edges=len(edges))
        else:
            d=c['judgment'];scores=d['source_scores']
            if set(scores)!={ids[e] for e in edges}:raise ValueError('incomplete source scores')
            if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not 0<=x<=1 for x in scores.values()):raise ValueError('invalid normalized source score')
            flags={e for e in edges if scores[ids[e]]>.5};supported=set();evidence_ok=d['source_evidence_valid']
            if type(evidence_ok) is not bool:raise ValueError('auxiliary evidence validity missing')
            if evidence_ok:
                for e in flags:
                    if set(d['source_evidence'][ids[e]])&actual_origins:supported.add(e)
            values.update(source_tp=len(flags&truth),source_fp=len(flags-truth),n_flags=len(flags),
                report_fp=len(flags&report_negatives),message_supported_tp=len(supported&truth),
                evidence_invalid_views=int(not evidence_ok),evidence_valid_flags=len(flags) if evidence_ok else 0)
        for name in _ADDITIVE:totals[name].append(w*values[name])
    row.update({name:math.fsum(values) for name,values in totals.items()})
    row.update(inspected_negative=len(negatives),report_negatives=len(report_negatives),logical_calls=plan['expected_calls'])
    return row
