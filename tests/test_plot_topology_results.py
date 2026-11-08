"""Synthetic extraction fixtures only: never render invented results."""
from pathlib import Path
import copy
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pytest
from plot_topology_results import (LLM_ROWS,REFERENCE_ROWS,figure_data,verified_analysis,render)


def synthetic_summary():
    rows=[]
    for route in ('aligned','shuffled'):
        for spec in LLM_ROWS+REFERENCE_ROWS:
            identity,label,selector,grouper,k,s,allowance=spec
            for condition in ('clean','false_report'):
                infeasible=identity in ('selected24','network24')
                recall=.7 if identity=='path' else .4 if identity=='exact' else .5
                rows.append({'workload_size':16,'environment':route,'condition':condition,
                    'selector':selector,'grouper':grouper,'k':k,'s':s,'input_byte_allowance':allowance,
                    'case_clusters':12,'missing':0,'infeasible':72 if infeasible else 0,'assignments':72,
                    'information_access':['anonymous directed topology only'],'llm_performance_defined':not infeasible,
                    'source_recall':recall if condition=='false_report' else None,
                    'source_recall_upper':recall+.1 if condition=='false_report' else None,
                    'source_fpr':.02 if identity=='path' else .03,'source_fpr_upper':.05,
                    'mean_calls':4 if condition=='clean' else 5,'mean_tokens':1000 if condition=='clean' else 3000,
                    'rule_grouped_recall':.5 if condition=='false_report' else None,
                    'rule_joined_recall':.8 if condition=='false_report' else None,'rule_grouped_fp':7,'rule_joined_fp':14})
    contrast={route:{'per_case':[{'case_id':f'c{i}'} for i in range(12)],'deltas':{
        'recall':{'mean':.3,'mean_lower':.2,'mean_upper':.4},
        'clean_selected_fpr':{'mean':-.01,'mean_lower':-.03,'mean_upper':.02}}} for route in ('aligned','shuffled')}
    return rows,contrast


def test_fixed_filters_matched_costs_bounds_and_different_budget_references():
    rows,contrast=synthetic_summary()
    # A similarly named extra budget or small workload cannot enter main rows.
    decoys=[]
    for row in rows[:2]:
        for key,value in [('workload_size',4),('k',4),('input_byte_allowance',60000)]:
            other=copy.deepcopy(row);other[key]=value;other['mean_tokens']=999999;decoys.append(other)
    data=figure_data(rows+decoys,contrast);panel=data['panels']['aligned']['rows']
    assert [r['id'] for r in panel[:5]]==['path','exact','random','mesa','learned']
    assert all((r['k'],r['s'],r['input_byte_allowance'])==(7,2,24000) for r in panel[:5])
    path=panel[0];assert path['mean_calls']==4.5 and path['mean_tokens']==2000
    assert path['recall']=={'value':.7,'lower':.7,'upper':pytest.approx(.8)}
    refs={r['id']:r for r in panel if r['id'] in ('selected24','selected60','network24','network60')}
    assert refs['network24']['defined'] is False and refs['network24']['recall'] is None
    assert refs['network24']['mean_calls'] is None and refs['network24']['mean_tokens'] is None
    assert refs['network60']['defined'] and (refs['network60']['k'],refs['network60']['s'])==(13,13)
    assert refs['selected60']['k']==7 and refs['selected60']['input_byte_allowance']==60000


def test_deterministic_detector_uses_clean_benign_denominator_and_not_llm_cost():
    rows,contrast=synthetic_summary();panel=figure_data(rows,contrast)['panels']['aligned']['rows']
    grouped=next(r for r in panel if r['id']=='rule_grouped');joined=next(r for r in panel if r['id']=='rule_joined')
    assert grouped['fpr']['value']==pytest.approx(7/(7*72))
    assert joined['fpr']['value']==pytest.approx(14/(7*72))
    assert grouped['mean_calls']==joined['mean_calls']==0 and grouped['mean_tokens']==0
    assert grouped['s']==2 and joined['s']==7 and 'pooled join' in joined['grouper']
    assert grouped['recall']['value']==.5 and joined['recall']['value']==.8


def test_missing_policy_inconsistent_primary_and_non_topology_inputs_rejected():
    rows,contrast=synthetic_summary()
    with pytest.raises(ValueError,match='missing requested policy'):figure_data(rows[1:],contrast)
    bad=copy.deepcopy(contrast);bad['aligned']['deltas']['recall']['mean']=.35
    with pytest.raises(ValueError,match='disagrees'):figure_data(rows,bad)
    bad=copy.deepcopy(rows);bad[0]['information_access']=['roles and messages']
    with pytest.raises(ValueError,match='non-topology'):figure_data(bad,contrast)
    with pytest.raises(ValueError,match='duplicate'):figure_data(rows+[rows[0]],contrast)


def test_undefined_main_policy_cannot_be_silently_zero_filled():
    rows,contrast=synthetic_summary();rows[0]['infeasible']=1;rows[0]['llm_performance_defined']=False
    with pytest.raises(ValueError,match='fully feasible'):figure_data(rows,contrast)


def test_refuses_incomplete_or_missing_analysis_without_creating_figure_directory(tmp_path):
    root=tmp_path/'topology_workload_test';root.mkdir();out=tmp_path/'figures'
    with pytest.raises(ValueError,match='complete analysis manifest'):render(root,out)
    assert not out.exists()
    (root/'analysis').mkdir();(root/'analysis/manifest.json').write_text(json.dumps({'coverage':{'complete':False}}))
    with pytest.raises(ValueError,match='incomplete'):render(root,out)
    assert not out.exists()


def test_analysis_hash_gate_binds_summary_to_frozen_inputs_and_sources(tmp_path):
    root=tmp_path/'topology_workload_test';analysis=root/'analysis';analysis.mkdir(parents=True)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    source='scripts/minimal.py';(root/'source/scripts').mkdir(parents=True);(root/'source'/source).write_text('# fixture\n')
    (analysis/'source/scripts').mkdir(parents=True);(analysis/'source'/source).write_text('# analysis fixture\n')
    plan={'unique_views':2,'assignments':3,'expectations':1,'source_hashes':{source:sha(root/'source'/source)}}
    (root/'plan.json').write_text(json.dumps(plan));(root/'coverage.json').write_text(json.dumps({'complete':True,'unique_planned_views':2,'unique_scored_views':2}))
    for name in ('episodes.jsonl','assignments.jsonl','exact_random_plans.jsonl','views.jsonl'):(root/name).write_text('{}\n')
    rows,contrast=synthetic_summary();(analysis/'summary.json').write_text(json.dumps({'rows':rows}));(analysis/'primary_contrast.json').write_text(json.dumps(contrast))
    manifest={'run_name':root.name,'coverage':{'complete':True,'unique_views':2,'assignments':4,'case_clusters':12},
        'inputs':{name:sha(root/name) for name in ('plan.json','episodes.jsonl','assignments.jsonl','exact_random_plans.jsonl','views.jsonl')},
        'output_sha256':{name:sha(analysis/name) for name in ('summary.json','primary_contrast.json')},
        'analysis_source_sha256':{source:sha(analysis/'source'/source)},'execution_source_sha256':plan['source_hashes']}
    (analysis/'manifest.json').write_text(json.dumps(manifest))
    assert verified_analysis(root)[0]==rows
    (analysis/'summary.json').write_text('{}')
    with pytest.raises(ValueError,match='output hash'):verified_analysis(root)
    (analysis/'summary.json').write_text(json.dumps({'rows':rows}));(root/'views.jsonl').write_text('changed\n')
    with pytest.raises(ValueError,match='stale analysis input'):verified_analysis(root)
