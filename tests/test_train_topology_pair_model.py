import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import train_topology_pair_model as training
from topology_only_policies import FEATURES,pair_features
from record_allocation import view_key
from monitor_record_workflows import edge_ids


def fixture():
    records=[];indexed={}
    for cid,count in [('case_a',2),('case_b',1)]:
        for i in range(count):
            rec={'id':cid+str(i),'case_id':cid,'split':'development',
                 'graph':{'nodes':['registry','reporter','reviewer'],'edges':[['registry','reporter'],['reporter','reviewer']],
                          'roles':{'registry':'secret trust annotation','reporter':'semantic field'}},
                 'labels':{'direct_false_edges':[['reporter','reviewer']] if cid=='case_a' else [],
                           'propagated_false_edges':[],'benign_error_edges':[],'incorrect_outcome_edges':[]},
                 'messages':[{'private_payload':'must not enter learner'}]}
            records.append(rec);indexed[rec['id']]=calls_for(rec)
    return records,indexed


def calls_for(rec):
    truth=set(map(tuple,rec['labels']['direct_false_edges']));ids=edge_ids(rec)
    return {view_key(group):{'judgment':{'source_scores':{ids[e]:float(e in truth) for e in group}}} for group in training.small_groups(rec)}


def test_learner_boundary_has_only_anonymous_topology_and_numerical_targets(monkeypatch):
    records,indexed=fixture();received=[]
    original=training.fit_pair_model
    def spy(examples,alpha):
        received.extend(examples)
        assert alpha==.01
        assert all(set(e)=={'graph','group','reward','weight'} for e in examples)
        assert all(set(e['graph'])=={'nodes','edges'} for e in examples)
        assert all(type(v) is int for e in examples for edge in e['graph']['edges'] for v in edge)
        assert 'registry' not in json.dumps(examples) and 'private_payload' not in json.dumps(examples)
        return original(examples,alpha)
    monkeypatch.setattr(training,'fit_pair_model',spy)
    model,audit=training.train(records,indexed)
    assert received and set(model)=={'features','coef','alpha','supervision'}
    assert model['features']==list(FEATURES) and model['alpha']==.01
    assert audit['case_total_weights']==pytest.approx({'case_a':.5,'case_b':.5})
    assert audit['episode_total_weights']==pytest.approx({'case_a0':.25,'case_a1':.25,'case_b0':.5})


def test_abstention_penalizes_only_selected_benign_source_edges_with_no_tp_credit():
    records,indexed=fixture();rec=records[0];both=list(map(tuple,rec['graph']['edges']))
    assert training.reward(rec,both,{'abstention':{}})==-2
    assert training.reward(rec,[('reporter','reviewer')],{'abstention':{}})==0
    assert training.reward(records[-1],both,{'abstention':{}})==-4
    call={'judgment':{'source_scores':{v:1. for v in edge_ids(rec).values()}}}
    assert training.reward(rec,both,call)==-1


def test_semantic_metadata_and_actor_renaming_do_not_change_fit():
    records,indexed=fixture();model,_=training.train(records,indexed);changed=copy.deepcopy(records)
    mapping={'registry':'z','reporter':'x','reviewer':'a'}
    for r in changed:
        r['graph']['nodes']=[mapping[n] for n in reversed(r['graph']['nodes'])]
        r['graph']['edges']=[[mapping[u],mapping[v]] for u,v in r['graph']['edges']]
        r['graph']['roles']={'poison':'semantic decoy'};r['graph']['trusted']={'poison':True}
        r['messages']=[{'unselected_payload':'different'}]
        r['labels']['direct_false_edges']=[[mapping[u],mapping[v]] for u,v in r['labels']['direct_false_edges']]
    altered,_=training.train(changed,{r['id']:calls_for(r) for r in changed})
    assert np.allclose(model['coef'],altered['coef'],atol=1e-12,rtol=1e-12)


def test_missing_pair_is_not_silently_removed_and_non_development_is_rejected():
    records,indexed=fixture();indexed[records[0]['id']].pop(view_key(training.small_groups(records[0])[-1]))
    with pytest.raises(ValueError,match='incomplete'):training.train(records,indexed)
    records,indexed=fixture();records[0]['split']='held_out'
    with pytest.raises(ValueError,match='development'):training.train(records,indexed)


def test_overlap_check_includes_background_cases_not_only_target_ids():
    workload=[{'case_id':'new_target','labels':{'case_ids':['new_target','case_a']}}]
    with pytest.raises(ValueError,match='overlap'):training.require_disjoint(['case_a'],workload)
    result=training.require_disjoint(['old_case'],workload)
    assert result['overlap']==[] and result['workload_all_case_ids']==2


def test_source_score_threshold_is_strict_and_invalid_scores_raise():
    records,_=fixture();rec=records[0];edge=('reporter','reviewer');cid=edge_ids(rec)[edge]
    assert training.reward(rec,[edge],{'judgment':{'source_scores':{cid:.5}}})==0
    assert training.reward(rec,[edge],{'judgment':{'source_scores':{cid:.5001}}})==1
    for value in (True,float('nan'),-1,2):
        with pytest.raises(ValueError):training.reward(rec,[edge],{'judgment':{'source_scores':{cid:value}}})
