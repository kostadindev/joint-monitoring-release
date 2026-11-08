from pathlib import Path
import copy,json,sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze_topology_causal_results as a


def row(case='c',seed=0,grouper='directed_path'):
    d=dict(episode_id=case,case_id=case,environment='flat',condition='combined',selector='structural_path_pairs',grouper=grouper,policy=f'{case}:{grouper}:{seed}',seed=seed,k=7,s=2,input_byte_allowance=24000,
        status='scored',metrics_available=True,n_truth=2,n_known_benign=9,n_unknown=2,selected_truth=2,selected_known_benign=4,selected_unknown=1,
        tp=1.,fp=1.,flagged_unknown=1.,abstained_known_positive=0.,abstained_known_benign=0.,abstained_unknown=0.,abstained_channels=0.,
        source_any_recovery=1.,source_both_recovery=0.,source_all_recovery=0.,source_any_recovery_upper=1.,source_both_recovery_upper=0.,source_all_recovery_upper=0.,
        input_tokens=100,output_tokens=50,replay_credits=.01,requests=4,judge_calls=4,abstention_calls=0)
    d['record_matching_within_source_metrics']={k:v for k,v in d.items() if k in ('tp','fp','flagged_unknown','abstained_known_positive','abstained_known_benign','abstained_unknown','abstained_channels','source_any_recovery','source_both_recovery','source_all_recovery','source_any_recovery_upper','source_both_recovery_upper','source_all_recovery_upper')}
    d['record_matching_pooled_source_metrics']=copy.deepcopy(d['record_matching_within_source_metrics'])
    return d


def test_partial_run_rejected_before_metrics_or_output(tmp_path):
    run=tmp_path/'run';run.mkdir();(run/'plan.json').write_text(json.dumps({'unique_views':1415}));(run/'coverage.json').write_text(json.dumps({'complete':False,'completed_views':1414}))
    with pytest.raises(ValueError,match='incomplete'):a.analyze_complete(run,tmp_path/'report')
    assert not (tmp_path/'report').exists()


def test_gate_rejects_lying_completion_and_missing_metric_row():
    record={'id':'c','graph':{'edges':[[0,1]]},'labels':{'false_origin_edges':[[0,1]],'known_benign_source_edges':[],'unknown_source_edges':[]}}
    plan={'unique_views':1,'view_keys_sha256':a.digest(['v']),'episodes':1}
    assignment={'episode_id':'c','policy':'p','feasible':True,'view_keys':['v']}
    metric={'episode_id':'c','policy':'p','status':'scored','metrics_available':True}
    with pytest.raises(ValueError,match='coverage'):a.require_complete(plan,[record],[assignment],[],{},[metric])
    with pytest.raises(ValueError,match='metrics'):a.require_complete(plan,[record],[assignment],[],{'v':{}},[])
    a.require_complete(plan,[record],[assignment],[],{'v':{}},[metric])


def test_unknown_labels_not_benign_any_both_not_reconstructed_from_tp():
    result=a.one_case([row()], 'judge')
    assert result['source_recall']==.5 and result['selected_known_benign_fpr']==.25
    assert result['network_known_benign_fpr']==pytest.approx(1/9)
    assert result['selected_unknown_fraction']==pytest.approx(1/7)
    assert result['mean_source_any_recovery']==1 and result['mean_source_both_recovery']==0
    assert result['unknown_label_pessimistic_utility']==-3
    # Exact random both is an independently enumerated probability, not a
    # function of expected TP/2 or independent marginal recovery probabilities.
    r=row();r.update(tp=1,source_any_recovery=.5,source_both_recovery=.5,source_all_recovery=.5)
    result=a.one_case([r],'judge');assert result['mean_source_any_recovery']==.5 and result['mean_source_both_recovery']==.5


def test_seeds_average_inside_case_and_twins_remain_separate():
    rows=[row('a',i) for i in range(3)]+[row('b',0)]
    rows[-1].update(tp=0.,fp=0.)
    twin=copy.deepcopy(rows[0]);twin.update(episode_id='twin',condition='honest_error_combined',policy='twin');rows.append(twin)
    plans={(r['episode_id'],r['policy']):{'comparison_scope':'prospective_topology_only'} for r in rows}
    summary=a.summary_rows(rows,plans);combined=next(r for r in summary if r['condition']=='combined' and r['method']=='judge')
    assert combined['case_clusters']==2 and combined['assignment_rows']==4
    assert combined['source_recall']==.25 # not seed-weighted .375
    assert len([r for r in summary if r['condition']=='honest_error_combined'])==3


def test_structured_unassessed_is_not_zero_false_alarm_success():
    r=row();r['record_matching_within_source_metrics'].update(tp=0.,fp=0.,abstained_known_positive=2,abstained_known_benign=4,abstained_unknown=1,abstained_channels=7)
    result=a.one_case([r],'structured_within')
    assert result['decision_unassessed_fraction']==1
    assert result['source_recall']==0 and result['source_recall_upper']==1
    assert result['selected_known_benign_fpr_upper']==1 and result['mean_judge_calls']==0


def test_grouping_contrast_requires_same_selected_edges_and_sizes():
    left=row();right=row(grouper=a.ARMS[1]);plans={}
    for r in (left,right):plans[(r['episode_id'],r['policy'])]={'selected':[(i,i+1) for i in range(7)],'groups':[[(0,1)],[(1,2),(2,3)],[(3,4),(4,5)],[(5,6),(6,7)]],'group_sizes':[1,2,2,2]}
    assert a.grouping_contrasts([left,right],plans)['aggregate'][0]['deltas']['source_recall']['mean']==0
    plans[(right['episode_id'],right['policy'])]['selected'][0]=(9,10)
    with pytest.raises(ValueError,match='changes acquisition'):a.grouping_contrasts([left,right],plans)


def test_actual_cost_counts_unique_http_not_policy_replays(tmp_path):
    requests=tmp_path/'requests';requests.mkdir();(requests/'r.json').write_text(json.dumps({'ledger_request_id':'x','state':'complete','usage':{'cost':.003}}))
    ledger={'calls':{'x':{'state':'settled','charge':.003,'batch':'phase'},'unrelated':{'state':'settled','charge':5,'batch':'other'}}}
    assert a.cost_audit(tmp_path,ledger)['settled_cost']==.003
    (requests/'duplicate.json').write_text((requests/'r.json').read_text())
    with pytest.raises(ValueError,match='duplicate'):a.cost_audit(tmp_path,ledger)


def test_release_mode_rejects_tampered_candidate_before_verifier_or_outputs(tmp_path,monkeypatch):
    candidate=tmp_path/'candidate';candidate.mkdir();(candidate/'data.json').write_text('{}')
    (candidate/'manifest.json').write_text(json.dumps({'files_sha256':{'data.json':'wrong'}}))
    monkeypatch.setattr(a.subprocess,'run',lambda *args,**kwargs:pytest.fail('ran unverified candidate code'))
    with pytest.raises(ValueError,match='hash/path'):a.from_release(candidate,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_release_mode_requires_successful_bundled_verifier(tmp_path,monkeypatch):
    candidate=tmp_path/'candidate';candidate.mkdir();(candidate/'manifest.json').write_text(json.dumps({'files_sha256':{}}))
    class Rejected:
        returncode=1;stderr='candidate incomplete';stdout=''
    monkeypatch.setattr(a.subprocess,'run',lambda *args,**kwargs:Rejected())
    with pytest.raises(ValueError,match='verifier rejected'):a.from_release(candidate,tmp_path/'out')
    assert not (tmp_path/'out').exists()
