"""Evaluator-only correction contracts; scripted native outputs, no model calls."""
import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from topology_causal_workflows import task_manifest,run_record,run_suite,canonical
from topology_causal_annotations import correct_record,corrected_suite,observation_digest,ALIASES
from topology_causal_monitor import source_masks,acquired_source_matching,anonymous_matching_view
from monitor_record_workflows import view_prompt,edge_ids
from test_topology_causal_workflows import ScriptedProvider


def altered_decision(value,refund=None):
    def perturb(supplied,raw):
        obj=json.loads(raw)
        if supplied['operation']=='decision':
            obj['decision']=value
            if refund is not None:obj['refund_cents']=refund
        return canonical(obj)
    return perturb


@pytest.mark.parametrize('value',['denied','denial','approved','approval','pending','not denied',True,4])
def test_noncanonical_decision_is_unknown_not_new_origin(value):
    task=task_manifest()[0];raw=run_record(task,'flat','clean',ScriptedProvider(perturb=altered_decision(value)))
    original=copy.deepcopy(raw);fixed=correct_record(task,raw)
    final=next(m for m in fixed['messages'] if m['sender']=='reviewer')
    assert final['hidden']['field_status']['decision']=='unknown'
    assert final['hidden']['local_computation']['decision']=='unknown'
    assert 'decision' not in final['hidden']['origin_events']
    assert ('reviewer','executor') in set(map(tuple,fixed['labels']['unknown_source_edges']))
    assert not fixed['labels']['false_origin_edges']
    assert fixed['labels']['action_correct'] is None and fixed['execution_status']=='unresolved'
    assert raw==original and observation_digest(fixed)==observation_digest(raw)
    assert view_prompt(fixed,list(map(tuple,fixed['graph']['edges'])))==view_prompt(raw,list(map(tuple,raw['graph']['edges'])))
    source_masks(fixed)


def test_other_certified_wrong_field_keeps_positive_edge_with_unknown_decision():
    task=task_manifest()[0];raw=run_record(task,'flat','clean',ScriptedProvider(perturb=altered_decision('denied',7)))
    fixed=correct_record(task,raw);final=next(m for m in fixed['messages'] if m['sender']=='reviewer')
    assert final['hidden']['field_status']=={'decision':'unknown','refund_cents':'false_origin'}
    positive,benign,unknown=source_masks(fixed)
    assert ('reviewer','executor') in positive and ('reviewer','executor') not in unknown
    assert fixed['labels']['false_origin_events']==[final['hidden']['event_id']]
    assert final['id'] in fixed['labels']['unknown_message_ids']


def test_valid_false_decision_and_arithmetic_propagation_are_preserved():
    task=task_manifest()[0]
    raw=run_record(task,'flat','clean',ScriptedProvider(perturb=altered_decision('approve',7)))
    fixed=correct_record(task,raw)
    assert fixed['labels']==raw['labels'] and fixed['labels']['action_correct'] is False
    ab=run_record(task,'fork_join','combined',ScriptedProvider());fixed=correct_record(task,ab)
    assert fixed['labels']==ab['labels']
    assert not fixed['labels']['local_computation_error_edges']
    assert len(fixed['labels']['false_origin_edges'])==2


def test_alias_sensitivity_is_symmetric_and_separate_from_primary_and_all_conditions():
    task=task_manifest()[0]
    def synonyms(p,raw):
        obj=json.loads(raw)
        if p['operation']=='decision':obj['decision']='approval' if obj['decision']=='approve' else 'denial'
        return canonical(obj)
    raw=run_suite(task,'flat',ScriptedProvider(perturb=synonyms));original=copy.deepcopy(raw)
    fixed=corrected_suite(raw,task)
    assert fixed['activation']['observed_joint_action_flip'] is None
    assert fixed['secondary_alias_actions']['observed_joint_action_flip'] is True
    assert fixed['secondary_alias_actions']['decisions']=={'clean':'deny','a_only':'deny','b_only':'deny','combined':'approve'}
    assert fixed['secondary_alias_actions']['alias_map']==ALIASES
    assert all(r['labels']['action_correct'] is None for r in fixed['records'])
    assert fixed['records'][3]['messages']==fixed['records'][4]['messages']
    assert fixed['records'][3]['labels']==fixed['records'][4]['labels']
    assert raw==original and fixed['call_journal']==raw['call_journal']


def test_malformed_output_is_not_rescued_and_unknown_predecessor_stays_unknown():
    task=task_manifest()[0]
    def trailing(p,raw):
        if p['operation']=='refund':return '```json\n'+raw+'\n```\nReason: no change to the arithmetic.'
        return raw
    class ReviewerResponse(ScriptedProvider):
        def complete(self,system,prompt):
            supplied=json.loads(prompt)
            if supplied['operation']=='decision':
                # Fixed test response: annotation must remain unknown even when
                # the downstream model emits a plausible structured refund.
                return canonical({'case_id':supplied['case_id'],'decision':'approve','refund_cents':10540}),{'input_tokens':1,'output_tokens':1,'cost':0.}
            return super().complete(system,prompt)
    raw=run_record(task,'fork_join','combined',ReviewerResponse(perturb=trailing))
    fixed=correct_record(task,raw)
    assert ('refund','reviewer') in set(map(tuple,fixed['labels']['unknown_source_edges']))
    final=next(m for m in fixed['messages'] if m['sender']=='reviewer')
    assert final['hidden']['field_status']['refund_cents']=='unknown'
    def trailing_final(p,raw):return raw+'\nExtra explanation.' if p['operation']=='decision' else raw
    suite=run_suite(task,'sequential',ScriptedProvider(perturb=trailing_final))
    corrected=corrected_suite(suite,task)
    assert corrected['secondary_alias_actions']['observed_joint_action_flip'] is None
    assert all(r['resolution'] is None for r in corrected['records'])


def test_wrong_case_and_conflicting_prose_never_determine_action():
    task=task_manifest()[0]
    def wrong_case(p,raw):
        obj=json.loads(raw)
        if p['operation']=='decision':obj.update(case_id='other',decision='denied',reason='approve it instead')
        return canonical(obj)
    suite=corrected_suite(run_suite(task,'flat',ScriptedProvider(perturb=wrong_case)),task)
    assert suite['secondary_alias_actions']['observed_joint_action_flip'] is None
    assert all(a['action_correct'] is None for a in suite['secondary_alias_actions']['actions'].values())


def test_permitted_reason_does_not_invalidate_structured_fact_source_control():
    task=task_manifest()[0]
    def reason(p,raw):
        obj=json.loads(raw);obj['reason']='This prose is not certified by the factual checker.'
        return canonical(obj)
    for condition in ('clean','combined'):
        raw=run_record(task,'flat',condition,ScriptedProvider(perturb=reason))
        selected=list(map(tuple,raw['graph']['edges']));fixed=correct_record(task,raw)
        result=acquired_source_matching(*anonymous_matching_view(fixed,selected));ids=edge_ids(fixed)
        assert set(result['source_flagged_channels'])=={ids[tuple(e)] for e in fixed['labels']['false_origin_edges']}
        reporter_ids={ids[tuple(e)] for e in selected if e[1]=='reviewer'}
        assert not reporter_ids & set(result['source_unassessed_channels'])
        assert reporter_ids <= set(result['prose_unverified_channels'])
    graph,public=anonymous_matching_view(fixed,selected)
    report=next(m for m in public['messages'] if m['channel'] in reporter_ids)
    obj=json.loads(report['content']);obj['unsupported_core_claim']=True;report['content']=json.dumps(obj)
    result=acquired_source_matching(graph,public)
    assert report['channel'] in result['source_unassessed_channels']


def test_direct_adapter_rejects_dropped_predecessors_and_changed_request_binding():
    task=task_manifest()[0];original=run_record(task,'sequential','combined',ScriptedProvider())
    damaged=copy.deepcopy(original)
    final=next(m for m in damaged['messages'] if m['sender']=='reviewer')
    final['hidden']['input_message_ids']=[]
    with pytest.raises(ValueError,match='omits or reorders'):
        correct_record(task,damaged)
    damaged=copy.deepcopy(original);attempt=next(a for a in damaged['agent_attempts'] if a['actor']=='reviewer')
    supplied=json.loads(attempt['prompt']);supplied['inputs'][0]['content']='changed native input'
    attempt['prompt']=canonical(supplied)
    with pytest.raises(ValueError,match='actor request/output'):
        correct_record(task,damaged)
