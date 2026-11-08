"""Independent annotation/provenance regression checks; scripted calls only."""
import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from topology_causal_workflows import task_manifest,run_record,run_suite,canonical
from topology_causal_annotations import correct_record,corrected_suite,observation_digest
from test_topology_causal_workflows import ScriptedProvider


def test_omitting_a_delivered_predecessor_cannot_reclassify_origin():
    task=task_manifest()[0]
    raw=run_record(task,'fork_join','combined',ScriptedProvider())
    raw['messages'][-1]['hidden']['input_message_ids'].pop()
    with pytest.raises(ValueError,match='actual delivered predecessor'):
        correct_record(task,raw)


@pytest.mark.parametrize('tamper',['content','prompt'])
def test_native_output_and_actual_actor_inputs_are_bound(tamper):
    task=task_manifest()[0];raw=run_record(task,'flat','clean',ScriptedProvider())
    final=raw['messages'][-1]
    if tamper=='content':final['content']+=' '
    else:
        attempt=next(a for a in raw['agent_attempts'] if a['call_key']==final['hidden']['event_id'])
        prompt=json.loads(attempt['prompt']);prompt['inputs'].pop();attempt['prompt']=canonical(prompt)
    with pytest.raises(ValueError,match='actor request/output'):
        correct_record(task,raw)


def test_alias_action_never_changes_source_partition_or_native_attempts():
    def spelling(p,raw):
        obj=json.loads(raw)
        if p['operation']=='decision':
            obj['decision']='approved' if obj['decision']=='approve' else 'denied'
            obj['reason']='This prose is deliberately outside structured evaluation.'
        return canonical(obj)
    task=task_manifest()[0];raw=run_suite(task,'flat',ScriptedProvider(perturb=spelling))
    before=copy.deepcopy(raw);fixed=corrected_suite(raw,task)
    assert raw==before and fixed['call_journal']==raw['call_journal']
    assert fixed['activation']['observed_joint_action_flip'] is None
    assert fixed['secondary_alias_actions']['observed_joint_action_flip'] is True
    for original,corrected in zip(raw['records'],fixed['records']):
        assert observation_digest(original)==observation_digest(corrected)
        assert corrected['labels']['action_correct'] is None
        parts=[set(map(tuple,corrected['labels'][k])) for k in
               ('false_origin_edges','known_benign_source_edges','unknown_source_edges')]
        assert not any(parts[i]&parts[j] for i in range(3) for j in range(i))
        assert set.union(*parts)==set(map(tuple,corrected['graph']['edges']))
        assert ('reviewer','executor') in parts[2]
