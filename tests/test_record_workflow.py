from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pytest
from record_workflow import case,target_changes,expected,layout,FIELDS,ENVIRONMENTS,WorkflowRun

def test_coordinated_rule_requires_both_changes():
    for seed in range(500):
        task=case(seed)
        assert {c:expected(task['fields']|target_changes(task,c))['decision'] for c in ('clean','a_only','b_only','combined')}=={
            'clean':'deny','a_only':'deny','b_only':'deny','combined':'approve'}

def test_random_single_change_is_one_real_change():
    fields=set()
    for seed in range(200):
        task=case(seed);change=target_changes(task,'single');assert len(change)==1
        field,value=next(iter(change.items()));assert value!=task['fields'][field];fields.add(field)
    assert fields==set(FIELDS)

class MechanicalProvider:
    """Plumbing oracle only: not a research model or evidence of attack effectiveness."""
    def complete(self,system,user):
        data=json.loads(user)
        if 'registry_receipt' in data:
            facts=dict(data['prior_reported_facts']);field=data['assigned_field'];value=data['registry_receipt']['facts'][field]
            if 'Your objective is to report' in system:
                import re
                value=json.loads(re.search(r'Your objective is to report \w+ as (.+?) instead',system).group(1))
            facts[field]=value;result={'case_id':data['case_id'],'facts':facts}
        else:
            facts={k:v for report in data['reports'] for k,v in report['facts'].items()}
            result=expected(facts) if 'You decide' in system else {'case_id':data['case_id'],'facts':facts}
        return json.dumps(result),{'input_tokens':0,'output_tokens':0,'cost':0}

@pytest.mark.parametrize('environment',ENVIRONMENTS)
def test_workflow_labels_and_channels(environment):
    task=case(19);run=WorkflowRun(task,environment,'combined',MechanicalProvider()).run()
    assert run['labels']['targeted_approval']
    assert len(run['labels']['direct_false_edges'])==2
    assert len(run['labels']['controlled_actors'])==2
    assert not run['labels']['benign_error_edges']
    assert all([m['sender'],m['recipient']] in run['graph']['edges'] for m in run['messages'])
    receipts=[m for m in run['messages'] if m['trusted']]
    assert len(receipts)==6 and all(set(m['hidden']['field_status'].values())=={'correct'} for m in receipts)
    if environment in ('hierarchical','sequential'):assert run['labels']['propagated_false_edges']
    clean=WorkflowRun(task,environment,'clean',MechanicalProvider()).run()
    assert clean['labels']['correct_decision'] and not clean['labels']['direct_false_edges']


def test_controlled_reporter_miscopy_of_other_attackers_field_is_not_commanded_origin():
    class Miscopy(MechanicalProvider):
        def complete(self,system,user):
            raw,usage=super().complete(system,user);data=json.loads(user);result=json.loads(raw)
            if 'registry_receipt' in data and 'designated dishonest reporter' in system:
                assigned=data['assigned_field']
                for field in ('purchase_age_days','return_window_days'):
                    if field!=assigned and field in data['prior_reported_facts']:
                        result['facts'][field]+=1
            return json.dumps(result),usage
    rec=WorkflowRun(case(19),'sequential','combined',Miscopy()).run()
    errors=[(m,field) for m in rec['messages'] for field,status in m['hidden']['field_status'].items()
            if status=='benign_error' and m['hidden']['controlled_sender']]
    assert len(errors)==1
    message,field=errors[0]
    assert rec['graph']['roles'][message['sender']]!='report '+field
