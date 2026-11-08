from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pytest
from record_workflow import WorkflowRun,case
from test_record_workflow import MechanicalProvider
from monitor_record_workflows import view_prompt,validity_views,parse_judgment,inspect_view,edge_ids

def test_monitor_receives_no_private_actor_instructions_or_hidden_labels():
    rec=WorkflowRun(case(42),'flat','combined',MechanicalProvider()).run()
    es=[tuple(rec['graph']['edges'][0])];before=view_prompt(rec,es)
    rec['labels']={'SECRET':'SECRET'};rec['agent_calls']=[{'system':'SECRET'}];rec['condition']='SECRET'
    for m in rec['messages']:m['hidden']={'SECRET':'SECRET'}
    assert before==view_prompt(rec,es) and 'SECRET' not in before
    body=json.loads(before)
    assert all((m['sender'],m['recipient']) in es for m in body['messages'])

def test_validity_library_has_singletons_provenance_and_full():
    rec=WorkflowRun(case(42),'flat','clean',MechanicalProvider()).run();views=validity_views(rec)
    assert sum(len(v)==1 for v in views)==13
    assert sum(len(v)==2 for v in views)==6
    assert max(map(len,views))==13

def test_message_evidence_maps_only_to_observed_channels():
    d={'incident_score':100,'source_scores':{'e00':100},'content_scores':{'e00':100},'uncertain_source':False,'evidence':['m0']}
    r=parse_judgment(json.dumps(d),['e00'],{'m0':'e00'})
    assert r['evidence']==['e00'] and r['schema_repairs']==['mapped_observed_message_ids_to_channels']
    assert r['source_scores']=={'e00':1}
    with pytest.raises(ValueError):parse_judgment(json.dumps(d),['e00'],{'m0':'UNOBSERVED'})

def test_format_retry_is_bounded_recorded_and_charged(tmp_path):
    import threading
    rec=WorkflowRun(case(42),'flat','clean',MechanicalProvider()).run();edges=(tuple(rec['graph']['edges'][0]),)
    eid=edge_ids(rec)[edges[0]]
    good=json.dumps({'incident_score':0,'source_scores':{eid:0},'content_scores':{eid:0},'uncertain_source':False,'evidence':[]})
    class Provider:
        model='test';temperature=0;calls=0
        def complete(self,system,prompt):
            self.calls+=1
            return ('our our our' if self.calls==1 else good),{'input_tokens':10,'output_tokens':2,'cost':.001}
    p=Provider();result=inspect_view(rec,edges,p,tmp_path/'calls.jsonl',threading.Lock())
    assert p.calls==2 and result['request_count']==2 and result['usage']['input_tokens']==20
    assert result['usage']['cost']==.002 and result['judgment']['incident_score']==0
    assert len((tmp_path/'calls.jsonl').read_text().splitlines())==2

@pytest.mark.parametrize('change',[{'source_scores':{}},{'incident_score':101},{'uncertain_source':'false'},{'evidence':['unknown']}])
def test_judge_malformed_response_is_not_a_negative(change):
    d={'incident_score':0,'source_scores':{'e00':0},'content_scores':{'e00':0},'uncertain_source':False,'evidence':[]}
    d.update(change)
    with pytest.raises((ValueError,TypeError,KeyError)):parse_judgment(json.dumps(d),['e00'])
