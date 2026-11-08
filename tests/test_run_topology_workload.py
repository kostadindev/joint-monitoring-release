"""Reused judge evidence must match actual requests, not just a lookup key."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import threading
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_topology_workload import (MODEL,SYSTEM_TEXT,validate_call,reuse_calls,
                                    frozen_write,coverage,dispatch_order,uncommitted_attempts)
from record_audit_workloads import workload
from record_workflow import canonical
from record_workload_benchmark import inspect
from monitor_record_workflows import edge_ids,view_prompt


def fixture(tmp_path, responses=None):
    rec=workload(0,4)
    edges=[tuple(rec['graph']['edges'][0])]
    cid=edge_ids(rec)[edges[0]]
    raw=json.dumps({'incident_score':0,'source_scores':{cid:0},'content_scores':{cid:0},
                    'uncertain_source':True,'evidence':[],'reason':'No visible contradiction',
                    'source_evidence':{}})
    class Provider:
        model=MODEL; temperature=0; max_output_tokens=1024
        def __init__(self):self.responses=iter(responses or [raw])
        def complete(self,system,prompt):
            return next(self.responses),{'input_tokens':20,'output_tokens':10,'cost':.001}
    call=inspect(rec,edges,SYSTEM_TEXT,Provider(),tmp_path/'raw.jsonl',threading.Lock())
    key=hashlib.sha256(canonical([SYSTEM_TEXT,view_prompt(rec,edges)]).encode()).hexdigest()
    call['view_key']=key
    return rec,edges,key,call


def test_exact_reuse_preserves_original_calls_and_provenance(tmp_path):
    rec,edges,key,call=fixture(tmp_path)
    source=tmp_path/'source.jsonl';source.write_text(json.dumps(call)+'\n')
    destination=tmp_path/'new.jsonl'
    calls,n=reuse_calls({key:(rec,edges)},[source],destination)
    assert n==1 and calls[key]['reuse_provenance']['line']==1
    assert calls[key]['usage']==call['usage']
    again,n=reuse_calls({key:(rec,edges)},[source],destination)
    assert n==0 and again==calls


@pytest.mark.parametrize('field,value',[('prompt','changed'),('system','changed'),
    ('model','other-model'),('temperature',.5),('max_output_tokens',512),
    ('prompt_sha256','wrong'),('attempt',1)])
def test_rejects_request_setting_or_hash_changes(tmp_path,field,value):
    rec,edges,key,call=fixture(tmp_path)
    call['attempts'][0][field]=value
    with pytest.raises(ValueError):validate_call(call,key,rec,edges)


def test_rejects_score_tampering_and_cost_tampering(tmp_path):
    rec,edges,key,call=fixture(tmp_path)
    bad=copy.deepcopy(call);bad['judgment']['source_scores'][edge_ids(rec)[edges[0]]]=1.
    with pytest.raises(ValueError):validate_call(bad,key,rec,edges)
    bad=copy.deepcopy(call);bad['usage']['cost']=0
    with pytest.raises(ValueError):validate_call(bad,key,rec,edges)


def test_bounded_abstention_is_reusable_but_not_an_answer(tmp_path):
    rec,edges,key,call=fixture(tmp_path,['invalid','invalid'])
    validate_call(call,key,rec,edges)
    assert 'abstention' in call and call['request_count']==2
    call['judgment']={}
    with pytest.raises(ValueError):validate_call(call,key,rec,edges)


def test_import_validates_entire_source_before_writing(tmp_path):
    rec,edges,key,call=fixture(tmp_path)
    bad=copy.deepcopy(call);bad['raw']='changed'
    source=tmp_path/'source.jsonl';source.write_text(json.dumps(call)+'\n'+json.dumps(bad)+'\n')
    destination=tmp_path/'new.jsonl'
    with pytest.raises(ValueError):reuse_calls({key:(rec,edges)},[source],destination)
    assert not destination.exists()


def test_freeze_refuses_change_and_coverage_keeps_infeasible(tmp_path):
    path=tmp_path/'plan.json';frozen_write(path,'one');frozen_write(path,'one')
    with pytest.raises(ValueError):frozen_write(path,'two')
    rows=[{'feasible':True,'view_keys':['a']},{'feasible':False,'view_keys':['b']}]
    c=coverage(rows,[],{'a':{'usage':{'cost':.001}}},1)
    assert c['complete'] and c['by_status']=={'scored':1,'missing':0,'infeasible':1}


def test_orphan_raw_attempt_blocks_resampling_until_committed(tmp_path):
    rec,edges,key,call=fixture(tmp_path)
    assert uncommitted_attempts(tmp_path/'raw.jsonl',{})==[key]
    assert uncommitted_attempts(tmp_path/'raw.jsonl',{key:call})==[]


def test_dispatch_prioritizes_complete_primary_and_controls_without_scores():
    records=[{'id':'large','workload_size':16},{'id':'small','workload_size':4}]
    def row(key,selector='structural_path_pairs',grouper='directed_path',episode='large',k=7,s=2):
        return {'episode_id':episode,'selector':selector,'grouper':grouper,'k':k,'s':s,'feasible':True,'view_keys':[key]}
    rows=[row('main'),row('full',selector='all_edges',grouper='full_selected',k=13,s=13),
          row('small',episode='small'),row('mesa',selector='mesa_static'),
          row('secondary',selector='degree',grouper='spectral',s=4)]
    jobs={key:None for key in ['main','full','small','mesa','secondary','expected']}
    expectations=[row('expected',grouper='exact_matched_random')]
    order,tiers=dispatch_order(records,rows,expectations,jobs)
    assert tiers=={'main':0,'full':0,'small':1,'mesa':1,'secondary':2,'expected':0}
    assert [tiers[key] for key in order]==sorted(tiers.values())
    assert dispatch_order(list(reversed(records)),list(reversed(rows)),expectations,jobs)==(order,tiers)
