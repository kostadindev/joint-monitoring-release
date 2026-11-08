from pathlib import Path
import sys,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pytest
from record_workflow import WorkflowRun,case,ENVIRONMENTS
from test_record_workflow import MechanicalProvider
from record_strategies import SELECTORS,select,group,policies,rule_judge

@pytest.mark.parametrize('env',ENVIRONMENTS)
def test_all_allocations_respect_k_s_and_cover_selected_once(env):
    rec=WorkflowRun(case(123),env,'combined',MechanicalProvider()).run()
    for p in policies(rec,seeds=(0,)):
        assert len(p['selected'])==p['k']
        assert sorted(e for g in p['groups'] for e in g)==p['selected']
        assert all(1<=len(g)<=p['s'] for g in p['groups'])
        if p['grouper']=='random_matched_provenance':
            control=group(rec,p['selected'],p['s'],'provenance',p['seed'])
            assert list(map(len,control))==p['group_sizes']

def test_selection_never_reads_content_or_labels():
    rec=WorkflowRun(case(123),'flat','combined',MechanicalProvider()).run()
    other=copy.deepcopy(rec);other['messages']=[];other['labels']={'SECRET':True};other['condition']='SECRET'
    for method in SELECTORS:assert select(rec,7,method,2)==select(other,7,method,2)

def test_rule_baseline_requires_selected_reference_and_does_not_claim_origin():
    rec=WorkflowRun(case(123),'flat','combined',MechanicalProvider()).run()
    attacked=tuple(rec['labels']['direct_false_edges'][0])
    reference=next(tuple(e) for e in rec['graph']['edges'] if e[0]=='registry' and e[1]==attacked[0])
    assert rule_judge(rec,[attacked])[attacked]==0
    assert rule_judge(rec,[attacked,reference])[attacked]==1

def test_entity_grouping_cannot_read_unselected_messages():
    rec=WorkflowRun(case(123),'flat','combined',MechanicalProvider()).run();es=select(rec,7,'random',2)
    before=group(rec,es,4,'entity_keys',2);other=copy.deepcopy(rec)
    for m in other['messages']:
        if (m['sender'],m['recipient']) not in es:m['payload']={'facts':{'SECRET':42}}
    assert group(other,es,4,'entity_keys',2)==before
