from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pytest
from situation_suite import FAMILIES,VARIANTS,instance,make_suite
from compare_situations import prompt,groupings,strategy_indices,parse
from night_budget import NightBudget

@pytest.mark.parametrize('family',FAMILIES)
def test_combined_is_only_violation(family):
    for seed in range(50):
        assert {v:instance(family,seed,v)['label'] for v in VARIANTS}=={v:v=='combined' for v in VARIANTS}

def test_prompt_has_no_hidden_labels():
    rec=instance('split_budget',123,'combined');es,_,_=groupings(rec)
    before=prompt(rec,es)
    rec.update(label=False,evidence_edges=[['SECRET','LABEL']],oracle_facts={'SECRET':'LABEL'},variant='SECRET')
    assert prompt(rec,es)==before
    assert 'SECRET' not in before

def test_partitions_and_graph_policy():
    rec=instance('restricted_release',17,'combined');es,ps,vs=groupings(rec)
    assert len(ps)==3 and len(vs)==11
    for p in ps:
        assert sorted(e for g in p for e in g)==es
        assert [len(g) for g in p]==[2,2]
    p=ps[strategy_indices(rec,ps)['structural']]
    assert all(a[1]==b[1] for a,b in p)

def test_splits_disjoint_and_calibration_negative():
    rs=make_suite()
    cal={r['case_seed'] for r in rs if r['split']=='calibration'}
    test={r['case_seed'] for r in rs if r['split']=='test'}
    assert not cal&test
    assert not any(r['label'] for r in rs if r['split']=='calibration')

def test_budget_persists_and_failure_reserves(tmp_path):
    p=tmp_path/'budget.json';b=NightBudget(p,'test',.1)
    k=b.reserve_call();b.settle(k,.01)
    b=NightBudget(p,'test',.1);k=b.reserve_call()
    with pytest.raises(RuntimeError):b.reserve_call()
    b.block(k)
    assert b.summary()['charged_or_reserved']==pytest.approx(.06)
    with pytest.raises(RuntimeError):NightBudget(p,'other',1).reserve_call()

@pytest.mark.parametrize('text',['garbage','{"score": 101}','{"score": true}','{"score": -1}'])
def test_malformed_is_not_benign(text):
    with pytest.raises((ValueError,KeyError)):parse(text)
