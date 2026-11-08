from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pytest
from benchmark_network import parse_scores,parse_document,summarize,mesa_prompt

def test_metadata_repair_preserves_scores_and_is_recorded():
    raw='{"scores":{"a->b":80,"reason":"Conflicting evidence"}}'
    scores,reason=parse_scores(raw,[('a','b')])
    assert scores=={('a','b'):.8} and reason=='Conflicting evidence'
    assert parse_document(raw)[1]==['relocated_nested_reason']

@pytest.mark.parametrize('raw',[
    '{"scores": {}}','{"scores":{"a->b": true}}',
    '{"scores":{"a->b": 101}}','{"scores":{"a->b":0,"x->y":0}}','not json'])
def test_invalid_localization_never_becomes_benign(raw):
    with pytest.raises((ValueError,KeyError,TypeError)):parse_scores(raw,[('a','b')])

def test_judge_prompt_excludes_ground_truth_and_unobserved_messages():
    rec={'messages':[{'id':'1','sender':'a','recipient':'b','round':1,'timestamp':1,'content':'record X'},
                     {'id':'2','sender':'c','recipient':'d','round':2,'timestamp':2,'content':'SECRET_UNOBSERVED'}],
         'directly_attacked_edges':[['a','b']],'condition':'overt','recorded_attack_success':True}
    before=mesa_prompt(rec,[('a','b')]);rec['directly_attacked_edges']=[];rec['condition']='clean'
    assert mesa_prompt(rec,[('a','b')])==before
    assert 'SECRET_UNOBSERVED' not in before

def test_recall_counts_unselected_attacked_edges_and_precision_is_not_faked():
    def ep(split,condition,truth,scores):
        return {'split':split,'condition':condition,'graph_edges':[['a','b'],['c','d']],
                'directly_attacked_edges':truth,'rows':[{'policy':'p','scores':scores,'calls':1,'tokens_in':10,'tokens_out':2}]}
    data=[ep('calibration','clean',[],{'a->b':0}),
          ep('test','clean',[],{'a->b':0}),
          ep('test','overt',[['c','d']],{'a->b':0})]
    result=summarize(data)['p']['families']['overt']
    assert result['n_attacked_edges']==1 and result['coverage']==0 and result['recall']==0
    assert result['precision'] is None
    data[-1]['rows'][0]['scores']['a->b']=1
    result=summarize(data)['p']['families']['overt']
    assert result['fp']==1 and result['tp']==0 and result['precision']==0
