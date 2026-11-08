"""Reference visibility is stricter than a score and weaker than cited proof."""
import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_completed_topology_grid import matching_pairs,view_counts,uniform_pooled,audit
from record_audit_workloads import workload
from monitor_record_workflows import edge_ids


def fixture():
    record=workload(0,16,'false_report','shuffled')
    pairs=matching_pairs(record,True);assert len(pairs)==1
    origin,reference,_,_=pairs[0]
    ids=edge_ids(record)
    call={'judgment':{'source_scores':{ids[origin]:1,ids[reference]:0}}}
    return record,pairs,origin,reference,call


def test_reference_must_be_in_same_judge_group_and_score_must_flag():
    r,pairs,origin,reference,call=fixture()
    assert view_counts(r,[origin],call,pairs)['matching_reference_visible_tp']==0
    counted=view_counts(r,[origin,reference],call,pairs)
    assert counted['score_tp']==counted['matching_reference_visible_tp']==1
    # No citation or rationale is required for this weaker evidence-availability
    # statistic; neither is invented by the audit.
    assert 'source_evidence' not in call['judgment']
    call['judgment']['source_scores'][edge_ids(r)[origin]]=0
    assert view_counts(r,[origin,reference],call,pairs)['matching_reference_visible_tp']==0
    assert view_counts(r,[origin,reference],{'abstention':{}},pairs)['score_tp']==0


@pytest.mark.parametrize('replace',['case','revision','truth'])
def test_wrong_case_revision_or_agreeing_reference_does_not_support(replace):
    r,pairs,origin,reference,call=fixture();r=copy.deepcopy(r)
    rid=pairs[0][3];ref=next(m for m in r['messages'] if m['id']==rid)
    claim=next(m for m in r['messages'] if m['id']==pairs[0][2])
    if replace=='case':ref['content']=ref['content'].replace(r['labels']['target_case_id'],'case_unrelated')
    elif replace=='revision':ref['content']=ref['content'].replace('revision 1.','revision 9.')
    else:ref['content']=claim['content']
    assert view_counts(r,[origin,reference],call,matching_pairs(r,True))['matching_reference_visible_tp']==0


def test_uniform_pooled_matcher_parses_observations_and_has_exact_pair_probability():
    records=[workload(0,16,c,route) for route in ('aligned','shuffled') for c in ('clean','false_report','honest_error')]
    result=uniform_pooled(records)
    for row in result['rows']:
        assert row['inspected_benign_fpr']==0
        if row['condition']!='clean':assert row['recall']==pytest.approx(7/26)
    # Labels are evaluator bookkeeping; removing origins cannot change the
    # publicly parsed contradictions supplied to the deterministic detector.
    r=records[1];before=matching_pairs(r);r=copy.deepcopy(r);r['labels']['false_origin_messages']=[]
    assert matching_pairs(r)==before and matching_pairs(r,True)==[]


def test_partial_main_analysis_is_refused_before_any_output(tmp_path):
    import json
    (tmp_path/'main/analysis').mkdir(parents=True)
    (tmp_path/'main/analysis/manifest.json').write_text(json.dumps({'coverage':{'complete':False}}))
    with pytest.raises(ValueError,match='complete main'):audit(tmp_path/'main',tmp_path/'supplement',tmp_path/'out')
    assert not (tmp_path/'out').exists()
