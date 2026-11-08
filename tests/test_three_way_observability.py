from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from three_way_observability import audit,worlds


def test_every_partial_observation_has_identical_correct_and_incorrect_counts():
    result=audit()
    assert sum(w['violation'] for w in worlds())==4
    assert len(result['views'])==8
    for view in result['views']:
        if len(view['channels'])<3:
            assert all(c['correct_worlds']==c['incorrect_worlds'] and c['correct_worlds']>0 for c in view['cells'])
        else:
            assert all(c['correct_worlds']+c['incorrect_worlds']==1 for c in view['cells'])
