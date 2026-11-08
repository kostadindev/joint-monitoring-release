import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_mesa_thresholds import audit_pairings


def test_zero_calibration_turns_low_suspicion_into_alert_but_fixed_rule_does_not(tmp_path):
    episodes=[]
    for family in ('static','dynamic'):
        for i,(split,label,score) in enumerate([('calibration',False,0)]*10+[('test',True,.1),('test',False,.5)]):
            row={'family':family,'condition':'pairing_0','split':split,'label':label,'score':score}
            episodes.append({'id':f'{family}:{i}','rows':[row],
                             'calls':[{'edges':[['worker_1','hub'],['worker_2','hub']], 'score':score,'raw':'consistent'}]})
    (tmp_path/'completed.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in episodes))
    result=audit_pairings(tmp_path)['static']['pairing_0']
    assert result['calibration_threshold']==0 and result['calibrated_tp']==1
    assert result['fixed_0.5_tp']==0 and result['fixed_0.5_fp']==0
