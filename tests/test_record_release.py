import copy
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pytest
from build_record_release import observations,opaque,remap,RESTORE
from verify_record_release import verify
from record_workflow import WorkflowRun,case
from test_record_workflow import MechanicalProvider


def test_public_export_omits_private_prompts_conditions_and_annotations():
    r=WorkflowRun(case(123),'flat','combined',MechanicalProvider()).run()
    pub=observations(r)
    assert pub['id']==opaque(r['id']) and 'combined' not in pub['id']
    assert not set(pub)&{'labels','condition','agent_calls','task_sha256'}
    assert all('hidden' not in m for m in pub['messages'])
    other=copy.deepcopy(r);other['condition']='SECRET';other['labels']={'SECRET':1};other['agent_calls']=['SECRET']
    for m in other['messages']:m['hidden']={'SECRET':1}
    assert observations(other)==pub
    assert remap({'episode_id':r['id'],'attempts':[{'episode_id':r['id']}]},{r['id']:pub['id']})=={'episode_id':pub['id'],'attempts':[{'episode_id':pub['id']}]}
    compile(RESTORE,'restore.py','exec')


def test_manifest_verifier_rejects_altered_file_before_loading_data(tmp_path):
    (tmp_path/'data.txt').write_text('original')
    want=hashlib.sha256((tmp_path/'data.txt').read_bytes()).hexdigest()
    (tmp_path/'manifest.json').write_text(json.dumps({'files_sha256':{'data.txt':want}}))
    (tmp_path/'data.txt').write_text('altered')
    with pytest.raises(ValueError,match='hash mismatch'):verify(tmp_path)
