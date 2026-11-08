"""Independent regression checks for exact evidence reuse and bounded retries."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from test_run_topology_workload import fixture
from run_topology_workload import SYSTEM_TEXT, validate_call, reuse_calls
from monitor_record_workflows import FORMAT_RETRY
from record_workflow import canonical


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1., True, '0.001'])
def test_nonfinite_or_nonnumeric_aggregate_cannot_pass_delta_check(tmp_path, value):
    rec, edges, key, call = fixture(tmp_path)
    call['usage']['cost'] = value
    with pytest.raises(ValueError):
        validate_call(call, key, rec, edges)


def test_cannot_retry_an_already_valid_primary_answer(tmp_path):
    rec, edges, key, call = fixture(tmp_path)
    first = copy.deepcopy(call['attempts'][0])
    second = copy.deepcopy(first)
    second['attempt'] = 1
    second['system'] = SYSTEM_TEXT + FORMAT_RETRY
    second['prompt_sha256'] = hashlib.sha256(canonical([second['system'], second['prompt']]).encode()).hexdigest()
    for field in ('attempt', 'system', 'prompt_sha256'):
        call[field] = second[field]
    call['attempts'] = [first, second]; call['request_count'] = 2
    call['usage'] = {f: first['usage'][f] + second['usage'][f] for f in first['usage']}
    with pytest.raises(ValueError, match='resampled valid answer'):
        validate_call(call, key, rec, edges)


def test_invalid_auxiliary_evidence_never_resamples_valid_primary_score(tmp_path):
    _, _, _, original = fixture(tmp_path)
    raw = json.loads(original['raw']); raw.pop('source_evidence')
    rec, edges, key, call = fixture(tmp_path, [json.dumps(raw), original['raw']])
    assert call['request_count'] == 1
    assert not call['judgment']['source_evidence_valid']
    validate_call(call, key, rec, edges)


@pytest.mark.parametrize('location', ['call', 'attempt'])
@pytest.mark.parametrize('field,value', [('edges', []), ('planned_input_bytes', 1)])
def test_auxiliary_graph_and_byte_metadata_must_match_request(tmp_path, location, field, value):
    rec, edges, key, call = fixture(tmp_path)
    target = call if location == 'call' else call['attempts'][0]
    target[field] = value
    with pytest.raises(ValueError):
        validate_call(call, key, rec, edges)


def test_two_valid_distinct_draws_of_exact_prompt_are_not_cherry_picked(tmp_path):
    rec, edges, key, first = fixture(tmp_path)
    raw = json.loads(first['raw']); raw['reason'] = 'A different valid response'
    _, _, same_key, second = fixture(tmp_path, [json.dumps(raw)])
    assert same_key == key
    validate_call(first, key, rec, edges); validate_call(second, key, rec, edges)
    a, b = tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'
    a.write_text(json.dumps(first) + '\n'); b.write_text(json.dumps(second) + '\n')
    destination = tmp_path / 'new.jsonl'
    with pytest.raises(ValueError, match='conflicting raw judgments'):
        reuse_calls({key: (rec, edges)}, [a, b], destination)
    assert not destination.exists()
