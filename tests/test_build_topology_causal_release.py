"""Local release/replay tests using fake transports only; no actual run outputs."""
import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import build_topology_causal_release as release
import run_topology_causal_workflows as generation
import run_topology_causal_monitor as monitor
from test_run_topology_causal_monitor import response,judgment


@pytest.fixture(scope='module')
def candidate_fixture(tmp_path_factory):
    root=tmp_path_factory.mktemp('causal_release_fixture');gen=root/'native';judge=root/'monitor'
    plan,tasks,suites=generation.prepare(gen,root/'budget.json')
    # A completely unsuccessful native positive control must still be released.
    def native_transport(body):
        operation=json.loads(body['input'][1]['content'])['operation']
        result=response('' if operation=='routing' else 'opaque native output retained',cost=.000001)
        result['model']=body['model'];return result
    result=generation.run_phase(gen,plan,tasks,suites,phase='fixture',phase_cap=1.,new_suite_limit=12,workers=3,
        transport_factory=lambda:native_transport)
    assert result['complete']
    prepared=monitor.prepare(gen,judge);plan,records,jobs,assignments,expectations,_=prepared
    abstention_prompt=monitor.view_prompt(*jobs[sorted(jobs)[0]])
    def judge_transport(body):
        prompt=body['input'][1]['content']
        return response('malformed judge answer' if prompt==abstention_prompt else judgment(prompt),cost=.000001)
    monitor.run_phase(judge,prepared,phase='fixture',phase_cap=1.,new_view_limit=len(jobs),workers=4,
        transport_factory=lambda:judge_transport)
    calls=monitor.load_calls(judge,jobs);monitor.complete_analysis(judge,records,assignments,expectations,jobs,calls)
    (gen/'.env').write_text('SECRET_TEST_SENTINEL=never_include')
    (judge/'provider_cache').mkdir();(judge/'provider_cache'/'not-an-artifact.json').write_text('do not copy')
    out=root/'candidate';result=release.build(gen,judge,out)
    return root,gen,judge,out,result


def rehash(root,name):
    manifest=release.read(root/'manifest.json');manifest['files_sha256'][name]=release.sha(root/name);release.write(root/'manifest.json',manifest)


def test_full_candidate_retains_all_failed_worlds_and_role_free_allocator_boundary(candidate_fixture):
    root,gen,judge,out,result=candidate_fixture
    assert result['records']==60 and result['native_suites']==12 and result['case_clusters']==4
    assert result['all_planned_worlds_retained'] and result['unknown_source_channels']>0
    assert result['verified'] and result['judge_views']>0 and result['judge_abstentions']==1
    manifest=release.read(out/'manifest.json')
    assert not any('.env' in name or 'provider_cache' in name for name in manifest['files_sha256'])
    assert not any('never_include' in p.read_text() for p in out.rglob('*.json'))
    rows=release.rows(out/'allocator_inputs.jsonl')
    assert rows and all(set(r)=={'graph','k','s'} and set(r['graph'])=={'nodes','edges'} for r in rows)
    observations=release.rows(out/'judge_data/observations.jsonl')
    assert len(observations)==60
    assert all('agent_attempts' not in r and 'labels' not in r and 'roles' not in r['graph'] for r in observations)
    assert all('hidden' not in m and 'kind' not in m for r in observations for m in r['messages'])
    assert any(m['content']=='' for r in observations for m in r['messages'])
    labels=release.rows(out/'evaluation/labels.jsonl')
    assert all('unknown_source_edges' in r['metadata']['labels'] for r in labels)
    assert all(r['metadata']['execution_status']=='unresolved' for r in labels)
    assert result['actual_spend']['generation']['confirmed_credits']>0
    assert result['actual_spend']['monitor']['confirmed_credits']>0


def test_verifier_rejects_role_leak_and_unknown_label_reclassification_even_after_manifest_rehash(candidate_fixture):
    _,_,_,out,_=candidate_fixture
    for name,mutate in [
        ('allocator_inputs.jsonl',lambda values:values[0].update(role_hint='registry')),
        ('evaluation/labels.jsonl',lambda values:values[0]['metadata']['labels'].update(unknown_source_edges=[]))]:
        path=out/name;original=path.read_bytes();manifest=(out/'manifest.json').read_bytes()
        try:
            values=release.rows(path);mutate(values);release.write_rows(path,values);rehash(out,name)
            with pytest.raises(ValueError,match='allocator|split mismatch'):release.verify(out)
        finally:path.write_bytes(original);(out/'manifest.json').write_bytes(manifest)


def test_metrics_tampering_detected_by_frozen_replay_not_only_manifest_hash(candidate_fixture):
    _,_,_,out,_=candidate_fixture;name='audit/monitor/per_record_metrics.json';path=out/name
    original=path.read_bytes();manifest=(out/'manifest.json').read_bytes()
    try:
        values=release.read(path);values[0]['invented_metric']='not emitted by frozen analyzer';release.write(path,values);rehash(out,name)
        with pytest.raises(ValueError,match='metrics replay mismatch'):release.verify(out)
    finally:path.write_bytes(original);(out/'manifest.json').write_bytes(manifest)


def test_verification_does_not_need_original_paths_or_ledger(candidate_fixture):
    root,gen,judge,out,_=candidate_fixture
    gen_backup=root/'native_hidden';judge_backup=root/'monitor_hidden';ledger=root/'budget.json';ledger_backup=root/'budget_hidden.json'
    gen.rename(gen_backup);judge.rename(judge_backup);ledger.rename(ledger_backup)
    try:assert release.verify(out)['verified']
    finally:gen_backup.rename(gen);judge_backup.rename(judge);ledger_backup.rename(ledger)


def test_incomplete_generation_refuses_candidate_before_creation(tmp_path):
    gen=tmp_path/'native';generation.prepare(gen,tmp_path/'ledger.json');judge=tmp_path/'monitor';judge.mkdir()
    out=tmp_path/'must_not_exist'
    with pytest.raises(ValueError,match='missing complete|incomplete'):release.build(gen,judge,out)
    assert not out.exists() and not (tmp_path/'ledger.json').exists()


def test_existing_candidate_is_not_overwritten(candidate_fixture):
    _,gen,judge,out,_=candidate_fixture
    with pytest.raises(ValueError,match='already exists'):release.build(gen,judge,out)


def test_missing_required_judge_view_cannot_be_hidden_by_rehashing_coverage_manifest(candidate_fixture):
    _,_,_,out,_=candidate_fixture
    path=next((out/'audit/monitor/completed_views').glob('*.json'));name=str(path.relative_to(out))
    original=path.read_bytes();manifest_bytes=(out/'manifest.json').read_bytes()
    try:
        path.unlink();manifest=release.read(out/'manifest.json');manifest['files_sha256'].pop(name)
        manifest['monitor_files'].remove(str(path.relative_to(out/'audit/monitor')))
        release.write(out/'manifest.json',manifest)
        with pytest.raises(ValueError,match='incomplete planned judge coverage'):release.verify(out)
    finally:path.write_bytes(original);(out/'manifest.json').write_bytes(manifest_bytes)


def test_actual_native_spend_counts_original_requests_once_not_replayed_predecessors(candidate_fixture):
    _,_,_,out,result=candidate_fixture
    records=release.read(out/'private/generation/records.json')
    attempts=[a for r in records for a in r['agent_attempts']]
    request_ids={a['request_lineage']['ledger_request_id'] for a in attempts}
    actual=result['actual_spend']['generation']
    assert len(request_ids)==actual['actual_ledger_requests']
    assert sum(a['usage']['cost'] for a in attempts)>actual['confirmed_credits']
    assert any(a['reused'] for a in attempts)
    assert actual['uncertain_retained_credits']==actual['active_reserved_credits']==0


def test_release_keeps_original_annotations_and_replays_separate_correction(candidate_fixture):
    _,_,_,out,result=candidate_fixture
    originals=release.read(out/'private/generation/records.json')
    corrected=release.read(out/'audit/monitor/records.json')
    from topology_causal_annotations import observation_digest,VERSION
    assert result['original_native_observations_preserved']
    assert result['annotation_correction_version']==VERSION
    assert result['secondary_alias_suites']==12
    for original,new in zip(originals,corrected):
        assert 'annotation_revision' not in original
        assert new['annotation_revision']['original_record_sha256']==release.digest(original)
        assert new['annotation_revision']['original_labels']==original['labels']
        assert observation_digest(original)==observation_digest(new)
    correction=release.read(out/'audit/monitor/annotation_correction.json')
    sensitivity=release.read(out/'audit/monitor/secondary_alias_actions.json')
    assert correction['version']==VERSION and len(correction['suites'])==12
    assert len(sensitivity)==12 and all('Posthoc' in x['scope'] for x in sensitivity)


@pytest.mark.parametrize('artifact',['annotation_correction','secondary_alias_actions'])
def test_correction_sidecars_cannot_be_changed_even_after_manifest_rehash(candidate_fixture,artifact):
    _,_,_,out,_=candidate_fixture;name='audit/monitor/'+artifact+'.json';path=out/name
    original=path.read_bytes();manifest=(out/'manifest.json').read_bytes()
    try:
        value=release.read(path)
        if artifact=='annotation_correction':value['version']='silently changed domain'
        else:value[0]['observed_joint_action_flip']=True
        release.write(path,value);rehash(out,name)
        with pytest.raises(ValueError,match=artifact+' replay mismatch'):release.verify(out)
    finally:path.write_bytes(original);(out/'manifest.json').write_bytes(manifest)
