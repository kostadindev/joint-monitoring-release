from fractions import Fraction
import copy
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import topology_only_policies as topology
import topology_workload_plan as planner
from record_audit_workloads import workload
from record_workflow import canonical
from monitor_record_workflows import SYSTEM, FORMAT_RETRY, view_prompt
from record_workload_benchmark import EVIDENCE, analyze
from record_workload_exact_random import evaluate_expected

SYSTEM_WITH_EVIDENCE = SYSTEM + EVIDENCE


def model_for(rec):
    graph, _ = planner.anonymize_graph(rec['graph'])
    return topology.fit_pair_model([{'graph': graph, 'group': [graph['edges'][0]],
                                     'reward': 0., 'weight': 1.}], alpha=.01)


def allocation_signature(rows):
    return [{k: a[k] for k in ('selector', 'grouper', 'k', 's', 'seed', 'selected', 'groups')}
            for a in rows if a['comparison_scope'] != 'privileged_location_oracle']


def test_same_graph_mapping_ignores_labels_roles_content_and_serialization():
    record = workload(0, 4)
    graph = record['graph']; poisoned = copy.deepcopy(graph)
    poisoned['roles'] = {'registry': 'malicious priority'}
    poisoned['messages'] = [{'target': 'secret'}]; poisoned['trust'] = 'poison'
    poisoned['nodes'].reverse(); poisoned['edges'].reverse(); poisoned['sink'] = 'incorrect'
    assert planner.anonymize_graph(graph) == planner.anonymize_graph(poisoned)
    a, restore = planner.anonymize_graph(graph)
    assert set(a) == {'nodes', 'edges'}
    assert all(type(n) is int for n in a['nodes'])
    assert {(restore[u], restore[v]) for u, v in a['edges']} == set(map(tuple, graph['edges']))
    assert list(planner.policies(graph)) == list(planner.policies(poisoned))
    for condition in ('clean', 'honest_error', 'false_report'):
        for routing in ('aligned', 'shuffled'):
            other = workload(0, 16, condition, routing, target_history='updated')
            assert planner.anonymize_graph(other['graph']) == (a, restore)
            assert list(planner.policies(other['graph'])) == list(planner.policies(graph))


def test_allocator_never_receives_metadata_or_original_names(monkeypatch):
    rec = workload(1, 4)
    planner._catalog.cache_clear()
    original_select, original_group = topology.select, topology.group
    seen = []
    def check(graph):
        assert set(graph) == {'nodes', 'edges'}
        assert all(type(n) is int for n in graph['nodes'])
        assert all(type(n) is int for e in graph['edges'] for n in e)
        seen.append(copy.deepcopy(graph))
    def select(graph, *args, **kwargs):
        check(graph); return original_select(graph, *args, **kwargs)
    def group(graph, *args, **kwargs):
        check(graph); return original_group(graph, *args, **kwargs)
    monkeypatch.setattr(topology, 'select', select)
    monkeypatch.setattr(topology, 'group', group)
    list(planner.policies(rec['graph']))
    assert seen
    assert all(g == seen[0] for g in seen)


def test_full_grid_counts_budgets_hashes_and_exact_expectation(tmp_path):
    rec = workload(0, 4); model = model_for(rec)
    jobs, rows, expectations = planner.plan_library([rec], SYSTEM_WITH_EVIDENCE, model)
    assert len(rows) == 352
    assert len(expectations) == 3
    assert sum(a['comparison_scope'] == 'privileged_location_oracle' for a in rows) == 2
    assert sum(a['selector'] == 'learned_pairs' for a in rows) == 3
    for a in rows:
        assert len(a['selected']) == len(set(a['selected'])) == a['k']
        assert sorted(e for part in a['groups'] for e in part) == sorted(a['selected'])
        assert all(1 <= len(part) <= a['s'] for part in a['groups'])
        for part, key, size in zip(a['groups'], a['view_keys'], a['group_input_bytes']):
            prompt = view_prompt(rec, part)
            assert key == hashlib.sha256(canonical([SYSTEM_WITH_EVIDENCE, prompt]).encode()).hexdigest()
            assert size == len((SYSTEM_WITH_EVIDENCE + FORMAT_RETRY + prompt).encode())
            if a['feasible']:
                assert key in jobs
    for p in expectations:
        baseline = next(a for a in rows if a['selector'] == 'structural_path_pairs' and
                        a['grouper'] == 'directed_path' and a['k'] == 7 and a['s'] == 2 and a['seed'] == p['seed'])
        assert list(map(tuple, p['selected'])) == baseline['selected']
        assert p['group_sizes'] == sorted(map(len, baseline['groups']))
        assert sum(Fraction(v['weight_numerator'], v['weight_denominator']) for v in p['weighted_views']) == p['expected_calls']
        for edge in p['selected']:
            assert sum(Fraction(v['weight_numerator'], v['weight_denominator'])
                       for v in p['weighted_views'] if edge in v['edges']) == 1
        assert evaluate_expected(rec, p, {})['evaluation_status'] == 'missing'
    analyzed = analyze([rec], rows, {}, tmp_path, expectations)
    assert len(analyzed) == 355
    assert all(a['evaluation_status'] in ('missing', 'infeasible') for a in analyzed)
    assert all('source_tp' not in a for a in analyzed)


def test_payload_changes_only_views_not_topology_choices():
    rec = workload(2, 4)
    other = copy.deepcopy(rec)
    other['labels'] = {'poison': 'do not read'}
    other['condition'] = 'poison'; other['case_id'] = 'poison'; other['graph']['roles'] = {'a': 'secret'}
    other['workflow'] += ' Public text changes the judge prompt only.'
    for message in other['messages']:
        message['hidden'] = {'poison': True}
        message['content'] += ' additional visible text'
    jobs_a, rows_a, expected_a = planner.plan_library([rec], SYSTEM_WITH_EVIDENCE, include_oracles=False)
    jobs_b, rows_b, expected_b = planner.plan_library([other], SYSTEM_WITH_EVIDENCE, include_oracles=False)
    assert allocation_signature(rows_a) == allocation_signature(rows_b)
    assert not set(jobs_a) & set(jobs_b)
    for a, b in zip(expected_a, expected_b):
        assert (a['selected'], a['group_sizes']) == (b['selected'], b['group_sizes'])
    # Allocator works even when all observation/evaluation sections are absent.
    assert list(planner.policies(rec['graph']))


def test_matched_random_size_vectors_and_location_oracle_separation():
    rec = workload(4, 4)
    rows = list(planner.policies(rec['graph']))
    assert not any('oracle' in a['selector'] for a in rows)
    for a in rows:
        if a['grouper'] != 'directed_path':
            continue
        b = next(b for b in rows if b['selector'] == a['selector'] and b['k'] == a['k'] and
                 b['s'] == a['s'] and b['seed'] == a['seed'] and b['grouper'] == 'random_matched_sizes')
        assert a['selected'] == b['selected']
        assert sorted(map(len, a['groups'])) == sorted(map(len, b['groups']))
    oracles = list(planner.oracle_policies(rec))
    assert [o['k'] for o in oracles] == [1, 2]
    assert all(o['comparison_scope'] == 'privileged_location_oracle' for o in oracles)


def test_relaxed_references_do_not_repair_or_truncate_primary_allocations():
    rec = workload(0, 16)
    jobs, rows, _ = planner.plan_library([rec], SYSTEM_WITH_EVIDENCE)
    full = [a for a in rows if a['selector'] == 'all_edges']
    assert len(full) == 2
    normal = next(a for a in full if a['input_byte_allowance'] == 24000)
    relaxed = next(a for a in full if a['input_byte_allowance'] == 60000)
    assert normal['groups'] == relaxed['groups'] and normal['view_keys'] == relaxed['view_keys']
    assert not normal['feasible'] and relaxed['feasible']
    assert normal['view_keys'][0] in jobs
    assert len([a for a in rows if a['k'] == 7 and a['grouper'] == 'full_selected_relaxed']) == 21


def test_exact_random_refuses_positive_probability_overflow():
    rec = workload(0, 4)
    for m in rec['messages']:
        m['content'] += 'x' * 25000
    with pytest.raises(ValueError, match='positive-probability'):
        planner.plan_library([rec], SYSTEM_WITH_EVIDENCE, include_oracles=False, seeds=(0,))


def test_mapping_and_catalog_are_validated_and_not_mutable_across_calls():
    rec = workload(0, 4)
    rows = list(planner.policies(rec['graph'], seeds=(0,)))
    original = copy.deepcopy(rows)
    rows[0]['selected'].clear()
    assert list(planner.policies(rec['graph'], seeds=(0,))) == original
    for seeds in ((), (0, 0), ('0',)):
        with pytest.raises(ValueError):
            list(planner.policies(rec['graph'], seeds=seeds))
    bad = copy.deepcopy(rec['graph']); bad['edges'].append(['unknown', 'reviewer'])
    with pytest.raises(ValueError):
        planner.anonymize_graph(bad)
    with pytest.raises(ValueError, match='duplicate episode'):
        planner.plan_library([rec, rec], SYSTEM_WITH_EVIDENCE, seeds=(0,))
    bad = copy.deepcopy(rec); bad['messages'].append(bad['messages'][0])
    with pytest.raises(ValueError, match='duplicate message'):
        planner.plan_library([bad], SYSTEM_WITH_EVIDENCE, seeds=(0,))
    model = model_for(rec); model['features'][2] = 'registry_fraction'
    with pytest.raises(ValueError, match='feature schema'):
        list(planner.policies(rec['graph'], model))
