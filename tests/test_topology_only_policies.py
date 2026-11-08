"""Contract checks for anonymous-topology-only allocation and structural features."""
from collections import Counter
import copy
from itertools import combinations
from pathlib import Path
import random
import sys

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from topology_only_policies import (FEATURES, SELECTORS, GROUPERS, validate_graph,
                                    select, group, allocate, pair_features, fit_pair_model)


GRAPH = {'nodes': list(range(8)),
         'edges': [[0, 1], [0, 2], [0, 3], [1, 4], [2, 4], [3, 5],
                   [4, 6], [5, 6], [6, 7], [5, 3]]}


def fitted():
    edges = [tuple(e) for e in GRAPH['edges']]
    examples = [{'graph': GRAPH, 'group': list(c), 'reward': float(len(c) == 2), 'weight': 1.}
                for c in [(e,) for e in edges] + list(combinations(edges, 2))]
    return fit_pair_model(examples, alpha=.1)


@pytest.mark.parametrize('extra', ['roles', 'messages', 'trust', 'labels', 'node_attrs', 'edge_attrs', 'sink'])
def test_rejects_poisoned_graph_metadata_at_every_public_entry(extra):
    bad = copy.deepcopy(GRAPH); bad[extra] = {'registry': 'reviewer', 'target': 7}
    for call in [lambda: validate_graph(bad), lambda: select(bad, 3),
                 lambda: group(bad, [[0, 1]], 2), lambda: pair_features(bad, [[0, 1]]),
                 lambda: allocate(bad, 3, 2),
                 lambda: fit_pair_model([{'graph': bad, 'group': [[0, 1]], 'reward': 1., 'weight': 1.}])]:
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize('bad', [
    {'nodes': ['registry', 'reviewer'], 'edges': [['registry', 'reviewer']]},
    {'nodes': [True, 2], 'edges': [[True, 2]]},
    {'nodes': [0, 1], 'edges': [[0, 1, {'trusted': True}]]},
    {'nodes': [0, 1], 'edges': [{'source': 0, 'target': 1}]},
    {'nodes': [0, 0], 'edges': []},
    {'nodes': [0, 1], 'edges': [[0, 2]]},
    {'nodes': [0, 1], 'edges': [[0, 1], [0, 1]]},
    {'nodes': [0, 1], 'edges': [[0, 0]]},
    nx.DiGraph([(0, 1)]),
    {'graph': GRAPH},
])
def test_rejects_names_attributes_multiedges_and_record_wrappers(bad):
    with pytest.raises(ValueError):
        validate_graph(bad)


def test_feature_relabeling_and_input_order_invariance():
    graphs = [GRAPH, {'nodes': [0, 1, 2, 3], 'edges': [[0, 1], [1, 0], [2, 3]]}]
    for graph in graphs:
        edges = [tuple(e) for e in graph['edges']]
        groups = [(e,) for e in edges] + list(combinations(edges, 2))
        for seed in range(4):
            rng = random.Random(seed); labels = rng.sample(range(-200, 1000), len(graph['nodes']))
            mapping = dict(zip(graph['nodes'], labels))
            renamed = {'nodes': list(reversed(labels)),
                       'edges': [[mapping[u], mapping[v]] for u, v in reversed(edges)]}
            for c in groups:
                mapped = [[mapping[u], mapping[v]] for u, v in reversed(c)]
                np.testing.assert_allclose(pair_features(graph, list(c)), pair_features(renamed, mapped))
    assert len(FEATURES) == len(set(FEATURES))
    assert not any(term in name for name in FEATURES
                   for term in ('registry', 'reviewer', 'executor', 'role', 'trust', 'message', 'id_'))


def test_all_policies_obey_budget_disjoint_coverage_and_group_cap():
    model = fitted()
    for seed in (0, 19):
        for selector in SELECTORS:
            for k in (0, 3, 7, len(GRAPH['edges'])):
                selected = select(GRAPH, k, selector, seed, model)
                assert len(selected) == len(set(selected)) == k
                assert set(selected) <= set(map(tuple, GRAPH['edges']))
                for s in (1, 2, 4, max(1, k)):
                    for grouper in GROUPERS:
                        result = group(GRAPH, selected, s, grouper, seed, model)
                        if grouper == 'full_selected' and k > s:
                            assert result is None
                        else:
                            assert sorted(e for c in result for e in c) == sorted(selected)
                            assert all(1 <= len(c) <= s for c in result)
                assert selected == select(GRAPH, k, selector, seed, model)


def test_random_matching_preserves_reference_size_vector():
    for seed in range(10):
        selected = select(GRAPH, 7, 'structural_path_pairs', seed)
        for s in (1, 2, 3, 7):
            reference = group(GRAPH, selected, s, 'directed_path', seed)
            randomized = group(GRAPH, selected, s, 'random_matched_sizes', seed)
            assert sorted(map(len, randomized)) == sorted(map(len, reference))
            assert sorted(e for c in reference for e in c) == sorted(e for c in randomized for e in c)


def test_structure_recognizes_paths_without_privileged_endpoints():
    chain = {'nodes': [91, -5, 700], 'edges': [[91, -5], [-5, 700]]}
    out = allocate(chain, 2, 2, 'structural_path_pairs', 'directed_path', 7)
    assert out['groups'] == [[(-5, 700), (91, -5)]]
    forks = {'nodes': [0, 1, 2], 'edges': [[0, 1], [0, 2]]}
    assert group(forks, forks['edges'], 2, 'directed_path') == [[(0, 1)], [(0, 2)]]
    assert allocate(chain, 2, 1, grouper='full_selected')['feasible_group_cap'] is False


def test_tied_edges_receive_no_fixed_numeric_id_priority():
    # All edges are structurally identical. Seeds must vary the selected leaf;
    # tight numeric labels may not deterministically privilege one position.
    star = {'nodes': [10, 11, 12, 13, 14], 'edges': [[10, x] for x in (11, 12, 13, 14)]}
    for method in ('random', 'degree', 'betweenness', 'pagerank', 'mesa_static', 'downstream_coverage'):
        counts = Counter(select(star, 1, method, seed)[0][1] for seed in range(160))
        assert set(counts) == {11, 12, 13, 14}
        assert min(counts.values()) >= 20
        assert max(counts.values()) <= 60


def test_reversing_serialization_changes_nothing():
    permuted = {'nodes': list(reversed(GRAPH['nodes'])), 'edges': list(reversed(GRAPH['edges']))}
    model = fitted()
    for selector in SELECTORS:
        a = select(GRAPH, 7, selector, 22, model)
        b = select(permuted, 7, selector, 22, model)
        assert a == b
        for grouper in GROUPERS:
            assert group(GRAPH, a, 4, grouper, 22, model) == group(permuted, b, 4, grouper, 22, model)


def test_training_is_explicitly_supervised_and_deployment_rejects_role_model():
    model = fitted()
    assert model['features'] == list(FEATURES)
    assert 'development rewards' in model['supervision']
    assert len(model['coef']) == len(FEATURES)
    group(GRAPH, select(GRAPH, 7, 'learned_pairs', model=model), 2, 'learned_pairs', model=model)
    bad = copy.deepcopy(model); bad['features'][2] = 'protected_chain'
    with pytest.raises(ValueError, match='feature schema'):
        select(GRAPH, 7, 'learned_pairs', model=bad)
    bad = copy.deepcopy(model); bad['roles'] = {'0': 'registry'}
    with pytest.raises(ValueError, match='model schema'):
        select(GRAPH, 7, 'learned_pairs', model=bad)
    bad = copy.deepcopy(model); bad['coef'][0] = float('nan')
    with pytest.raises(ValueError):
        select(GRAPH, 7, 'learned_pairs', model=bad)
    for method in ('select', 'group'):
        with pytest.raises(ValueError, match='requires a model'):
            (select(GRAPH, 2, 'learned_pairs') if method == 'select'
             else group(GRAPH, GRAPH['edges'][:2], 2, 'learned_pairs'))


def test_invalid_budgets_selections_and_empty_graph():
    for k in (-1, 100, True, 2.5):
        with pytest.raises(ValueError):
            select(GRAPH, k)
    for s in (0, -1, True, 2.5):
        with pytest.raises(ValueError):
            group(GRAPH, [[0, 1]], s)
    for selected in ([[0, 1], [0, 1]], [[7, 0]], [['0', 1]], [[True, 4]]):
        with pytest.raises(ValueError):
            group(GRAPH, selected, 2)
    for selector in SELECTORS[:-1]:
        assert select({'nodes': [], 'edges': []}, 0, selector) == []
    assert allocate({'nodes': [], 'edges': []}, 0, 1)['groups'] == []
    with pytest.raises(ValueError):
        fit_pair_model([])


def test_rng_domains_are_stable_and_distinct():
    import hashlib
    import topology_only_policies as module
    # Literal serialization pins the versioned seed contract across processes.
    expected = int.from_bytes(hashlib.sha256(b'["topology-only-rng-v2","reindex",7]').digest(), 'big')
    assert module.stream_seed(7, 'reindex') == expected
    domains = ['reindex', 'group:directed_path', 'group:matched_random_shuffle']
    domains += ['selection:' + name for name in SELECTORS]
    domains += ['group:' + name for name in ('learned_pairs', 'shared_receiver', 'line_community', 'spectral')]
    assert len({module.stream_seed(7, d) for d in domains}) == len(domains)
    assert module.stream_seed(7, 'reindex') != module.stream_seed(8, 'reindex')
    with pytest.raises(ValueError):
        module.stream_seed(True, 'reindex')
    with pytest.raises(ValueError):
        module.stream_seed(0, '')


def test_actual_rng_calls_use_distinct_domains_and_shared_reference(monkeypatch):
    import topology_only_policies as module
    real_random = module.random.Random
    calls = []
    def tracked(seed):
        calls.append(seed)
        return real_random(seed)
    monkeypatch.setattr(module.random, 'Random', tracked)
    seed = 37
    chosen = module.select(GRAPH, 7, 'structural_path_pairs', seed)
    assert calls == [module.stream_seed(seed, 'reindex'),
                     module.stream_seed(seed, 'selection:structural_path_pairs')]
    calls.clear()
    ref = module.group(GRAPH, chosen, 2, 'directed_path', seed)
    assert calls == [module.stream_seed(seed, 'reindex'),
                     module.stream_seed(seed, 'group:directed_path')]
    calls.clear()
    matched = module.group(GRAPH, chosen, 2, 'random_matched_sizes', seed)
    assert calls == [module.stream_seed(seed, 'reindex'),
                     module.stream_seed(seed, 'group:directed_path'),
                     module.stream_seed(seed, 'group:matched_random_shuffle')]
    assert sorted(map(len, ref)) == sorted(map(len, matched))
    assert len(set(calls)) == 3


def test_changing_shuffle_stream_cannot_change_selection_or_reference_sizes(monkeypatch):
    import topology_only_policies as module
    derive = module.stream_seed
    original = []
    for seed in range(8):
        chosen = module.select(GRAPH, 7, 'structural_path_pairs', seed)
        ref = module.group(GRAPH, chosen, 2, 'directed_path', seed)
        random_groups = module.group(GRAPH, chosen, 2, 'random_matched_sizes', seed)
        original.append((chosen, ref, random_groups))
    def changed(seed, domain):
        return derive(seed, domain + ':independent-test-override' if domain == 'group:matched_random_shuffle' else domain)
    monkeypatch.setattr(module, 'stream_seed', changed)
    changed_partitions = 0
    for seed, (chosen, ref, random_groups) in enumerate(original):
        assert module.select(GRAPH, 7, 'structural_path_pairs', seed) == chosen
        assert module.group(GRAPH, chosen, 2, 'directed_path', seed) == ref
        new = module.group(GRAPH, chosen, 2, 'random_matched_sizes', seed)
        assert sorted(map(len, new)) == sorted(map(len, ref))
        changed_partitions += new != random_groups
    assert changed_partitions > 0


def test_conditional_random_partition_law_is_uniform_and_relabel_invariant(monkeypatch):
    """Enumerate all shuffles, rather than rely on a noisy statistical test."""
    from itertools import permutations
    import topology_only_policies as module
    base = {'nodes': [0, 1, 2, 3], 'edges': [[0, 1], [1, 2], [2, 3], [3, 0]]}
    real_random = module.random.Random
    root_seed = 19
    shuffle_seed = module.stream_seed(root_seed, 'group:matched_random_shuffle')
    distributions = []
    for mapping in ({i: i for i in range(4)}, {0: 91, 1: -20, 2: 77, 3: 13}):
        graph = {'nodes': list(mapping.values()),
                 'edges': [[mapping[u], mapping[v]] for u, v in base['edges']]}
        inverse = {v: k for k, v in mapping.items()}
        counts = Counter()
        for permutation in permutations(range(4)):
            class ExhaustiveShuffle:
                def shuffle(self, values):
                    old = list(values)
                    values[:] = [old[i] for i in permutation]
            def factory(seed):
                return ExhaustiveShuffle() if seed == shuffle_seed else real_random(seed)
            monkeypatch.setattr(module.random, 'Random', factory)
            result = module.group(graph, graph['edges'], 2, 'random_matched_sizes', root_seed)
            assert sorted(map(len, result)) == [2, 2]
            normalized = tuple(sorted(tuple(sorted((inverse[u], inverse[v]) for u, v in part))
                                      for part in result))
            counts[normalized] += 1
        assert len(counts) == 3
        assert set(counts.values()) == {8}
        distributions.append(counts)
    assert distributions[0] == distributions[1]
