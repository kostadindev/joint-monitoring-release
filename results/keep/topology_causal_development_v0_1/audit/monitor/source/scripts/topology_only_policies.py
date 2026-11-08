"""Strict anonymous-topology allocation, independent of observations and labels.

Public graphs are exactly {"nodes": [int, ...], "edges": [[int, int], ...]}.
The input is a simple directed graph without self-loops (cycles are allowed).
No node/edge attributes, role strings, messages, trust or evaluator metadata are
accepted. Integer IDs are opaque: they enter no score or feature. Structure may
legitimately reveal hub/source/sink patterns; semantic role annotations may not.

Seeded random internal reindexing and random tie priorities prevent ID order from
privileging an actor. Features are permutation invariant. On symmetric graphs an
individual seeded allocation need not be equivariant under relabeling; its random
law is invariant, not a promise of identical mapped choices for the same seed.
SHA-256 domain-separated streams isolate reindexing, each acquisition policy,
directed-path grouping and matched-random shuffling. Sharing a root seed does not
restart the same random stream for those distinct operations.

Selectors/groupers return original anonymous integer edges. Full-selected grouping
returns None if its group exceeds s. Byte feasibility belongs to the planner, not
this allocator: the allocator never sees message lengths or payloads. Matching
random grouping uses the directed-path partition's group-size multiset. This draws
one uniform partition; exact expectation is a separate evaluation procedure.

The optional learned model uses only FEATURES at deployment. Fitting numerical
rewards derived from development labels is supervised, not attack-label-free.
"""
from __future__ import annotations

from itertools import combinations
import hashlib
import json
import math
import random

import networkx as nx
import numpy as np
from network_strategies import select as _network_select, group as _network_group

RNG_VERSION = 'topology-only-rng-v2'

SELECTORS = ('random', 'degree', 'pagerank', 'betweenness', 'mesa_static',
             'downstream_coverage', 'structural_path_pairs', 'learned_pairs')
GROUPERS = ('singleton', 'directed_path', 'random_matched_sizes', 'shared_receiver',
            'line_community', 'spectral', 'full_selected', 'learned_pairs')
FEATURES = ('intercept', 'group_size', 'directed_two_path', 'reciprocal_pair',
            'shared_sender', 'shared_receiver', 'shared_endpoint',
            'mean_sender_indegree', 'mean_sender_outdegree',
            'mean_recipient_indegree', 'mean_recipient_outdegree',
            'mean_recipient_reachability', 'recipient_reachability_overlap',
            'node_count', 'edge_density')


def _integer(value, name, minimum=None):
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f'{name} must be an integer' + (f' >= {minimum}' if minimum is not None else ''))
    return value


def stream_seed(seed, domain):
    """Stable independent-purpose seed; never Python's process-salted hash().

    Domains describe algorithmic operations only, not episodes, labels, names,
    content or outcomes. The serialization and version are part of the protocol.
    """
    _integer(seed, 'seed')
    if type(domain) is not str or not domain:
        raise ValueError('RNG domain must be a nonempty string')
    payload = json.dumps([RNG_VERSION, domain, seed], separators=(',', ':')).encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest(), 'big')


def validate_graph(graph):
    """Reject extra information rather than silently stripping it."""
    if type(graph) is not dict or set(graph) != {'nodes', 'edges'}:
        raise ValueError('graph must contain exactly anonymous nodes and edges')
    if not isinstance(graph['nodes'], (list, tuple)) or not isinstance(graph['edges'], (list, tuple)):
        raise ValueError('nodes and edges must be sequences')
    nodes = list(graph['nodes'])
    for node in nodes:
        _integer(node, 'node ID')
    if len(set(nodes)) != len(nodes):
        raise ValueError('duplicate node')
    edges = []
    for edge in graph['edges']:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError('each edge must contain exactly two integer endpoints')
        u, v = edge
        _integer(u, 'edge endpoint'); _integer(v, 'edge endpoint')
        if u not in nodes or v not in nodes:
            raise ValueError('edge endpoint absent from nodes')
        if u == v:
            raise ValueError('self-loop channels are unsupported')
        edges.append((u, v))
    if len(set(edges)) != len(edges):
        raise ValueError('duplicate edge')
    g = nx.DiGraph()
    g.add_nodes_from(sorted(nodes)); g.add_edges_from(sorted(edges))
    return g


def _edges(g, edges):
    if not isinstance(edges, (list, tuple)):
        raise ValueError('selected edges must be a sequence')
    result = []
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError('invalid selected edge')
        u, v = edge
        _integer(u, 'selected endpoint'); _integer(v, 'selected endpoint')
        if not g.has_edge(u, v):
            raise ValueError('selected edge absent from graph')
        result.append((u, v))
    if len(set(result)) != len(result):
        raise ValueError('duplicate selected edge')
    return sorted(result)


def _context(graph, seed):
    original = validate_graph(graph)
    _integer(seed, 'seed')
    names = sorted(original.nodes())
    random.Random(stream_seed(seed, 'reindex')).shuffle(names)
    to_original = dict(enumerate(names))
    to_internal = {node: i for i, node in to_original.items()}
    g = nx.DiGraph()
    g.add_nodes_from(range(len(names)))
    g.add_edges_from(sorted((to_internal[u], to_internal[v]) for u, v in original.edges()))
    return original, g, to_internal, to_original


def _two_path(a, b):
    return len(set(a + b)) == 3 and (a[1] == b[0] or b[1] == a[0])


def _features(g, edges):
    if not 1 <= len(edges) <= 2:
        raise ValueError('features require a singleton or pair')
    a, b = edges[0], edges[-1]
    paired = len(edges) == 2
    n = len(g); scale = max(1, n - 1)
    descendants = [nx.descendants(g, e[1]) for e in edges]
    mean = lambda f: sum(f(e) for e in edges) / len(edges)
    union = descendants[0] | descendants[-1]
    return np.asarray([
        1., len(edges) / 2., float(paired and _two_path(a, b)),
        float(paired and a == (b[1], b[0])),
        float(paired and a[0] == b[0]), float(paired and a[1] == b[1]),
        float(paired and bool(set(a) & set(b))),
        mean(lambda e: g.in_degree(e[0]) / scale),
        mean(lambda e: g.out_degree(e[0]) / scale),
        mean(lambda e: g.in_degree(e[1]) / scale),
        mean(lambda e: g.out_degree(e[1]) / scale),
        sum(map(len, descendants)) / (len(edges) * scale),
        len(descendants[0] & descendants[-1]) / max(1, len(union)) if paired else 0.,
        n / 20., nx.density(g)], dtype=float)


def pair_features(graph, group):
    """Fixed feature order; no ID, metadata or observed-content features."""
    g = validate_graph(graph)
    return _features(g, _edges(g, group))


def _coefficients(model):
    if type(model) is not dict or set(model) != {'features', 'coef', 'alpha', 'supervision'}:
        raise ValueError('model schema must contain only features, coef, alpha, supervision')
    if list(model['features']) != list(FEATURES):
        raise ValueError('model feature schema mismatch')
    if model['supervision'] != 'numerical development rewards; topology-only deployment':
        raise ValueError('model supervision declaration mismatch')
    if type(model['alpha']) not in (int, float) or not math.isfinite(model['alpha']) or model['alpha'] < 0:
        raise ValueError('invalid model alpha')
    if not isinstance(model['coef'], (list, tuple)) or len(model['coef']) != len(FEATURES):
        raise ValueError('model coefficient dimension mismatch')
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in model['coef']):
        raise ValueError('model coefficients must be finite numbers')
    return np.asarray(model['coef'], dtype=float)


def fit_pair_model(examples, alpha=1.):
    """Fit weighted ridge rewards supplied by an external development evaluator.

    Each example has exactly graph, group, reward, weight. Reward construction and
    case-balanced weights belong to the caller; no episode records enter here.
    Alpha is not selected internally. No held-out data or labels are read.
    """
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha < 0:
        raise ValueError('alpha must be finite and nonnegative')
    xs, ys, weights = [], [], []
    for example in examples:
        if type(example) is not dict or set(example) != {'graph', 'group', 'reward', 'weight'}:
            raise ValueError('training example must contain graph, group, reward, weight only')
        reward, weight = example['reward'], example['weight']
        if type(reward) not in (int, float) or not math.isfinite(reward):
            raise ValueError('reward must be finite')
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
            raise ValueError('weight must be finite and positive')
        xs.append(pair_features(example['graph'], example['group']))
        ys.append(reward); weights.append(weight)
    if not xs:
        raise ValueError('empty training data')
    x = np.asarray(xs); y = np.asarray(ys); w = np.asarray(weights)
    w /= w.sum()
    penalty = np.eye(len(FEATURES)) * alpha; penalty[0, 0] = 0.
    coef = np.linalg.lstsq(x.T @ (w[:, None] * x) + penalty, x.T @ (w * y), rcond=None)[0]
    return {'features': list(FEATURES), 'coef': coef.tolist(), 'alpha': float(alpha),
            'supervision': 'numerical development rewards; topology-only deployment'}


def _path_score(g, pair):
    f = _features(g, pair)
    # All directed two-paths are eligible. Degree and reachability are weak
    # structural preferences; no distinguished endpoint or role receives a bonus.
    return 1. + .25 * (f[8] + f[10]) + .25 * f[11]


def _acquire_pairs(g, k, seed, coef=None):
    rng = random.Random(seed)
    remaining = set(g.edges()); out = []
    while len(out) < k:
        room = k - len(out)
        candidates = [(e,) for e in sorted(remaining)]
        if room >= 2:
            candidates += [p for p in combinations(sorted(remaining), 2)
                           if coef is not None or _two_path(*p)]
        if coef is None:
            def score(c):
                # Prefer complete two-paths; singleton filling is budget padding.
                if len(c) == 2:
                    return (1., _path_score(g, c))
                e = c[0]
                return (0., g.degree(e[0]) + g.degree(e[1]) +
                        len(nx.descendants(g, e[1])) / max(1, len(g)))
        else:
            def score(c):
                return (float(_features(g, c) @ coef) / len(c),)
        values = [(score(c), rng.random(), c) for c in candidates]
        chosen = max(values, key=lambda v: (v[0], v[1]))[2]
        out.extend(chosen); remaining.difference_update(chosen)
    return sorted(out)


def select(graph, k, method='random', seed=0, model=None):
    """Select exactly k distinct edges; k above available edges is an error."""
    _, g, _, restore = _context(graph, seed)
    _integer(k, 'k', 0)
    if k > g.number_of_edges():
        raise ValueError('k exceeds graph edges')
    if method not in SELECTORS:
        raise ValueError('unknown selector')
    coef = _coefficients(model) if model is not None else None
    if method == 'learned_pairs' and coef is None:
        raise ValueError('learned_pairs requires a model')
    if k == 0:
        return []
    acquisition_seed = stream_seed(seed, 'selection:' + method)
    if method in ('structural_path_pairs', 'learned_pairs'):
        chosen = _acquire_pairs(g, k, acquisition_seed, coef if method == 'learned_pairs' else None)
    else:
        chosen = _network_select(g, k, method, acquisition_seed)
    return sorted((restore[u], restore[v]) for u, v in chosen)


def _path_groups(g, selected, s, seed, coef=None):
    """Greedily merge disjoint groups using structural links, capped by s.

    With s=2 these are pairs; for larger s, connected two-path affinity can form
    branching groups as well as simple chains. This is not a Hamiltonian-path rule.
    """
    rng = random.Random(seed)
    groups = [[e] for e in selected]
    while True:
        candidates = []
        for i, a in enumerate(groups):
            for j in range(i):
                b = groups[j]
                if len(a) + len(b) > s:
                    continue
                if coef is None:
                    links = [(x, y) for x in a for y in b if _two_path(x, y)]
                    score = sum(_path_score(g, p) for p in links)
                else:
                    # Learned deployment groups remain singleton/pair views.
                    if len(a) + len(b) > 2:
                        continue
                    score = (float(_features(g, a + b) @ coef) -
                             float(_features(g, a) @ coef) - float(_features(g, b) @ coef))
                if score > 0:
                    candidates.append((score, rng.random(), i, j))
        if not candidates:
            break
        _, _, i, j = max(candidates)
        merged = sorted(groups[i] + groups[j])
        groups = [c for index, c in enumerate(groups) if index not in (i, j)] + [merged]
    return groups


def group(graph, selected, s, method='directed_path', seed=0, model=None):
    original, g, anonymous, restore = _context(graph, seed)
    _integer(s, 's', 1)
    if method not in GROUPERS:
        raise ValueError('unknown grouper')
    checked = _edges(original, selected)
    edges = sorted((anonymous[u], anonymous[v]) for u, v in checked)
    coef = _coefficients(model) if model is not None else None
    if method == 'learned_pairs' and coef is None:
        raise ValueError('learned_pairs requires a model')
    if not edges:
        return []
    if method == 'full_selected':
        return [checked] if len(edges) <= s else None
    if method == 'singleton' or s == 1:
        groups = [[e] for e in edges]
    elif method in ('directed_path', 'learned_pairs'):
        groups = _path_groups(g, edges, s, stream_seed(seed, 'group:' + method),
                              coef if method == 'learned_pairs' else None)
    elif method == 'random_matched_sizes':
        # The reference uses the SAME directed-path stream as its own arm.
        # Only membership randomization gets a separate, independent-purpose stream.
        sizes = sorted(map(len, _path_groups(g, edges, s, stream_seed(seed, 'group:directed_path'))))
        order = list(edges)
        random.Random(stream_seed(seed, 'group:matched_random_shuffle')).shuffle(order)
        groups = []; offset = 0
        for size in sizes:
            groups.append(order[offset:offset + size]); offset += size
    else:
        groups = _network_group(g, edges, s,
                                'receiver' if method == 'shared_receiver' else method,
                                stream_seed(seed, 'group:' + method))
    result = sorted(sorted((restore[u], restore[v]) for u, v in part) for part in groups)
    if sorted(e for part in result for e in part) != checked or any(not 1 <= len(part) <= s for part in result):
        raise AssertionError('invalid partition')
    return result


def allocate(graph, k, s, selector='random', grouper='directed_path', seed=0, model=None):
    selected = select(graph, k, selector, seed, model)
    groups = group(graph, selected, s, grouper, seed, model)
    return {'selector': selector, 'grouper': grouper, 'k': k, 's': s, 'seed': seed,
            'selected': selected, 'groups': groups,
            'group_sizes': sorted(map(len, groups)) if groups is not None else None,
            'feasible_group_cap': groups is not None,
            'information_access': 'anonymous directed topology only'}
