"""Prepare anonymous-topology workload allocations; never dispatch model calls.

Only the adapter reads named graph nodes/edges. A fixed, public anonymization seed
is independent of episodes, attack placement, observations, roles and labels.
Identical named graphs get identical mappings; labels identify endpoints for the
reversible mapping only. Every policy call receives exactly anonymous integer nodes
and directed edges. Names are restored after selection/grouping, for unchanged
judge prompts and exact-cache reuse. The judge can still read workflow semantics:
this restriction governs the allocator, not the evidence it purchases.

Input-byte feasibility is computed AFTER the allocation. It never changes a choice
or silently truncates history. Privileged location oracles are separate, explicitly
marked rows; their evaluator labels never enter a prospective allocation.

Return contract: plan_library(records, system, model=None) returns jobs,
assignments, expectations, compatible with record_workload_benchmark.analyze.
Exact expectation evaluation reuses record_workload_exact_random.evaluate_expected.
No learned weights are fit here. These are development plans, not held-out results.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
import random

import topology_only_policies as topology
from record_workflow import canonical
from record_strategies import policy_id
from monitor_record_workflows import FORMAT_RETRY, view_prompt
from record_workload_exact_random import weighted_subsets, _validate_weights, GROUPER

ANONYMIZATION_SEED = 20261106
PRIMARY_ALLOWANCE = 24000
RELAXED_ALLOWANCE = 60000
SELECTORS = ('random', 'degree', 'pagerank', 'betweenness', 'mesa_static',
             'downstream_coverage', 'structural_path_pairs')
GROUPERS = ('directed_path', 'random_matched_sizes', 'shared_receiver',
            'line_community', 'spectral')


def _named_graph(graph):
    """The adapter deliberately copies ONLY endpoint identities and adjacency."""
    if not isinstance(graph, dict) or not {'nodes', 'edges'} <= set(graph):
        raise ValueError('named graph must supply nodes and edges')
    if not isinstance(graph['nodes'], (list, tuple)) or not isinstance(graph['edges'], (list, tuple)):
        raise ValueError('nodes and edges must be sequences')
    nodes = list(graph['nodes'])
    if any(type(n) not in (str, int) for n in nodes) or len(set(nodes)) != len(nodes):
        raise ValueError('invalid or duplicate opaque source identities')
    edges = []
    for edge in graph['edges']:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError('edge must have exactly two endpoint identities')
        if any(type(n) not in (str, int) or n not in nodes for n in edge):
            raise ValueError('unknown or invalid endpoint')
        edges.append(tuple(edge))
    if len(set(edges)) != len(edges):
        raise ValueError('duplicate edge')
    # Canonical serialization order serves reproducibility, not a score/tie rule.
    return {'nodes': sorted(nodes, key=canonical),
            'edges': [list(e) for e in sorted(edges, key=canonical)]}


def anonymize_graph(graph, seed=ANONYMIZATION_SEED):
    """Return (strict integer graph, integer-to-original endpoint mapping)."""
    if type(seed) is not int:
        raise ValueError('anonymization seed must be an integer')
    named = _named_graph(graph)
    shuffled = list(named['nodes']); random.Random(seed).shuffle(shuffled)
    restore = dict(enumerate(shuffled))
    anonymous = {name: i for i, name in restore.items()}
    public = {'nodes': list(range(len(shuffled))),
              'edges': sorted([[anonymous[u], anonymous[v]] for u, v in named['edges']])}
    topology.validate_graph(public)
    return public, restore


def _restore_policy(p, restore):
    def edges(es):
        return sorted([(restore[u], restore[v]) for u, v in es], key=canonical)
    result = dict(p)
    result['selected'] = edges(p['selected'])
    if p['groups'] is None:
        raise ValueError('requested planner grid must fit its declared group cap')
    result['groups'] = sorted([edges(g) for g in p['groups']], key=canonical)
    result['comparison_scope'] = 'prospective_topology_only'
    return result


@lru_cache(maxsize=256)
def _catalog(named_json, model_json, seeds, anonymization_seed):
    graph, restore = anonymize_graph(json.loads(named_json), anonymization_seed)
    model = json.loads(model_json) if model_json is not None else None
    rows = []
    if len(graph['edges']) != 13:
        raise ValueError('this frozen workload grid requires exactly 13 directed channels')
    for seed in seeds:
        for k, selectors in ((7, SELECTORS), (4, ('random', 'structural_path_pairs'))):
            for selector in selectors:
                chosen = topology.select(graph, k, selector, seed)
                methods = [('singleton', 1)] + [(name, s) for s in (2, 4) for name in GROUPERS]
                methods.append(('full_selected', k))
                for method, cap in methods:
                    groups = topology.group(graph, chosen, cap, method, seed)
                    p = {'selector': selector, 'grouper': method, 'k': k, 's': cap,
                         'seed': seed, 'selected': chosen, 'groups': groups,
                         'information_access': 'anonymous directed topology only'}
                    row = _restore_policy(p, restore); rows.append(row)
                    if k == 7 and method == 'full_selected':
                        rows.append({**row, 'grouper': 'full_selected_relaxed',
                                     'input_byte_allowance': RELAXED_ALLOWANCE,
                                     'comparison_scope': 'relaxed_input_reference_topology_only'})
        if model is not None:
            row = topology.allocate(graph, 7, 2, 'learned_pairs', 'learned_pairs', seed, model)
            rows.append(_restore_policy(row, restore))
    all_edges = sorted(map(tuple, graph['edges']))
    full = _restore_policy({'selector': 'all_edges', 'grouper': 'full_selected',
                            'k': 13, 's': 13, 'seed': 0, 'selected': all_edges,
                            'groups': [all_edges],
                            'information_access': 'anonymous directed topology only'}, restore)
    full['comparison_scope'] = 'full_network_reference'
    rows += [full, {**full, 'grouper': 'full_selected_relaxed',
                    'input_byte_allowance': RELAXED_ALLOWANCE,
                    'comparison_scope': 'relaxed_full_network_reference'}]
    # Cache immutable serialization; callers cannot poison later episodes' rows.
    return canonical(rows)


def policies(graph, model=None, *, seeds=(0, 1, 2), anonymization_seed=ANONYMIZATION_SEED):
    """Generate mapped prospective/reference plans, reading graph adjacency only."""
    seeds = tuple(seeds)
    if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be nonempty unique integers')
    if model is not None:
        topology._coefficients(model)  # Reject old role-bearing schemas eagerly.
    yield from json.loads(_catalog(canonical(_named_graph(graph)),
                                   canonical(model) if model is not None else None,
                                   seeds, anonymization_seed))


def oracle_policies(record):
    """Privileged diagnostic targets, including clean counterfactual cases."""
    labels = record['labels']; target = labels['case_labels'][labels['target_case_id']]
    field = labels['target_field']
    report = tuple(target['report_edges'][field]); reference = tuple(target['reference_edges'][field])
    for name, es in (('target_location_oracle', [report]),
                     ('reference_location_oracle', sorted([reference, report]))):
        yield {'selector': name, 'grouper': 'diagnostic', 'k': len(es), 's': len(es),
               'seed': 0, 'selected': es, 'groups': [es],
               'information_access': 'privileged evaluation-only target/reference labels',
               'comparison_scope': 'privileged_location_oracle'}


def _view(record, system, edges):
    edges = sorted(map(tuple, edges))
    prompt = view_prompt(record, edges)
    return (hashlib.sha256(canonical([system, prompt]).encode()).hexdigest(),
            len((system + FORMAT_RETRY + prompt).encode()))


def _check_partition(record, p):
    selected = list(map(tuple, p['selected']))
    flat = [tuple(e) for part in p['groups'] for e in part]
    universe = set(map(tuple, record['graph']['edges']))
    if len(flat) != len(set(flat)) or sorted(flat) != sorted(selected) or len(selected) != p['k']:
        raise ValueError('invalid partition or acquisition count')
    if not set(selected) <= universe or any(not 1 <= len(part) <= p['s'] for part in p['groups']):
        raise ValueError('invalid selected edge or group cap')


def _expectation(record, system, baseline, view_info):
    selected = sorted(map(tuple, baseline['selected']))
    sizes = sorted(map(len, baseline['groups']))
    views = []
    for edges, weight in weighted_subsets(selected, sizes):
        key, nbytes = view_info(edges)
        if nbytes > PRIMARY_ALLOWANCE:
            raise ValueError('positive-probability exact-random view exceeds 24000 bytes; do not truncate its distribution')
        views.append({'edges': [list(e) for e in edges], 'weight': float(weight),
                      'weight_numerator': weight.numerator, 'weight_denominator': weight.denominator,
                      'view_key': key, 'input_bytes': nbytes})
    denominator = math.prod(math.factorial(j) ** count * math.factorial(count)
                            for j, count in Counter(sizes).items())
    p = {'episode_id': record['id'], 'selector': 'structural_path_pairs', 'grouper': GROUPER,
         'k': 7, 's': 2, 'seed': baseline['seed'], 'selected': [list(e) for e in selected],
         'group_sizes': sizes, 'weighted_views': views, 'view_keys': [v['view_key'] for v in views],
         'input_byte_allowance': PRIMARY_ALLOWANCE, 'feasible': True,
         'system_sha256': hashlib.sha256(system.encode()).hexdigest(),
         'expected_calls': len(sizes), 'expected_inspected_edges': 7,
         'randomization_partitions': math.factorial(7) // denominator,
         'expectation_not_a_single_execution': True,
         'information_access': 'anonymous directed topology only',
         'comparison_scope': 'prospective_topology_only_exact_expectation',
         'scope': 'Uniform matched-size partition expectation on exactly the directed_path selection; not one execution reading the candidate library.'}
    p['policy'] = policy_id(p)
    marginals, _, _ = _validate_weights(p)
    p['edge_inclusion_probabilities'] = [{'edge': list(e), 'probability': float(w)} for e, w in marginals.items()]
    return p


def plan_library(records, system, model=None, *, include_oracles=True, seeds=(0, 1, 2),
                 anonymization_seed=ANONYMIZATION_SEED):
    """Pure plan: exact old judge hashes, no file writes, providers, or spending."""
    seeds = tuple(seeds); jobs = {}; assignments = []; expectations = []; record_ids = set()
    for record in records:
        if record['id'] in record_ids:
            raise ValueError('duplicate episode ID')
        record_ids.add(record['id'])
        mids = [m['id'] for m in record['messages']]
        if len(mids) != len(set(mids)):
            raise ValueError('duplicate message IDs')
        rows = list(policies(record['graph'], model, seeds=seeds, anonymization_seed=anonymization_seed))
        if include_oracles:
            rows += list(oracle_policies(record))
        seen = set(); views = {}; matches = {}; exact_baselines = []
        def view_info(edges):
            group_key = tuple(sorted(map(tuple, edges)))
            if group_key not in views:
                views[group_key] = _view(record, system, group_key)
            return views[group_key]
        for p in rows:
            _check_partition(record, p)
            pid = policy_id(p)
            if pid in seen:
                raise ValueError('duplicate policy identity')
            seen.add(pid)
            selected = sorted(map(tuple, p['selected']))
            groups = [sorted(map(tuple, part)) for part in p['groups']]
            keys, sizes = zip(*(view_info(part) for part in groups))
            allowance = p.get('input_byte_allowance', PRIMARY_ALLOWANCE)
            if allowance not in (PRIMARY_ALLOWANCE, RELAXED_ALLOWANCE):
                raise ValueError('undeclared input allowance')
            feasible = all(size <= allowance for size in sizes)
            if feasible:
                for key, part in zip(keys, groups):
                    jobs.setdefault(key, (record, part))
            assignment = {'episode_id': record['id'], 'policy': pid, **p,
                          'selected': selected, 'groups': groups, 'view_keys': list(keys),
                          'group_input_bytes': list(sizes), 'input_byte_allowance': allowance,
                          'feasible': feasible, 'anonymization_seed': anonymization_seed}
            assignments.append(assignment)
            if p['grouper'] in ('directed_path', 'random_matched_sizes'):
                matches.setdefault((p['selector'], p['k'], p['s'], p['seed']), {})[p['grouper']] = assignment
            if p['selector'] == 'structural_path_pairs' and p['k'] == 7 and p['s'] == 2 and p['grouper'] == 'directed_path':
                exact_baselines.append(assignment)
        for arms in matches.values():
            a, b = arms['directed_path'], arms['random_matched_sizes']
            if a['selected'] != b['selected'] or sorted(map(len, a['groups'])) != sorted(map(len, b['groups'])):
                raise ValueError('unmatched randomized comparison')
        for baseline in exact_baselines:
            p = _expectation(record, system, baseline, view_info); expectations.append(p)
            for view in p['weighted_views']:
                jobs.setdefault(view['view_key'], (record, list(map(tuple, view['edges']))))
    return jobs, assignments, expectations
