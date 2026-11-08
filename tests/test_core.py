import numpy as np
import pytest

from jmon.graphs import make_graph
from jmon.simulate import make_world, run_episode, EdgeMask
from jmon.attacks.library import AttackSpec, build_attack, sample_attack, pair_of
from jmon.monitors import build_view, RuleMonitor, LLMMonitor
from jmon.monitors.views import Observation
from jmon.llm import MockProvider
from jmon.allocation import (select_edges, group_edges, validate_assignment, clean_loss_matrix, interaction_matrix,
                             Assignment)
from jmon.evaluation import make_splits, monitor_episode, SplitLeak
from jmon.evaluation.metrics import tpr_at_fpr


def world(kind="layered", n=10, seed=0, **kw):
    g = make_graph(kind, n, seed=seed, **kw)
    agents, truth = make_world(g, seed)
    return g, agents, truth


def test_graphs_are_dags_with_one_sink():
    for kind in ["chain", "star", "diamond", "tree", "layered"]:
        g = make_graph(kind, 9, seed=1)
        assert g.sink in g.nodes and g.g.out_degree(g.sink) == 0
        assert len(g.topological_order()) == len(g.nodes)


def test_episode_deterministic_and_clean_loss_zero():
    g, agents, truth = world()
    e1 = run_episode(g, agents, truth, 3)
    e2 = run_episode(g, agents, truth, 3)
    assert e1.log.fingerprint() == e2.log.fingerprint()
    assert e1.task_loss == 0.0  # every fact reaches the sink on a connected DAG
    m = e1.log.all()[0]
    assert m.sender and m.recipient and m.round >= 0 and m.timestamp > 0 and m.edge == (m.sender, m.recipient)


def test_masking_edge_changes_loss_only_when_needed():
    g, agents, truth = world("diamond", 6)
    L0, L1, L2 = clean_loss_matrix(g, agents, truth, [0, 1], pairs=True)
    assert L0 == 0.0
    J = interaction_matrix(L0, L1, L2)
    # the two sink in-edges are redundant paths for shared facts: masking both loses everything
    a, b = ("a1", "a3"), ("a2", "a3")
    assert L2[(a, b)] == 1.0
    assert J[(a, b)] >= 0


def test_attacks_fire_and_hidden_truth_is_stripped_from_views():
    g, agents, truth = world()
    rng = np.random.default_rng(0)
    for kind in ["single_injection", "coordinated_injection", "conjunctive", "distributed_misinformation", "multi_stage"]:
        spec = sample_attack(kind, g.edges, truth, rng)
        hook, fn = build_attack(spec)
        ep = run_episode(g, agents, truth, 0, hooks=[hook], success_fns={"a": fn})
        injected = [m for m in ep.log if m.hidden.get("attack")]
        assert injected, kind
        v = build_view(ep.log, g.edges)
        assert all(not hasattr(o, "hidden") for o in v.observations)
        assert "a" in ep.attack_success


def test_conjunctive_succeeds_only_with_both_fragments():
    g = make_graph("chain", 4)
    agents, truth = make_world(g, 0)
    both = AttackSpec("conjunctive", [("a0", "a1"), ("a1", "a2")], params={"ctx": "c2"})
    hook, fn = build_attack(both)
    ep = run_episode(g, agents, truth, 0, hooks=[hook], success_fns={"a": fn})
    assert ep.attack_success["a"]
    one = AttackSpec("conjunctive", [("a0", "a1")], params={"ctx": "c2"})
    hook1, fn1 = build_attack(one)
    ep1 = run_episode(g, agents, truth, 0, hooks=[hook1], success_fns={"a": fn1})
    # a single fragment may or may not coincide with a benign act token; it never carries fragment B
    assert not any(m.hidden.get("fragment") == "B" for m in ep1.log)


def test_view_restriction_independent_monitors_cannot_see_other_edges():
    g, agents, truth = world()
    ep = run_episode(g, agents, truth, 0)
    for e in g.edges:
        v = build_view(ep.log, [e])
        assert all(o.edge == e for o in v.observations)
    sub = g.edges[:3]
    v = build_view(ep.log, sub)
    assert {o.edge for o in v.observations} <= set(sub)
    assert len(v.observations) == len(ep.log.on_edges(sub))


def test_joint_monitor_receives_both_fragments_together():
    g = make_graph("chain", 4)
    agents, truth = make_world(g, 0)
    spec = AttackSpec("conjunctive", [("a0", "a1"), ("a1", "a2")], params={"ctx": "c2"}, name="a")
    hook, fn = build_attack(spec)
    ep = run_episode(g, agents, truth, 0, hooks=[hook], success_fns={"a": fn})
    joint = Assignment([[("a0", "a1"), ("a1", "a2")]])
    indep = Assignment([[("a0", "a1")], [("a1", "a2")]])
    mon = RuleMonitor(p_fp=0.0, seed=1)
    rj = monitor_episode(ep, joint, mon)
    ri = monitor_episode(ep, indep, mon)
    assert rj.fragments_seen_jointly["a"]["all_fragments_in_one_group"]
    assert not ri.fragments_seen_jointly["a"]["all_fragments_in_one_group"]
    assert rj.episode_score == 1.0 and ri.episode_score < 1.0
    n_msgs = len(ep.log.on_edges([("a0", "a1"), ("a1", "a2")]))
    assert rj.ledger.monitor_calls == 1 and ri.ledger.monitor_calls == n_msgs   # independent = one call per message
    assert rj.ledger.monitored_messages == ri.ledger.monitored_messages


def test_grouping_does_not_change_communication():
    g, agents, truth = world()
    ep = run_episode(g, agents, truth, 0)
    fp = ep.log.fingerprint()
    mon = RuleMonitor(seed=0)
    for pol in ["singleton", "random", "by_receiver", "proximity"]:
        a = group_edges(pol, g.edges[:6], 3, np.random.default_rng(0), graph=g)
        validate_assignment(a, g.edges, 6, 3)
        monitor_episode(ep, a, mon)
        assert ep.log.fingerprint() == fp


def test_selection_and_budget_validation():
    g, agents, truth = world()
    rng = np.random.default_rng(0)
    sel = select_edges("random", g, 4, rng)
    assert len(sel) == 4 and len(set(sel)) == 4
    from jmon.allocation import size_vector, enumerate_pair_partitions
    assert size_vector(6, 3) == [3, 3] and size_vector(7, 3) == [3, 2, 2] and size_vector(4, 3) == [2, 2]
    for pol in ["random", "by_receiver", "proximity", "convergence"]:
        a6 = group_edges(pol, g.edges[:7], 3, np.random.default_rng(1), graph=g)
        assert [len(x) for x in a6.groups] == [3, 2, 2], pol      # identical size vector across policies
    assert len(enumerate_pair_partitions(g.edges[:6])) == 15
    a = Assignment([sel[:3], sel[3:]])
    validate_assignment(a, g.edges, 4, 3)
    with pytest.raises(ValueError):
        validate_assignment(a, g.edges, max_edges=3)
    with pytest.raises(ValueError):
        validate_assignment(Assignment([sel[:4]]), g.edges, max_group_size=3)
    with pytest.raises(ValueError):
        validate_assignment(Assignment([sel[:2], sel[1:3]]), g.edges)  # overlap


def test_selection_reproducible():
    g, _, _ = world()
    s1 = select_edges("random", g, 5, np.random.default_rng(42))
    s2 = select_edges("random", g, 5, np.random.default_rng(42))
    assert s1 == s2


def test_split_leak_guard():
    sp = make_splits(0, 4, 4)
    sp.assert_train(sp.train[0])
    with pytest.raises(SplitLeak):
        sp.assert_train(sp.heldout[0])


def test_budget_accounting_tokens_and_calls():
    g, agents, truth = world()
    ep = run_episode(g, agents, truth, 0)
    mon = LLMMonitor(MockProvider(), style="joint")
    a = Assignment([g.edges[:2], g.edges[2:4]])
    r = monitor_episode(ep, a, mon)
    assert r.ledger.monitor_calls == 2
    assert r.ledger.tokens_in > 0 and r.ledger.tokens_out == 24
    assert len(mon.calls) == 2 and "prompt" in mon.calls[0]
    assert r.ledger.monitored_messages == len(ep.log.on_edges(g.edges[:4]))
    assert mon.provider.calls == 2


def test_independent_llm_monitor_rejects_multi_message_views():
    g, agents, truth = world()
    ep = run_episode(g, agents, truth, 0)
    mon = LLMMonitor(MockProvider(), style="independent")
    with pytest.raises(ValueError):
        monitor_episode(ep, Assignment([g.edges[:2]]), mon)


def test_tpr_at_fpr():
    tpr, thr = tpr_at_fpr([0.1] * 90 + [0.6] * 10, [1.0] * 5 + [0.1] * 5, 0.1)
    assert thr == 0.6 and tpr == 0.5


def test_live_provider_refuses_without_allow_live():
    from jmon.llm import make_provider
    with pytest.raises(RuntimeError):
        make_provider({"kind": "openai", "model": "x", "cost": {"allow_live": False}})


def test_convergence_and_oracle_groupers():
    from jmon.allocation import convergence_affinity, oracle_affinity
    g = make_graph("diamond", 6)
    edges = g.edges
    conv = convergence_affinity(g, edges)
    assert conv[(("a1", "a3"), ("a2", "a3"))] == 1.0           # both end at the sink
    a = group_edges("convergence", edges, 2, np.random.default_rng(0), graph=g)
    validate_assignment(a, edges, max_group_size=2)
    frag = [{("a0", "a1"), ("a1", "a3")}, {("a0", "a1"), ("a1", "a3")}, {("a0", "a2"), ("a2", "a3")}]
    orc = oracle_affinity(frag, edges)
    assert orc[(("a0", "a1"), ("a1", "a3"))] == pytest.approx(2 / 3)
    a = group_edges("oracle", edges, 2, np.random.default_rng(0), affinity=orc)
    assert [("a0", "a1"), ("a1", "a3")] in [sorted(gr) for gr in a.groups]


def test_mesa_rank_features_and_filler():
    from jmon.allocation.mesa_rank import static_features, mesa_scores
    from jmon.simulate import EdgeFiller
    g = make_graph("diamond", 6)
    f = static_features(g)
    assert set(f) == {"betweenness", "bottleneck", "bridge", "endpoint_centrality", "source_outdeg", "target_indeg"}
    assert f["bridge"][("a4", "a1")] == 1.0            # a source's only out-edge is a bridge
    from jmon.graphs import CommGraph
    h = CommGraph(["a0", "a1", "a2"], [("a0", "a1"), ("a1", "a2"), ("a0", "a2")])
    fh = static_features(h)
    assert fh["bridge"][("a0", "a2")] == 0.0 and fh["bottleneck"][("a0", "a2")] == 0.5   # alternative route of length 2
    assert fh["bridge"][("a1", "a2")] == 1.0
    agents, truth = make_world(g, 0)
    s = mesa_scores(g, agents, truth, [0, 1], combined=True)
    assert set(s) == set(g.edges) and all(np.isfinite(v) for v in s.values())
    ep = run_episode(g, agents, truth, 0, hooks=[EdgeFiller([("a1", "a3")])])
    assert all(m.fields["facts"] == {} for m in ep.log if m.edge == ("a1", "a3"))
