"""Structural pass (no model calls) on the summarising-relay model with attacks anywhere.

For each swarm, case and attack placement (attack modes balanced across records), run the
graph-only policies on the anonymous agent-to-agent graph and check, per fragment, whether a
judge group holds a channel carrying the false version of the attacked check together with a
channel carrying its true version. Also records WHERE that evidence sits: the agents whose
inputs contain both versions ('meeting points'), as hop distance to the decider.
Codex's policies are imported unchanged; two baselines watch the decider's inputs and the last
layers (backward breadth-first from the decider)."""
from __future__ import annotations
import itertools, json, math, os, random, sys, collections
from multiprocessing import Pool
import networkx as nx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, ROOT)
import topology_neighborhood_policies as nb          # Codex's code, unchanged
import topology_only_policies as legacy               # Codex's code, unchanged
from swarm.topology import FAMILIES, SOURCES
from swarm.case import make_case, assign, simulate, sample_attack, CHECKS, check_results

POLICIES = [("parent_complete", None, "nb"), ("bounded_cone", None, "nb"), ("structural_path_pairs", "directed_path", "nb"),
            ("path_coverage", "directed_path", "nb"), ("degree", "directed_path", "nb"), ("random", "directed_path", "nb"),
            ("random", "random_matched_sizes", "nb"), ("random", "shared_receiver", "legacy"),
            ("degree", "shared_receiver", "legacy"), ("last_layers", "shared_receiver", "own"),
            ("ego_degree", "ego", "own"), ("ego_random", "ego", "own")]
MODES = list(itertools.product(("worker", "relay", "channel"), repeat=2))


def anonymous(G, seed):
    agents = [n for n, d in G.nodes(data=True) if d["kind"] != "source"]
    ids = list(range(len(agents))); random.Random(seed).shuffle(ids)
    m = dict(zip(agents, ids))
    return {"nodes": sorted(ids), "edges": sorted([m[u], m[v]] for u, v in G.edges() if u in m and v in m)}, m


def last_layers(graph, k, s):
    """Backward BFS from the sink(s): the decider's inputs first, then the layer before, up to k;
    grouped by shared receiver in chunks of s. Uses topology only."""
    g = nx.DiGraph(); g.add_nodes_from(graph["nodes"]); g.add_edges_from(map(tuple, graph["edges"]))
    sinks = [n for n in g if g.out_degree(n) == 0 and g.in_degree(n) > 0]
    chosen, frontier, seen = [], list(sinks), set(sinks)
    while frontier and len(chosen) < k:
        nxt = []
        for v in frontier:
            for u in sorted(g.predecessors(v)):
                if len(chosen) < k: chosen.append((u, v))
                if u not in seen: seen.add(u); nxt.append(u)
        frontier = nxt
    by_recv = collections.defaultdict(list)
    for e in chosen: by_recv[e[1]].append(e)
    return [grp[i:i + s] for grp in by_recv.values() for i in range(0, len(grp), s)]


def ego(graph, k, s, order="degree", seed=0):
    """Each agent's input and output channels read together (chunks of s), agents taken in
    degree order or random order, until k channels are acquired. Topology only."""
    g = nx.DiGraph(); g.add_nodes_from(graph["nodes"]); g.add_edges_from(map(tuple, graph["edges"]))
    nodes = sorted(g, key=lambda n: (-g.degree(n), n)) if order == "degree" else random.Random(seed).sample(sorted(g), g.number_of_nodes())
    taken, groups = set(), []
    for n in nodes:
        if len(taken) >= k: break
        es = [e for e in list(g.in_edges(n)) + list(g.out_edges(n)) if e not in taken][: k - len(taken)]
        for i in range(0, len(es), s):
            groups.append(es[i:i + s]); taken.update(es[i:i + s])
    return groups


def allocate(graph, k, s, sel, grp, impl, seed):
    if impl == "own" and sel == "ego_degree": return ego(graph, k, s, "degree", seed)
    if impl == "own" and sel == "ego_random": return ego(graph, k, s, "random", seed)
    if impl == "own": return last_layers(graph, k, s)
    if impl == "nb": return nb.allocate(graph, k, s, selector=sel, grouper=grp, seed=seed)["groups"]
    a = legacy.allocate(graph, k, s, selector=sel, grouper=grp, seed=seed)
    return a["groups"] if isinstance(a, dict) else a


def one(job):
    fam, tseed, cseed, assurance = job
    G = FAMILIES[fam](seed=tseed, assurance=assurance); rng = random.Random(cseed)
    owner, cross = assign(G, rng)
    pair = rng.sample(list(CHECKS), 2)
    modes = MODES[cseed % len(MODES)]
    f = make_case(rng, pair)
    atks, rejected = sample_attack(G, f, owner, cross, pair, rng, modes=modes)
    if atks is None:
        return [{"family": fam, "assurance": assurance, "tseed": tseed, "cseed": cseed, "modes": "+".join(modes), "infeasible": True}]
    msgs, dec, _ = simulate(G, f, owner, cross, atks)
    truth = check_results(f, set(SOURCES))
    def version(m, c):
        out = set()
        if c in m["checks"]: out.add("false" if m["checks"][c][0] != truth[c] else "true")
        src = CHECKS[c]
        for k in SOURCES[src]:
            if k in m["facts"] and k in ("invoice_amount", "bank_change_verified", "vendor_verified", "sanctions_hit", "goods_received"):
                out.add("false" if m["facts"][k] != f[k] else "true")
        return out
    tags = {e: {a["check"]: version(m, a["check"]) for a in atks} for e, m in msgs.items()}
    # where the evidence sits: receivers whose inputs hold both versions
    dist = nx.single_target_shortest_path_length(G, "decider")
    dist = dict(dist)
    meet = {}
    for a in atks:
        c = a["check"]; pts = []
        for v in G:
            ins = [tags[(u, v)].get(c, set()) for u in G.predecessors(v) if (u, v) in tags]
            outs = [tags[(v, w)].get(c, set()) for w in G.successors(v) if (v, w) in tags]
            if any("false" in x for x in ins) and any("true" in x for x in ins): pts.append(v)
            elif any("true" in x for x in ins) and any("false" in x for x in outs): pts.append(v)
        meet[c] = sorted(dist.get(v, 99) for v in pts)
    graph, m = anonymous(G, cseed + 17)
    inv = {v: k for k, v in m.items()}
    E = len(graph["edges"])
    rows = []
    base = {"family": fam, "assurance": assurance, "tseed": tseed, "cseed": cseed, "modes": "+".join(modes), "E": E, "decision_AB": dec,
            "rejected": rejected, "meeting_depths": meet, "min_meeting_depth": min((min(v) for v in meet.values() if v), default=None)}
    for kfrac, s in itertools.product((0.125, 0.25, 0.5), (2, 4, 8)):
        k = max(s, math.ceil(kfrac * E))
        for sel, grp, impl in POLICIES:
            try:
                groups = allocate(graph, k, s, sel, grp, impl, cseed + 991)
            except Exception as ex:
                rows.append({**base, "policy": f"{sel}/{grp or 'default'}", "k": k, "kfrac": kfrac, "s": s, "error": str(ex)[:80]}); continue
            if groups is None: continue
            seen = 0
            for a in atks:
                c = a["check"]
                seen += any(any("false" in tags[(inv[e[0]], inv[e[1]])][c] for e in g) and
                            any("true" in tags[(inv[e[0]], inv[e[1]])][c] for e in g) for g in groups)
            rows.append({**base, "policy": f"{sel}/{grp or 'default'}", "k": k, "kfrac": kfrac, "s": s,
                         "contradiction_seen": seen / len(atks), "both_seen": float(seen == len(atks))})
    return rows


if __name__ == "__main__":
    jobs = [(fam, t, 1000 * t + c, a) for fam in FAMILIES for t in range(4) for c in range(18) for a in ("line", "decider")]
    with Pool(min(10, os.cpu_count())) as p:
        rows = [r for rs in p.map(one, jobs) for r in rs]
    json.dump(rows, open(os.path.join(ROOT, "swarm/out/structural.json"), "w"))
    print(len(jobs), "records", len(rows), "rows", sum("error" in r for r in rows), "errors",
          sum(r.get("infeasible", False) for r in rows), "infeasible")
