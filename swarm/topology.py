"""Swarm generators. Each returns a networkx DiGraph that is a DAG (agents run once, in
topological order). Node attrs: kind in {source, agent, decider}, role, team. Every edge is
a one-to-one channel. Facts originate at sources; each source fact is read by several
workers so that it reaches the decider along more than one route (redundancy).

Families (modelled on documented systems):
  orchestrator  dispatcher -> sub-orchestrators -> workers -> aggregators -> decider
                (orchestrator-worker decomposition, as in GTG-1002), plus peer and QA links
  teams         several teams (lead + specialists, intra-team peer links), leads report to
                a committee; an audit team receives copies across teams (MESA hybrid style)
  pipeline      intake -> analysts -> overlapping reviewers -> approvers -> decider, with a
                compliance desk that reads sources directly (review/approval workflow)
"""
from __future__ import annotations
import random
import networkx as nx

# A vendor-payment approval case: sources hold facts; the decider applies a rule.
SOURCES = {
    "erp_ledger": ["invoice_amount", "po_amount", "approval_limit", "payment_terms"],
    "vendor_registry": ["vendor_verified", "vendor_country", "vendor_since"],
    "contract_db": ["contract_end", "contract_owner", "contract_value"],
    "sanctions_list": ["sanctions_hit"],
    "bank_master": ["bank_account_changed", "bank_change_verified", "bank_country"],
    "procurement": ["goods_received", "receiving_dept"],
}
FACTS = [f for fs in SOURCES.values() for f in fs]
WORKER_ROLES = {
    "erp_ledger": "accounts-payable clerk", "vendor_registry": "vendor-onboarding checker",
    "contract_db": "contract analyst", "sanctions_list": "sanctions screener",
    "bank_master": "bank-details verifier", "procurement": "goods-receipt checker",
}


def _source_nodes(G):
    for s in SOURCES:
        G.add_node(s, kind="source", role=s.replace("_", " "), team="data")


def _assign_sources(workers, rng, cross_check=0.25, team_of=None):
    """Functional assignment: worker i belongs to the function of its team (or round-robin) and reads
    that function's source; a share of workers also cross-check one other source. Every source is
    read by at least two workers in different teams, so each fact has redundant routes."""
    src_list = list(SOURCES)
    reads = {}
    for i, w in enumerate(workers):
        home = src_list[(team_of(w) if team_of else i) % len(src_list)]
        reads[w] = {home}
        if rng.random() < cross_check:
            reads[w].add(rng.choice([x for x in src_list if x != home]))
    home = {w: src_list[(team_of(w) if team_of else i) % len(src_list)] for i, w in enumerate(workers)}
    for x in src_list:                                   # every source is cross-read by >= 2 workers of other teams
        outside = [w for w in workers if x in reads[w] and home[w] != x]
        pool = [w for w in workers if home[w] != x and x not in reads[w]]
        rng.shuffle(pool)
        while len(outside) < 2 and pool:
            w = pool.pop(); reads[w].add(x); outside.append(w)
    return reads


def orchestrator(n_workers=36, n_subs=6, seed=0, assurance="line"):
    rng = random.Random(seed); G = nx.DiGraph(family="orchestrator"); _source_nodes(G)
    G.add_node("dispatcher", kind="agent", role="dispatcher (orchestrator)", team="control")
    subs = [f"sub_{i}" for i in range(n_subs)]
    workers = [f"w_{i}" for i in range(n_workers)]
    aggs = [f"agg_{i}" for i in range(max(2, n_subs // 2))]
    for s in subs:
        G.add_node(s, kind="agent", role="sub-orchestrator", team=s); G.add_edge("dispatcher", s)
    names = {f"sub_{i}": list(SOURCES)[i % len(SOURCES)].split("_")[0] + " team" for i in range(n_subs)}
    for s_ in subs:
        G.nodes[s_]["team"] = names[s_]
    team_index = {w: i % n_subs for i, w in enumerate(workers)}
    reads = _assign_sources(workers, rng, team_of=team_index.get)
    for i, w in enumerate(workers):
        sub = subs[i % n_subs]
        src = sorted(reads[w])
        home = list(SOURCES)[team_index[w] % len(SOURCES)]
        G.add_node(w, kind="agent", role=WORKER_ROLES[home], team=names[sub], reads=src)
        G.add_edge(sub, w)
        for s in src:
            G.add_edge(s, w)
    for a in aggs:
        G.add_node(a, kind="agent", role="result aggregator", team="control")
    # sub-orchestrators come in two roles: dispatch (sub_i) and collect (col_i), so the graph stays acyclic
    for i in range(n_subs):
        c = f"col_{i}"; G.add_node(c, kind="agent", role="sub-orchestrator (results)", team=names[subs[i]])
        G.add_edge(subs[i], c)
        G.add_edge(c, aggs[i % len(aggs)])
        if rng.random() < 0.5:
            G.add_edge(c, aggs[(i + 1) % len(aggs)])
    src_list = list(SOURCES)
    for i, w in enumerate(workers):
        G.add_edge(w, f"col_{team_index[w]}")             # report to own team's collector
        for x in reads[w]:                                # consultation: cross-checks go to the owning team
            own = src_list.index(x) % n_subs
            if own != team_index[w]:
                G.add_edge(w, f"col_{own}")
        if rng.random() < 0.15:                           # occasional skip-level report
            G.add_edge(w, aggs[i % len(aggs)])
    for i, w in enumerate(workers):                       # peer links (lower index -> higher index keeps a DAG)
        if rng.random() < 0.3:
            j = rng.randrange(i + 1, n_workers) if i + 1 < n_workers else None
            if j is not None:
                G.add_edge(w, workers[j])
    G.add_node("qa", kind="agent", role="quality-assurance reviewer", team="control")
    for w in rng.sample(workers, 8):
        G.add_edge(w, "qa")
    G.add_node("decider", kind="decider", role="payment approver", team="control")
    heads = [f"div_{i}" for i in range(2)]
    for i, h in enumerate(heads):
        G.add_node(h, kind="agent", role="division head", team="control")
        G.add_edge(h, "decider")
    for i, a in enumerate(aggs):
        G.add_edge(a, heads[i % 2])
        if rng.random() < 0.3:
            G.add_edge(a, heads[(i + 1) % 2])
    if assurance == "none":
        G.remove_node("qa")
    else:
        G.add_edge("qa", "decider" if assurance == "decider" else aggs[0])
    return G


def teams(n_teams=6, team_size=7, seed=0, assurance="line"):
    rng = random.Random(seed); G = nx.DiGraph(family="teams"); _source_nodes(G)
    specialists = []
    for t in range(n_teams):
        G.add_node(f"lead_{t}", kind="agent", role="team lead", team=f"team_{t}")
        for k in range(team_size):
            specialists.append((t, f"t{t}_s{k}"))
    tmap = {s: t for t, s in specialists}
    reads = _assign_sources([s for _, s in specialists], rng, team_of=tmap.get)
    fn = {t: list(SOURCES)[t % len(SOURCES)] for t in range(n_teams)}
    for t in range(n_teams):
        G.nodes[f"lead_{t}"]["team"] = fn[t].split("_")[0] + " team"
    for t, s in specialists:
        src = sorted(reads[s])
        G.add_node(s, kind="agent", role=WORKER_ROLES[fn[t]], team=fn[t].split("_")[0] + " team", reads=src)
        for x in src:
            G.add_edge(x, s)
        G.add_edge(s, f"lead_{t}")
    for t, sp in specialists:                             # consultation: cross-checks go to the owning team's lead
        for x in G.nodes[sp]["reads"]:
            own = list(SOURCES).index(x) % n_teams
            if own != t:
                G.add_edge(sp, f"lead_{own}")
    for t in range(n_teams):                              # intra-team peer review chain
        members = [s for tt, s in specialists if tt == t]
        for a, b in zip(members, members[1:]):
            if rng.random() < 0.5:
                G.add_edge(a, b)
    for t in range(n_teams):                              # cross-team lead links (DAG by index)
        for u in range(t + 1, n_teams):
            if rng.random() < 0.3:
                G.add_edge(f"lead_{t}", f"lead_{u}")
    G.add_node("audit_lead", kind="agent", role="audit lead", team="audit")
    for k in range(3):
        a = f"auditor_{k}"; G.add_node(a, kind="agent", role="auditor", team="audit")
        for _, s in rng.sample(specialists, 4):
            G.add_edge(s, a)
        G.add_edge(a, "audit_lead")
    G.add_node("committee", kind="agent", role="approval committee chair", team="control")
    n_dept = max(2, n_teams // 3)
    for d_ in range(n_dept):
        G.add_node(f"dept_{d_}", kind="agent", role="department head", team="control")
        G.add_edge(f"dept_{d_}", "committee")
    for t in range(n_teams):
        G.add_edge(f"lead_{t}", f"dept_{t % n_dept}")
        if rng.random() < 0.25:
            G.add_edge(f"lead_{t}", f"dept_{(t + 1) % n_dept}")
    G.add_edge("audit_lead", "committee")
    G.add_node("decider", kind="decider", role="payment approver", team="control")
    G.add_edge("committee", "decider")
    if assurance == "decider":
        G.add_edge("audit_lead", "decider")
    if assurance == "none":
        G.remove_nodes_from([n for n in list(G) if G.nodes[n].get("team") == "audit"])
    return G


def pipeline(n_analysts=20, n_reviewers=8, n_approvers=3, seed=0, assurance="line"):
    rng = random.Random(seed); G = nx.DiGraph(family="pipeline"); _source_nodes(G)
    G.add_node("intake", kind="agent", role="intake coordinator", team="intake")
    analysts = [f"an_{i}" for i in range(n_analysts)]
    reads = _assign_sources(analysts, rng)
    for i, a in enumerate(analysts):
        src = sorted(reads[a]); home = list(SOURCES)[i % len(SOURCES)]
        G.add_node(a, kind="agent", role=WORKER_ROLES[home], team="analysis: " + home.split("_")[0], reads=src)
        G.add_edge("intake", a)
        for x in src:
            G.add_edge(x, a)
    coords = {}
    for x in SOURCES:
        c = f"coord_{x.split('_')[0]}"; coords[x] = c
        G.add_node(c, kind="agent", role="function coordinator", team="analysis: " + x.split("_")[0])
    for i, a in enumerate(analysts):
        home = list(SOURCES)[i % len(SOURCES)]
        G.add_edge(a, coords[home])
        for x in reads[a]:
            if x != home: G.add_edge(a, coords[x])        # cross-checks go to the owning function's coordinator
    reviewers = [f"rev_{i}" for i in range(n_reviewers)]
    cl = list(coords.values())
    for r in reviewers:
        G.add_node(r, kind="agent", role="reviewer", team="review")
        for c in rng.sample(cl, 3):
            G.add_edge(c, r)
    for i, c in enumerate(cl):
        if not any(G.has_edge(c, r) for r in reviewers):
            G.add_edge(c, reviewers[i % n_reviewers])
    for a in rng.sample(analysts, max(2, n_analysts // 6)):   # a few analysts also report straight to a reviewer
        G.add_edge(a, rng.choice(reviewers))
    seniors = [f"sr_{i}" for i in range(max(3, n_reviewers // 2))]
    for i, sr in enumerate(seniors):
        G.add_node(sr, kind="agent", role="senior reviewer", team="senior review")
        for r in rng.sample(reviewers, 3):
            G.add_edge(r, sr)
    for i, r in enumerate(reviewers):
        if not any(G.has_edge(r, sr) for sr in seniors):
            G.add_edge(r, seniors[i % len(seniors)])
    approvers = [f"appr_{i}" for i in range(n_approvers)]
    for p in approvers:
        G.add_node(p, kind="agent", role="approver", team="approval")
        for sr in rng.sample(seniors, min(3, len(seniors))):
            G.add_edge(sr, p)
    for i, sr in enumerate(seniors):                      # every senior reviewer reports to an approver
        if not any(G.has_edge(sr, p) for p in approvers):
            G.add_edge(sr, approvers[i % n_approvers])
    G.add_node("compliance", kind="agent", role="compliance desk", team="compliance",
               reads=["sanctions_list", "bank_master", "vendor_registry"], always_cross=True)
    for x in ("sanctions_list", "bank_master", "vendor_registry"):
        G.add_edge(x, "compliance")
    G.add_node("decider", kind="decider", role="payment approver", team="control")
    for p in approvers:
        G.add_edge(p, "decider")
    if assurance == "none":
        G.remove_node("compliance")
    else:
        G.add_edge("compliance", "decider" if assurance == "decider" else approvers[0])
    return G


def _sized(fn, ranges):
    def make(seed=0, **kw):
        r = random.Random(f"size-{fn.__name__}-{seed}")
        args = {k: r.randint(*v) for k, v in ranges.items()}
        args.update(kw)
        return fn(seed=seed, **args)
    make.__name__ = fn.__name__
    return make


FAMILIES = {"orchestrator": _sized(orchestrator, {"n_workers": (24, 54), "n_subs": (6, 6)}),
            "teams": _sized(teams, {"n_teams": (6, 9), "team_size": (4, 8)}),
            "pipeline": _sized(pipeline, {"n_analysts": (18, 48), "n_reviewers": (6, 14), "n_approvers": (3, 4)})}


def function_of(G, n):
    """The source function a node specialises in (from its team name), or None for generic relays."""
    team = G.nodes[n].get("team", "")
    for s in SOURCES:
        if s.split("_")[0] in team:
            return s
    return None


def stats(G):
    agents = [n for n, d in G.nodes(data=True) if d["kind"] != "source"]
    und = G.to_undirected()
    deg = sorted((G.degree(n) for n in agents), reverse=True)
    return {"family": G.graph["family"], "agents": len(agents), "sources": len(SOURCES),
            "channels": G.number_of_edges(), "is_dag": nx.is_directed_acyclic_graph(G),
            "is_tree": nx.is_tree(und), "cycles_undirected": G.number_of_edges() - G.number_of_nodes() + 1,
            "max_degree": deg[0], "median_degree": deg[len(deg) // 2],
            "routes_to_decider_median": sorted(len(list(nx.all_simple_paths(G, s, "decider", cutoff=8)))
                                               for s in SOURCES)[len(SOURCES) // 2]}
