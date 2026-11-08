"""Run one episode of the synthetic multi-agent workflow and log every message.

The episode is a single pass in topological order over ``rounds`` rounds. In round r,
every agent with inbound messages from round r-1 (or sources in round 0) acts. For a DAG
of depth d, d+1 rounds deliver everything to the sink.

Attacks are applied through ``AttackHook`` objects that may (a) replace an agent by a
malicious one (malicious-agent model) or (b) rewrite messages on an edge after they are
produced (channel-interception model). Masking an edge (for clean-execution probes) is a
hook that drops messages on that edge.

The result is immutable once returned: monitors operate on the log afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol
import numpy as np

from jmon.graphs import CommGraph
from jmon.messages import Message, MessageLog, Edge
from jmon.agents.sim_agent import SimAgent


class AttackHook(Protocol):
    def on_message(self, m: Message, rng: np.random.Generator) -> Message | None:
        """Return a (possibly rewritten) message or None to drop it."""
        ...


@dataclass
class Episode:
    graph: CommGraph
    log: MessageLog
    sink_output: dict
    truth: dict
    task_loss: float
    attack_success: dict[str, bool] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


class EdgeMask:
    """Drops all messages on the given edges (clean-execution ablation probe)."""

    def __init__(self, edges: list[Edge]) -> None:
        self.edges = set(edges)

    def on_message(self, m: Message, rng):
        return None if m.edge in self.edges else m


def make_world(graph: CommGraph, seed: int, n_facts: int = 6, redundancy: float = 0.5,
               benign_token_rate: float = 0.5) -> tuple[dict[str, SimAgent], dict[str, int]]:
    """Assign facts to source agents. With probability ``redundancy`` a fact is held by two
    sources, which creates path redundancy and hence pairwise edge interaction."""
    rng = np.random.default_rng([seed, 1])
    sources = [n for n in graph.nodes if graph.g.in_degree(n) == 0]
    truth = {f"k{i}": int(rng.integers(0, 100)) for i in range(n_facts)}
    holders: dict[str, dict[str, int]] = {s: {} for s in sources}
    for k, v in truth.items():
        chosen = [sources[rng.integers(len(sources))]]
        if len(sources) > 1 and rng.random() < redundancy:
            other = sources[rng.integers(len(sources))]
            if other not in chosen:
                chosen.append(other)
        for s in chosen:
            holders[s][k] = v
    agents = {}
    for i, n in enumerate(graph.nodes):
        agents[n] = SimAgent(n, holders.get(n, {}), rng=np.random.default_rng([seed, 2, i]),
                             benign_token_rate=benign_token_rate)
    return agents, truth


def run_episode(graph: CommGraph, agents: dict[str, SimAgent], truth: dict[str, int], seed: int,
                hooks: list[AttackHook] | None = None, success_fns: dict[str, Callable[["Episode"], bool]] | None = None,
                rounds: int | None = None) -> Episode:
    hooks = hooks or []
    rng = np.random.default_rng([seed, 3])
    for a in agents.values():
        a.reset()
    log = MessageLog()
    counter = [0]
    t = [0.0]

    def make_msg(u, v, r, content, fields):
        counter[0] += 1
        t[0] += 1.0
        return Message(f"m{counter[0]:05d}", u, v, r, t[0], content, fields, {})

    order = graph.topological_order()
    depth = _depths(graph)
    rounds = rounds if rounds is not None else max(depth.values()) + 1
    inbox: dict[str, list[Message]] = {n: [] for n in graph.nodes}
    for r in range(rounds):
        for n in order:
            if depth[n] != r:
                continue
            msgs = agents[n].act(inbox[n], graph.out_edges(n), r, make_msg)
            for m in msgs:
                for h in hooks:
                    m = h.on_message(m, rng)
                    if m is None:
                        break
                if m is None:
                    continue
                log.append(m)
                inbox[m.recipient].append(m)
    sink = agents[graph.sink]
    sink_output = {"facts": dict(sink.known), "ctx": list(sink.ctx_seen), "act": list(sink.act_seen)}
    wrong = sum(1 for k, v in truth.items() if sink.known.get(k) != v)
    ep = Episode(graph, log, sink_output, truth, wrong / max(1, len(truth)))
    for name, fn in (success_fns or {}).items():
        ep.attack_success[name] = bool(fn(ep))
    return ep


def _depths(graph: CommGraph) -> dict[str, int]:
    d = {}
    for n in graph.topological_order():
        preds = graph.predecessors(n)
        d[n] = 0 if not preds else 1 + max(d[p] for p in preds)
    return d


class EdgeFiller:
    """MESA's masking probe: keeps the edge but replaces its messages with role-consistent,
    information-free fillers (no facts, no tokens)."""

    def __init__(self, edges: list[Edge]) -> None:
        self.edges = set(edges)

    def on_message(self, m: Message, rng):
        if m.edge not in self.edges:
            return m
        return Message(m.msg_id, m.sender, m.recipient, m.round, m.timestamp,
                       "Report: nothing new to add.", {"facts": {}}, dict(m.hidden))
