"""A monitoring assignment: which edges are observed and how they are grouped.

Constraints (configurable; see research/research_proposal.md section 4):
- ``max_edges``: number of monitored channels (MESA's budget unit).
- ``max_group_size``: messages a single monitor can read jointly (context bound, in edges).
- ``max_groups``: number of monitors.
- overlap: by default groups are disjoint (each edge inspected once); ``allow_overlap``
  permits an edge in several groups, which costs its messages once per group.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

from jmon.messages import Edge


@dataclass
class Assignment:
    groups: list[list[Edge]]
    policy: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def edges(self) -> list[Edge]:
        out: list[Edge] = []
        for g in self.groups:
            for e in g:
                if e not in out:
                    out.append(e)
        return out

    def to_dict(self) -> dict:
        return {"policy": self.policy, "groups": [[list(e) for e in g] for g in self.groups], "meta": self.meta}


def validate_assignment(a: Assignment, all_edges: list[Edge], max_edges: int | None = None,
                        max_group_size: int | None = None, max_groups: int | None = None,
                        allow_overlap: bool = False) -> None:
    es = set(all_edges)
    seen: set[Edge] = set()
    for g in a.groups:
        if not g:
            raise ValueError("empty group")
        if max_group_size is not None and len(g) > max_group_size:
            raise ValueError(f"group {g} exceeds max_group_size={max_group_size}")
        for e in g:
            if e not in es:
                raise ValueError(f"edge {e} not in graph")
            if e in seen and not allow_overlap:
                raise ValueError(f"edge {e} assigned twice")
            seen.add(e)
    if max_edges is not None and len(a.edges) > max_edges:
        raise ValueError(f"{len(a.edges)} edges exceed max_edges={max_edges}")
    if max_groups is not None and len(a.groups) > max_groups:
        raise ValueError(f"{len(a.groups)} groups exceed max_groups={max_groups}")
