"""Global one-to-one assignment and bounded competing split/merge optimization."""

from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import lil_matrix

from .candidates import Candidate
from .config import Config


def select(candidates: list[Candidate], config: Config) -> tuple[list[Candidate], dict]:
    locked = [e for e in candidates if e.locked]
    used_a = {i for e in locked for i in e.source}
    used_b = {j for e in locked for j in e.target}
    graph = nx.Graph()
    viable = []
    for e in candidates:
        if (
            e.locked
            or set(e.source) & used_a
            or set(e.target) & used_b
            or e.combined < config.match_floor
        ):
            continue
        nodes = [(0, i) for i in e.source] + [(1, j) for j in e.target]
        for node in nodes[1:]:
            graph.add_edge(nodes[0], node)
        viable.append(e)
    components = sorted(nx.connected_components(graph), key=lambda c: min(c))
    membership = {
        node: k for k, component in enumerate(components) for node in component
    }
    groups = defaultdict(list)
    for edge in viable:
        groups[membership[(0, edge.source[0])]].append(edge)
    selected = list(locked)
    diagnostics = {
        "locked": len(locked),
        "bipartite_components": 0,
        "complex_components": 0,
        "solver": "scipy assignment / HiGHS MILP",
        "optimal": True,
    }
    for k, nodes in enumerate(components):
        edges = groups[k]
        gains = np.array(
            [
                (len(e.source) + len(e.target)) / 2 * (e.combined - config.match_floor)
                - config.group_penalty * (len(e.source) + len(e.target) - 2)
                for e in edges
            ]
        )
        # Quantize and use a bounded secondary preference for stable input-order ties.
        gains = np.round(gains, 6) + np.arange(len(edges), 0, -1) * (
            1e-9 / max(1, len(edges))
        )
        if all(len(e.source) == len(e.target) == 1 for e in edges):
            diagnostics["bipartite_components"] += 1
            left = sorted(i for side, i in nodes if side == 0)
            right = sorted(i for side, i in nodes if side == 1)
            li, ri = (
                {v: i for i, v in enumerate(left)},
                {v: i for i, v in enumerate(right)},
            )
            costs = np.full((len(left), len(right) + len(left)), 1e6)
            costs[:, len(right) :] = 0
            lookup = {}
            for e, g in zip(edges, gains):
                i, j = li[e.source[0]], ri[e.target[0]]
                costs[i, j] = -g
                lookup[i, j] = e
            rows, columns = linear_sum_assignment(costs)
            selected.extend(
                lookup[i, j]
                for i, j in zip(rows, columns)
                if (i, j) in lookup and costs[i, j] < 0
            )
        else:
            diagnostics["complex_components"] += 1
            node_index = {node: i for i, node in enumerate(sorted(nodes))}
            incidence = lil_matrix((len(nodes), len(edges)), dtype=float)
            for j, e in enumerate(edges):
                for node in [(0, i) for i in e.source] + [(1, i) for i in e.target]:
                    incidence[node_index[node], j] = 1
            solution = milp(
                -gains,
                integrality=np.ones(len(edges)),
                bounds=Bounds(0, 1),
                constraints=LinearConstraint(incidence.tocsc(), 0, 1),
                options={"time_limit": config.solver_seconds, "mip_rel_gap": 0.0},
            )
            if not solution.success:
                raise RuntimeError(
                    f"Complex component {k} not optimally solved: {solution.message}; increase solver_seconds"
                )
            selected.extend(
                e for e, x, g in zip(edges, solution.x, gains) if x > 0.5 and g > 0
            )
    return sorted(selected, key=lambda e: (e.source, e.target)), diagnostics
