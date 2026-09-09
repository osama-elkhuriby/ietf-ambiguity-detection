"""Bounded bidirectional candidates; retrieval edges are not alignment decisions."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from rapidfuzz.fuzz import ratio

from .config import Config
from .models import Clause, Document
from .scoring import content_tokens, score, tokens


@dataclass
class Candidate:
    source: tuple[int, ...]
    target: tuple[int, ...]
    scores: dict[str, float | None]
    exact: bool = False
    locked: bool = False

    @property
    def combined(self) -> float:
        return float(self.scores["combined"])


def joined(clauses: list[Clause], indices: tuple[int, ...]) -> Clause:
    first = clauses[indices[0]].model_copy(deep=True)
    first.normalized_text = " ".join(clauses[i].normalized_text for i in indices)
    first.reference_targets = sorted(
        {r for i in indices for r in clauses[i].reference_targets}
    )
    return first


def candidate_graph(
    a: Document,
    b: Document,
    config: Config,
    va: np.ndarray | None = None,
    vb: np.ndarray | None = None,
) -> list[Candidate]:
    ac, bc = a.clauses, b.clauses
    if not ac or not bc:
        return []
    semantic = np.maximum(0, va @ vb.T) if va is not None and vb is not None else None
    edges: set[tuple[int, int]] = set()

    def retrieve(left, right, sims):
        sections = list(dict.fromkeys(c.section_id for c in right))
        section_pos = {s: i for i, s in enumerate(sections)}
        titles = {c.section_id: c.section_heading for c in right}
        token_sets = [content_tokens(c.normalized_text) for c in right]
        local_cache = {}
        inverted = defaultdict(set)
        for j, c in enumerate(right):
            for token in token_sets[j]:
                inverted[token].add(j)
        exact = defaultdict(list)
        for j, c in enumerate(right):
            exact[c.normalized_text].append(j)
        for i, c in enumerate(left):
            if c.section_heading not in local_cache:
                best_section = max(
                    sections,
                    key=lambda s: (
                        ratio(c.section_heading.casefold(), titles[s].casefold()),
                        -section_pos[s],
                    ),
                )
                local_cache[c.section_heading] = [
                    j
                    for j, d in enumerate(right)
                    if abs(section_pos[d.section_id] - section_pos[best_section])
                    <= config.section_window
                ]
            local = local_cache[c.section_heading]
            overlap = Counter(
                j for t in content_tokens(c.normalized_text) for j in inverted[t]
            )
            ranking = lambda j, counts=overlap: (
                -counts[j] / max(1, len(token_sets[j])),
                j,
            )
            # Lexical union retains matches whose changed numbers/modals fool semantic ranking.
            if sims is not None:
                local_ids = np.array(local, dtype=int)
                chosen = local_ids[
                    np.argsort(-sims[i, local_ids], kind="stable")[: config.local_top_k]
                ].tolist()
                chosen += np.argsort(-sims[i], kind="stable")[
                    : config.global_top_k
                ].tolist()
            else:
                chosen = sorted(local, key=ranking)[: config.local_top_k]
                chosen += sorted(overlap, key=ranking)[: config.global_top_k]
            # Counter insertion order depends on hashed token-set iteration.
            # Resolve equal overlaps by source position for cross-process stability.
            chosen += sorted(overlap, key=lambda j, counts=overlap: (-counts[j], j))[
                : config.global_top_k
            ]
            chosen += exact[c.normalized_text]
            for j in set(chosen):
                yield i, j

    edges.update(retrieve(ac, bc, semantic))
    edges.update(
        (j, i)
        for i, j in retrieve(bc, ac, semantic.T if semantic is not None else None)
    )
    freq_a = Counter(c.normalized_text for c in ac)
    freq_b = Counter(c.normalized_text for c in bc)
    result = []
    for i, j in sorted(edges):
        x, y = ac[i], bc[j]
        scores = score(
            x,
            y,
            1 - abs(i / max(1, len(ac) - 1) - j / max(1, len(bc) - 1)),
            float(semantic[i, j]) if semantic is not None else None,
            config,
        )
        exact = x.normalized_text == y.normalized_text
        if scores["block_compatibility"] == 0 or (
            scores["combined"] < config.candidate_floor and not exact
        ):
            continue
        locked = (
            exact
            and freq_a[x.normalized_text] == freq_b[y.normalized_text] == 1
            and len(content_tokens(x.normalized_text)) >= 6
            and scores["structural"] == 1
        )
        if exact:
            scores["combined"] = 1.0
        result.append(Candidate((i,), (j,), scores, exact, locked))
    locked_a = {e.source[0] for e in result if e.locked}
    locked_b = {e.target[0] for e in result if e.locked}
    seen = set()
    groups = []
    for edge in result:
        i, j = edge.source[0], edge.target[0]
        if i in locked_a or j in locked_b:
            continue
        for split in (True, False):
            many = bc if split else ac
            anchor = j if split else i
            for size in range(2, config.max_group + 1):
                for start in range(
                    max(0, anchor - size + 1), min(anchor + 1, len(many) - size + 1)
                ):
                    ids = tuple(range(start, start + size))
                    source, target = ((i,), ids) if split else (ids, (j,))
                    if (
                        (source, target) in seen
                        or set(source) & locked_a
                        or set(target) & locked_b
                    ):
                        continue
                    seen.add((source, target))
                    if len({many[k].section_id for k in ids}) != 1 or any(
                        many[k].block_type
                        not in {"paragraph", "list_item", "definition"}
                        for k in ids
                    ):
                        continue
                    left, right = joined(ac, source), joined(bc, target)
                    one_tokens = content_tokens(
                        left.normalized_text if split else right.normalized_text
                    )
                    group_tokens = content_tokens(
                        right.normalized_text if split else left.normalized_text
                    )
                    if (
                        min(
                            len(one_tokens & group_tokens) / max(1, len(one_tokens)),
                            len(one_tokens & group_tokens) / max(1, len(group_tokens)),
                        )
                        < config.group_coverage
                    ):
                        continue
                    member_sets = [content_tokens(many[k].normalized_text) for k in ids]
                    if any(
                        len(s) < config.group_min_tokens
                        or len(s & one_tokens) / len(s) < config.member_coverage
                        for s in member_sets
                    ):
                        continue
                    if any(
                        not (
                            s
                            - set().union(
                                *(t for n, t in enumerate(member_sets) if n != z)
                            )
                        )
                        for z, s in enumerate(member_sets)
                    ):
                        continue
                    sem = None
                    if va is not None and vb is not None:
                        u, v = (
                            va[list(source)].sum(axis=0),
                            vb[list(target)].sum(axis=0),
                        )
                        sem = max(
                            0.0,
                            float(
                                u
                                @ v
                                / max(1e-12, np.linalg.norm(u) * np.linalg.norm(v))
                            ),
                        )
                    scores = score(left, right, edge.scores["position"], sem, config)
                    exact_group = tokens(left.normalized_text) == tokens(
                        right.normalized_text
                    )
                    if exact_group:
                        scores["combined"] = 1.0
                    if scores["combined"] >= config.match_floor:
                        groups.append(Candidate(source, target, scores, exact_group))
    return result + groups
