from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from .core import Action, Observation


@dataclass(frozen=True, slots=True)
class Component:
    color: int
    size: int
    min_x: int
    min_y: int
    max_x: int
    max_y: int
    center_x: int
    center_y: int
    cells: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class FrameFeatures:
    background: int
    color_counts: tuple[int, ...]
    component_counts: tuple[int, ...]
    non_background: int


def background_color(frame: np.ndarray) -> int:
    counts = np.bincount(np.asarray(frame, dtype=np.int64).ravel(), minlength=16)
    return int(counts.argmax())


def connected_components(frame: np.ndarray, include_background: bool = False) -> list[Component]:
    """Extract 4-connected same-colour objects without game-specific semantics."""

    grid = np.asarray(frame)
    height, width = grid.shape
    bg = background_color(grid)
    seen = np.zeros((height, width), dtype=bool)
    out: list[Component] = []
    for y0 in range(height):
        for x0 in range(width):
            if seen[y0, x0]:
                continue
            color = int(grid[y0, x0])
            seen[y0, x0] = True
            if not include_background and color == bg:
                continue
            queue = deque([(x0, y0)])
            cells: list[tuple[int, int]] = []
            while queue:
                x, y = queue.popleft()
                cells.append((x, y))
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= nx < width and 0 <= ny < height and not seen[ny, nx] and int(grid[ny, nx]) == color:
                        seen[ny, nx] = True
                        queue.append((nx, ny))
            xs = [p[0] for p in cells]
            ys = [p[1] for p in cells]
            cx = int(round(sum(xs) / len(xs)))
            cy = int(round(sum(ys) / len(ys)))
            # The arithmetic centroid can fall in a hollow object.  Pick its
            # nearest real cell so every proposed click lands on the object.
            center = min(cells, key=lambda p: abs(p[0] - cx) + abs(p[1] - cy))
            out.append(Component(
                color=color,
                size=len(cells),
                min_x=min(xs), min_y=min(ys), max_x=max(xs), max_y=max(ys),
                center_x=center[0], center_y=center[1], cells=tuple(cells),
            ))
    return out


def features(frame: np.ndarray) -> FrameFeatures:
    grid = np.asarray(frame)
    bg = background_color(grid)
    counts = np.bincount(grid.astype(np.int64).ravel(), minlength=16)[:16]
    comp_counts = [0] * 16
    for comp in connected_components(grid):
        if 0 <= comp.color < 16:
            comp_counts[comp.color] += 1
    return FrameFeatures(
        background=bg,
        color_counts=tuple(int(x) for x in counts),
        component_counts=tuple(comp_counts),
        non_background=int(grid.size - counts[bg]),
    )


def candidate_actions(obs: Observation, max_clicks: int = 48) -> list[Action]:
    """Create a small, object-centric action set from the official observation."""

    available = tuple(dict.fromkeys(int(x) for x in obs.available_actions))
    actions = [Action(aid) for aid in available if aid not in (0, 6)]
    if 6 not in available:
        return actions

    comps = connected_components(obs.frame)
    # Small/rare objects tend to be controls, tokens, or targets.  Size and
    # colour frequency are generic priors, not game labels.
    per_color = Counter(comp.color for comp in comps)
    comps.sort(key=lambda c: (per_color[c.color], c.size, c.color, c.min_y, c.min_x))
    points: list[tuple[int, int]] = []
    for comp in comps:
        points.append((comp.center_x, comp.center_y))
        if comp.size >= 4:
            points.extend([
                (comp.min_x, comp.min_y), (comp.max_x, comp.min_y),
                (comp.min_x, comp.max_y), (comp.max_x, comp.max_y),
            ])
        if comp.size > 16:
            points.extend((x, y) for x, y in comp.cells[::max(1, comp.size // 4)][:4])

    seen: set[tuple[int, int]] = set()
    for x, y in points:
        p = (max(0, min(63, int(x))), max(0, min(63, int(y))))
        if p in seen:
            continue
        seen.add(p)
        actions.append(Action(6, *p))
        if len(seen) >= max_clicks:
            break
    # Empty-looking click games still get a coarse information-seeking grid.
    if not seen:
        height, width = obs.frame.shape
        for y in range(4, height, 8):
            for x in range(4, width, 8):
                actions.append(Action(6, x, y))
                if len(actions) >= max_clicks:
                    return actions
    return actions


def abstraction_key(frame: np.ndarray, stride: int = 4) -> bytes:
    """Coarse object/palette bucket used for novelty, never for exact dedup."""

    f = features(frame)
    coarse = np.ascontiguousarray(np.asarray(frame)[::stride, ::stride], dtype=np.int8)
    summary = bytes(min(255, x) for x in f.color_counts + f.component_counts)
    return summary + coarse.tobytes()


def transition_value(before: np.ndarray, after: np.ndarray) -> float:
    """Game-agnostic information value of an observed transition."""

    a = np.asarray(before)
    b = np.asarray(after)
    changed = float(np.count_nonzero(a != b)) / max(1, a.size)
    fa, fb = features(a), features(b)
    palette_delta = sum(x != y for x, y in zip(fa.color_counts, fb.color_counts)) / 16.0
    object_delta = sum(abs(x - y) for x, y in zip(fa.component_counts, fb.component_counts)) / 16.0
    return changed + 0.35 * palette_delta + 0.20 * min(1.0, object_delta)
