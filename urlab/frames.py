"""FrameGraph -- the tf2 replacement.

A tiny transform tree: register edges, look up a chain. It differs from tf2 in one respect that
matters, and it is the reason three separate bugs in the ROS stack cannot recur here.

    tf2 buffers a TIME HISTORY of every transform and drops entries older than its cache window
    (10 s by default). A perception node that republishes SLOWLY -- a SAM3 connector estimate
    arriving every 6-12 s -- therefore has its transform EXPIRE between publishes, and the lookup
    fails even though the frame is being published perfectly well and the estimate is perfectly
    good. The ROS stack hit this three times and each time the fix was to widen a cache
    (tf_cache_s), which treats the symptom.

    FrameGraph keeps only the LATEST transform per edge, and keeps it forever. Nothing expires.
    Instead, staleness is something the CALLER asks about:

        T = frames.lookup('base_link', 'connector', max_age=30.0)   # None if too old

    The default is max_age=None, meaning "I don't care how old this is" -- correct for static
    edges (hand-eye) and for edges backed by a live source (base_link->tool0 reads RTDE on every
    call, so it is never stale). For an observation you must state your tolerance, and the
    ANSWER to "is this too old?" is then a property of the observation rather than an accident of
    a buffer size.

Not a general tf2: no interpolation, no time-travel lookups, no multi-publisher arbitration. It
does not need to be -- the chain here is short and known:

    base_link --(live, RTDE)--> tool0 --(static, hand-eye)--> camera --(observed)--> marker_N
                                                                                 \\-> connector
"""

import threading
import time

import numpy as np

from .transforms import inverse


class FrameGraph:
    """Undirected transform graph. Thread-safe; a perception thread can publish while the main
    thread looks up."""

    def __init__(self):
        self._edges = {}        # (parent, child) -> {'T', 'stamp', 'source'}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ writing
    def set_static(self, parent, child, T):
        """A transform that never changes and never goes stale (e.g. hand-eye tool0->camera)."""
        with self._lock:
            self._edges[(parent, child)] = {'T': np.array(T, dtype=float),
                                            'stamp': None, 'source': None}

    def set_live(self, parent, child, fn):
        """A transform read FRESH on every lookup (e.g. base_link->tool0 from RTDE).

        Never stale by construction -- there is no sampled value to go out of date. `fn` returns
        a 4x4 and is called inside the lookup, so it must be cheap and must not block."""
        with self._lock:
            self._edges[(parent, child)] = {'T': None, 'stamp': None, 'source': fn}

    def set_observed(self, parent, child, T, stamp=None):
        """A measurement, stamped so callers can reject it as stale.

        `stamp` should be the time the OBSERVATION was made (for a camera, the capture time),
        NOT the time the estimate was finished. With SAM3 taking 1-2 s per frame those differ by
        more than most staleness budgets, and using the wrong one makes a fresh detection look
        expired -- or worse, an expired one look fresh."""
        with self._lock:
            self._edges[(parent, child)] = {'T': np.array(T, dtype=float),
                                            'stamp': time.monotonic() if stamp is None else stamp,
                                            'source': None}

    def drop(self, parent, child):
        with self._lock:
            self._edges.pop((parent, child), None)

    # ------------------------------------------------------------------ reading
    def age(self, parent, child):
        """Seconds since the edge was last set. 0.0 for static/live edges, None if absent."""
        with self._lock:
            for key in ((parent, child), (child, parent)):
                e = self._edges.get(key)
                if e is not None:
                    return 0.0 if e['stamp'] is None else time.monotonic() - e['stamp']
        return None

    def lookup(self, target, source, max_age=None):
        """T_target_source as a 4x4, or None if there is no path -- or if any edge on the path is
        older than `max_age` seconds.

        Applying max_age to EVERY edge on the path (not just the endpoint) is deliberate: a chain
        is exactly as fresh as its stalest link, and a fresh camera->marker composed onto a
        minutes-old base->camera is a stale answer wearing a fresh timestamp."""
        with self._lock:
            path = self._path(target, source)
            if path is None:
                return None

            T = np.eye(4)
            for parent, child in path:
                e = self._edges.get((parent, child))
                if e is not None:                       # forward edge
                    step = e['source']() if e['source'] is not None else e['T']
                else:                                   # reverse edge
                    e = self._edges[(child, parent)]
                    src = e['source']() if e['source'] is not None else e['T']
                    step = inverse(src)
                if step is None:
                    return None
                if max_age is not None and e['stamp'] is not None:
                    if time.monotonic() - e['stamp'] > max_age:
                        return None
                T = T @ step
            return T

    def wait_for(self, target, source, timeout, max_age=None, poll=0.05):
        """Block until the transform is available (and fresh enough), or the timeout expires."""
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            T = self.lookup(target, source, max_age=max_age)
            if T is not None:
                return T
            time.sleep(poll)
        return None

    def frames(self):
        with self._lock:
            return sorted({f for edge in self._edges for f in edge})

    # ------------------------------------------------------------------ internals
    def _path(self, target, source):
        """BFS for a list of (parent, child) edges walking target -> source. Edges are traversable
        in both directions; the caller inverts the reversed ones."""
        if target == source:
            return []

        adjacent = {}
        for parent, child in self._edges:
            adjacent.setdefault(parent, set()).add(child)
            adjacent.setdefault(child, set()).add(parent)
        if target not in adjacent or source not in adjacent:
            return None

        prev, queue, seen = {}, [target], {target}
        while queue:
            node = queue.pop(0)
            if node == source:
                chain, cur = [], source
                while cur != target:
                    chain.append((prev[cur], cur))
                    cur = prev[cur]
                return list(reversed(chain))
            for nxt in adjacent.get(node, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    prev[nxt] = node
                    queue.append(nxt)
        return None

    def describe(self):
        """Human-readable dump -- the equivalent of `ros2 run tf2_tools view_frames`."""
        with self._lock:
            lines = []
            for (parent, child), e in sorted(self._edges.items()):
                if e['source'] is not None:
                    kind = 'live'
                elif e['stamp'] is None:
                    kind = 'static'
                else:
                    kind = f"observed {time.monotonic() - e['stamp']:.1f}s ago"
                lines.append(f'  {parent} -> {child}  [{kind}]')
            return '\n'.join(lines) if lines else '  (no frames)'
