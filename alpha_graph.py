"""ALPHA graph information for the PRIMAL3 backbone.

Re-implements, with identical outputs, the graph inputs that the original ALPHA
environment built (see legacy_alpha/mapf_gym.py):

  * static topology graph  (MAPFEnv.get_all_dist_map + observe/get_specific_nodes)
      nodes    = end/branch points of the medial-axis skeleton of the free space
                 (legacy_alpha/map_generator.py::get_map_nodes)
                 + the ego agent's position + the ego agent's goal
      features = [dx, dy, d(node, agent) - |node - agent|_1,
                          d(node, goal)  - |node - goal|_1,
                          d(node, agent) + d(node, goal) - d(agent, goal)]
                 d = 4-connected BFS distance, (dx, dy) = node - agent position;
                 agent row = [0, 0, 0, d(a,g) - |a-g|_1, 0],
                 goal row  = [g - a, d(a,g) - |a-g|_1, 0, 0].
      If there are more than NUM_NODES nodes, skeleton rows are dropped uniformly at
      random (python `random`, same call sequence as ALPHA) until NUM_NODES remain;
      the ego-agent row index is returned as `node_index`.

  * dynamic intention graph (MAPFEnv.get_intention)
      one node per agent: [x, y, mean_x, mean_y, var_x, var_y, fx, fy, |f|]
      over the first INTENT_STEPS cells of the agent's A* path (padded with the goal),
      f = (INTENT_STEPS-th cell - position), (fx, fy) normalised by |f|.

Implementation differences (outputs unchanged, see README):
  * skeleton-node extraction is vectorised / set-based instead of O(cells * nodes);
  * BFS distance maps use scipy's C BFS instead of a python list-queue;
  * d(agent, goal) is read from PRIMAL3's per-agent goal-BFS map (same BFS);
  * node features are computed for all agents at once;
  * A* results are cached per episode on (start, goal) (the map is static).
"""
import random
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from skimage import morphology

from alg_parameters import EnvParameters, NetParameters
from astar_4 import astar_4

UNREACHABLE = 999999  # ALPHA's distance for unreachable cells (shortest.cpp THRESH)


def get_map_nodes(world, rng=None):
    """ALPHA's skeleton key points: identical output (values and order) to
    legacy_alpha/map_generator.py::get_map_nodes. world: -1 obstacle, 0 free.
    rng seeds medial_axis' random tie-breaking (legacy left it unseeded)."""
    world_for_ske = 1 - (-1 * world)
    skeleton = morphology.medial_axis(world_for_ske, rng=rng)
    sk = skeleton.astype(np.int64)
    H, W = sk.shape

    # number of skeleton cells in the in-bounds 8-neighbourhood (legacy `neighbour`)
    padded = np.pad(sk, 1)
    count = sum(padded[1 + dx:1 + dx + H, 1 + dy:1 + dy + W]
                for dx in (-1, 0, 1) for dy in (-1, 0, 1)) - 1
    is_eb = (sk == 1) & ((count == 1) | (count >= 3))
    candidates = [(int(i), int(j)) for i, j in np.argwhere(is_eb)]  # row-major, as the legacy scan

    # legacy `mask_ebpoints`: for each surviving point in row-major order, drop L-shaped
    # neighbour pairs. The legacy per-border branches are exactly these four checks
    # restricted to in-bounds cells, and out-of-bounds cells are never points.
    alive = set(candidates)
    for i, j in candidates:
        if (i, j) not in alive:
            continue
        for a, b in (((i, j + 1), (i + 1, j)), ((i, j - 1), (i + 1, j)),
                     ((i - 1, j), (i, j + 1)), ((i - 1, j), (i, j - 1))):
            if a in alive and b in alive:
                alive.discard(a)
                alive.discard(b)
    return [[i, j] for i, j in candidates if (i, j) in alive]


def bfs_distance_maps(world, sources):
    """4-connected BFS distance from each source to every cell: int64 [len(sources), H, W];
    unreachable cells and obstacles = UNREACHABLE (legacy get_all_dist_map)."""
    H, W = world.shape
    free = (world != -1)
    idx = np.arange(H * W).reshape(H, W)
    right = free[:, :-1] & free[:, 1:]
    down = free[:-1, :] & free[1:, :]
    rows = np.concatenate([idx[:, :-1][right], idx[:-1, :][down]])
    cols = np.concatenate([idx[:, 1:][right], idx[1:, :][down]])
    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(H * W, H * W))
    sources = np.asarray(sources, dtype=np.int64).reshape(-1, 2)
    if len(sources) == 0:
        return np.zeros((0, H, W), dtype=np.int64)
    dist = shortest_path(graph, directed=False, unweighted=True,
                         indices=sources[:, 0] * W + sources[:, 1])
    dist[~np.isfinite(dist)] = UNREACHABLE
    return dist.astype(np.int64).reshape(len(sources), H, W)


class AlphaGraph:
    """Per-episode ALPHA graph state (static part built once, dynamic part per step)."""

    def __init__(self, obstacle_map, skeleton_seed=None):
        t0 = time.perf_counter()
        self.world = obstacle_map
        if skeleton_seed is None:
            skeleton_seed = np.random.randint(2 ** 31 - 1)  # reproducible under the global seed
        self.nodes = np.array(get_map_nodes(obstacle_map, rng=skeleton_seed), dtype=np.int64).reshape(-1, 2)  # [M, 2]
        self.node_dist = bfs_distance_maps(obstacle_map, self.nodes)  # [M, H, W]
        self._intent_cache = {}
        self.build_time = time.perf_counter() - t0
        # counters for profiling
        self.num_astar_calls = 0
        self.num_astar_cache_hits = 0
        self.node_feature_time = 0.0
        self.intention_time = 0.0

    def node_features(self, positions, goals, agent_to_goal):
        """positions, goals: int [N, 2]; agent_to_goal: int [N] BFS distance agent -> own goal.
        Returns graph_nodes float32 [N, NUM_NODES, NUM_FEATURE], node_index int64 [N]."""
        t0 = time.perf_counter()
        P = np.asarray(positions, dtype=np.int64)
        G = np.asarray(goals, dtype=np.int64)
        a2g = np.asarray(agent_to_goal, dtype=np.int64)
        S = self.nodes
        N, M = len(P), len(S)

        d_agent = self.node_dist[:, P[:, 0], P[:, 1]].T  # [N, M]
        d_goal = self.node_dist[:, G[:, 0], G[:, 1]].T
        mh_agent = np.abs(S[None] - P[:, None]).sum(-1)
        mh_goal = np.abs(S[None] - G[:, None]).sum(-1)
        d_agent = np.where(mh_agent == 0, 0, d_agent)
        d_goal = np.where(mh_goal == 0, 0, d_goal)

        feats = np.zeros((N, M + 2, NetParameters.NUM_FEATURE), dtype=np.int64)
        feats[:, :M, 0:2] = S[None] - P[:, None]
        feats[:, :M, 2] = d_agent - mh_agent
        feats[:, :M, 3] = d_goal - mh_goal
        feats[:, :M, 4] = d_agent + d_goal - a2g[:, None]
        detour = a2g - np.abs(G - P).sum(-1)
        feats[:, M, 3] = detour                # ego agent row
        feats[:, M + 1, 0:2] = G - P           # ego goal row
        feats[:, M + 1, 2] = detour

        graph_nodes = np.zeros((N, NetParameters.NUM_NODES, NetParameters.NUM_FEATURE), dtype=np.float32)
        node_index = np.zeros(N, dtype=np.int64)
        for i in range(N):
            # legacy: `while len(nodes) > NUM_NODES: nodes.remove(nodes[randrange(len(nodes) - 2)])`
            # (rows are pairwise distinct among the removable ones, so remove == delete-at-index)
            keep = list(range(M + 2))
            while len(keep) > NetParameters.NUM_NODES:
                del keep[random.randrange(len(keep) - 2)]
            graph_nodes[i, :len(keep)] = feats[i, keep]
            node_index[i] = len(keep) - 2
        self.node_feature_time += time.perf_counter() - t0
        return graph_nodes, node_index

    def _intent_path(self, start, goal):
        """First INTENT_STEPS cells of ALPHA's A* path (start first), padded with the goal."""
        path, _ = astar_4(self.world, start, goal)  # [start, ..., goal], same search as ALPHA's astar_4
        self.num_astar_calls += 1
        path = list(path[:EnvParameters.INTENT_STEPS])
        while len(path) < EnvParameters.INTENT_STEPS:
            path.append(goal)
        return path

    @staticmethod
    def _intent_feature(start, path):
        mean = np.mean(path, axis=0)
        variance = np.var(path, axis=0)
        dx = path[-1][0] - start[0]
        dy = path[-1][1] - start[1]
        mag = (dx ** 2 + dy ** 2) ** .5
        if mag != 0:
            dx = dx / mag
            dy = dy / mag
        return [start[0], start[1], mean[0], mean[1], variance[0], variance[1], dx, dy, mag]

    def intentions(self, positions, goals):
        """Returns agent_intent float32 [N, NUM_INTENTION_FEATURE] (legacy get_intention)."""
        t0 = time.perf_counter()
        N = len(positions)
        out = np.zeros((N, NetParameters.NUM_INTENTION_FEATURE), dtype=np.float32)
        prev_path = []
        for i in range(N):
            start = (int(positions[i][0]), int(positions[i][1]))
            goal = (int(goals[i][0]), int(goals[i][1]))
            if start != goal:
                key = (start, goal)
                if key in self._intent_cache:
                    self.num_astar_cache_hits += 1
                    feature, path = self._intent_cache[key]
                else:
                    path = self._intent_path(start, goal)
                    feature = self._intent_feature(start, path)
                    self._intent_cache[key] = (feature, path)
                prev_path = path
            elif EnvParameters.ALPHA_LEGACY_INTENT_CARRYOVER:
                # legacy bug: an agent on its goal reuses the previous agent's path
                if prev_path:
                    feature = self._intent_feature(start, prev_path)
                else:
                    prev_path = [goal] * EnvParameters.INTENT_STEPS
                    feature = self._intent_feature(start, prev_path)
            else:
                feature = self._intent_feature(start, [goal] * EnvParameters.INTENT_STEPS)
            out[i] = feature
        self.intention_time += time.perf_counter() - t0
        return out
