import collections
from typing import Tuple, List, Optional
from pyeda.inter import *
from collections import deque
from .PetriNet import PetriNet
import numpy as np

def deadlock_reachable_marking(
    pn: PetriNet, 
    bdd: BinaryDecisionDiagram, 
) -> Optional[List[int]]:
    """
    Combine a lightweight ILP (state equation solver via bounded integer search)
    with BDD: iterate satisfying assignments of `bdd`, for each marking that is dead:
      1) try to solve C x = M - M0 for integer x >= 0 (bounded search),
      2) if solution x found, try to construct an actual firing sequence (backtracking)
         that uses exactly the counts in x and leads from M0 to M.
    Return the first reachable dead marking found as a list of ints, or None.
    """
    # Prepare place variable names in the same order as pn.place_ids
    place_names = pn.place_ids
    place_vars = [exprvar(name) for name in place_names]

    P = len(pn.place_ids)
    T = len(pn.trans_ids)
    # C matrix: change = O - I, shape (places, transitions)
    C = (pn.O - pn.I).astype(int)

    M0 = np.array(pn.M0, dtype=int)

    # Helper: build marking list from assignment returned by bdd.satisfy_all()
    def assign_to_marking(assign) -> Optional[np.ndarray]:
        marking = []
        for var in place_vars:
            # assign keys in pyeda may be Var objects or names
            val = assign.get(var, None)
            if val is None:
                val = assign.get(var.name, None)
            if val is None:
                # if missing variable mapping, fail (shouldn't happen)
                return None
            marking.append(int(bool(val)))
        return np.array(marking, dtype=int)

    # Check dead: no transition enabled at marking
    def is_dead(marking: np.ndarray) -> bool:
        for t_idx in range(T):
            # enabled if all input places have token
            if np.all(marking >= pn.I[:, t_idx]):
                return False
        return True

    # Solve C x = rhs (rhs = M - M0) for integer x >=0 using bounded backtracking
    def solve_state_equation(rhs: np.ndarray) -> Optional[List[int]]:
        # Quick feasibility: rhs must be integer vector
        rhs = rhs.astype(int)

        # Choose an upper bound for each transition's firing count.
        # Heuristic bound: max_fire = 1 + sum(abs(rhs)) + len(P)
        # (keeps search limited; can be tuned)
        max_fire = int(1 + np.sum(np.abs(rhs)) + P)
        # But also avoid excessive bound for large nets
        max_fire = min(max_fire, 8 + P)  # cap to reasonable number

        # Precompute for pruning: for remaining transitions, the min and max contributions
        C_int = C.astype(int)  # shape (P, T)

        # DFS with pruning
        solution = [0] * T
        memo_prune = {}

        def dfs(k: int, partial: np.ndarray) -> bool:
            # partial = sum_{i<k} C[:,i] * solution[i]
            if k == T:
                # check equality
                return np.array_equal(partial, rhs)
            key = (k, tuple(partial.tolist()))
            if key in memo_prune:
                return False
            # compute residual we need from remaining variables
            residual = rhs - partial  # shape (P,)
            # compute possible min/max from remaining transitions with bounds [0, max_fire]
            # min_possible = sum_j min(0, C[:,j])*max_fire, max_possible = sum_j max(0,C[:,j])*max_fire
            rem_min = np.zeros(P, dtype=int)
            rem_max = np.zeros(P, dtype=int)
            for j in range(k, T):
                col = C_int[:, j]
                # positive contributions
                pos = np.maximum(0, col)
                neg = np.minimum(0, col)
                rem_max += pos * max_fire
                rem_min += neg * max_fire
            # prune: residual must be within [rem_min, rem_max]
            if np.any(residual < rem_min) or np.any(residual > rem_max):
                memo_prune[key] = False
                return False
            # try possible counts for variable k
            # We can compute tighter bounds for variable k by solving per-place constraints:
            # For each place i: col_i * x_k must be between (residual_i - rem_{others}_max) and (residual_i - rem_{others}_min)
            # For simplicity, use global 0..max_fire
            for val in range(0, max_fire + 1):
                solution[k] = val
                new_partial = partial + C_int[:, k] * val
                # small pruning: check if new_partial not exceeding rhs beyond possible remaining
                # compute remaining min/max for j>k
                rem_min2 = np.zeros(P, dtype=int)
                rem_max2 = np.zeros(P, dtype=int)
                for j in range(k+1, T):
                    col = C_int[:, j]
                    rem_max2 += np.maximum(0, col) * max_fire
                    rem_min2 += np.minimum(0, col) * max_fire
                residual2 = rhs - new_partial
                if np.any(residual2 < rem_min2) or np.any(residual2 > rem_max2):
                    continue
                if dfs(k+1, new_partial):
                    return True
            memo_prune[key] = False
            return False

        start_partial = np.zeros(P, dtype=int)
        ok = dfs(0, start_partial)
        if ok:
            return solution.copy()
        return None

    # Given vector x (counts per transition), try to construct an actual firing sequence
    # using exactly x counts, by backtracking trying enabled transitions.
    def construct_firing_sequence(M_target: np.ndarray, x_counts: List[int]) -> Optional[List[int]]:
        x_counts = [int(v) for v in x_counts]
        total_fires = sum(x_counts)
        # memoization set for (marking_tuple, tuple(counts_remaining))
        memo = set()

        def bt(cur_marking: Tuple[int, ...], counts_remaining: Tuple[int, ...]) -> Optional[List[int]]:
            if (cur_marking, counts_remaining) in memo:
                return None
            # if all counts done, check equality with target marking
            if sum(counts_remaining) == 0:
                if np.array_equal(np.array(cur_marking, dtype=int), M_target):
                    return []
                else:
                    memo.add((cur_marking, counts_remaining))
                    return None
            cur = np.array(cur_marking, dtype=int)
            # try every transition with remaining count > 0 and enabled
            for t_idx in range(T):
                if counts_remaining[t_idx] <= 0:
                    continue
                if np.all(cur >= pn.I[:, t_idx]):
                    nxt = cur - pn.I[:, t_idx] + pn.O[:, t_idx]
                    nxt = np.clip(nxt, 0, 1)
                    new_counts = list(counts_remaining)
                    new_counts[t_idx] -= 1
                    res = bt(tuple(nxt.tolist()), tuple(new_counts))
                    if res is not None:
                        return [t_idx] + res
            memo.add((cur_marking, counts_remaining))
            return None

        seq = bt(tuple(M0.tolist()), tuple(x_counts))
        return seq

    # Iterate satisfying assignments of provided BDD
    # pyeda's satisfy_all returns dicts mapping vars (or names) to 0/1
    try:
        sat_iter = bdd.satisfy_all()
    except Exception:
        # if bdd is not iterable or empty
        return None

    for assign in sat_iter:
        marking = assign_to_marking(assign)
        if marking is None:
            continue
        # Only consider dead markings
        if not is_dead(marking):
            continue
        # Solve state equation: C x = M - M0
        rhs = (marking - M0).astype(int)
        x = solve_state_equation(rhs)
        if x is None:
            # no integer solution under the bounds
            continue
        # attempt to build firing sequence of counts x
        seq = construct_firing_sequence(marking, x)
        if seq is not None:
            # Found reachable dead marking
            return [int(v) for v in marking.tolist()]

    return None
