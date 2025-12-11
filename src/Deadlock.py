from typing import List, Optional
import numpy as np
import pulp
import itertools
from pyeda.inter import BinaryDecisionDiagram
from .PetriNet import PetriNet


def deadlock_reachable_marking(
    pn: PetriNet,
    reach_bdd: BinaryDecisionDiagram,
) -> Optional[List[int]]:
    """
    Deadlock detection by hybrid BDD + ILP, without brute-force search.

    - BDD (Task 3) gives the set of reachable markings in 1-safe PN.
    - Expand BDD assignments into full 0/1 markings.
    - Precompute enabled[t][k]:
         enabled[t][k] = 1 if transition t is enabled at marking M^(k)
                         AND its successor marking is also reachable.
    - ILP:
        Variables: y_k ∈ {0,1} selecting exactly one reachable marking.
        Constraints:  sum(y_k) = 1
                      For each t: Σ_k enabled[t][k] * y_k = 0
        → select a marking where NO transition is enabled → deadlock.
    """

    # If BDD is empty → no reachable states → no deadlock.
    if reach_bdd.is_zero():
        return None

    I = np.array(pn.I, dtype=int)   # (T,P)
    O = np.array(pn.O, dtype=int)
    place_ids = pn.place_ids        # list of place names

    n_trans, n_places = I.shape

    # ---------------------------------------------------------
    # 1) Enumerate all reachable markings from the BDD
    # ---------------------------------------------------------
    support_vars = list(reach_bdd.support)
    support_names = set(v.name for v in support_vars)

    missing_place_ids = [pid for pid in place_ids if pid not in support_names]

    reachable_set = set()
    reachable_markings: List[List[int]] = []

    for assg in reach_bdd.satisfy_all():
        base = {v.name: int(val) for v, val in assg.items()}

        # No don't-care variables
        if not missing_place_ids:
            full = [base.get(pid, 0) for pid in place_ids]
            tup = tuple(full)
            if tup not in reachable_set:
                reachable_set.add(tup)
                reachable_markings.append(full)
        else:
            # Expand all missing vars
            for bits in itertools.product([0, 1], repeat=len(missing_place_ids)):
                full_map = dict(base)
                full_map.update(dict(zip(missing_place_ids, bits)))
                full = [full_map.get(pid, 0) for pid in place_ids]
                tup = tuple(full)
                if tup not in reachable_set:
                    reachable_set.add(tup)
                    reachable_markings.append(full)

    K = len(reachable_markings)
    if K == 0:
        return None

    R = np.array(reachable_markings, dtype=int)   # (K,P)

    # ---------------------------------------------------------
    # 2) Precompute enabled[t][k]
    # ---------------------------------------------------------
    enabled = np.zeros((n_trans, K), dtype=int)

    for t in range(n_trans):
        need = I[t]                            # (P,)
        ok_input = (R >= need).all(axis=1)     # enough tokens

        R_next = R - I[t] + O[t]               # fired result

        ok_reachable = np.array([
            tuple(R_next[k]) in reachable_set
            for k in range(K)
        ])

        enabled[t] = (ok_input & ok_reachable).astype(int)

    # ---------------------------------------------------------
    # 3) ILP: Select exactly one deadlock marking
    # ---------------------------------------------------------
    prob = pulp.LpProblem("Deadlock_Detection_On_Reachable", pulp.LpMinimize)

    # Decision variables: choose marking k
    y = [pulp.LpVariable(f"y_{k}", lowBound=0, upBound=1, cat="Binary")
         for k in range(K)]

    # Must select exactly one reachable marking
    prob += pulp.lpSum(y[k] for k in range(K)) == 1

    # Deadlock constraints: no transition is enabled
    for t in range(n_trans):
        prob += pulp.lpSum(enabled[t][k] * y[k] for k in range(K)) == 0

    # Solve
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        return None

    # ---------------------------------------------------------
    # 4) Extract chosen marking
    # ---------------------------------------------------------
    for k in range(K):
        v = y[k].value()
        if v is not None and v > 0.5:
            return [int(x) for x in reachable_markings[k]]

    return None
