from typing import List, Optional
import numpy as np
import pulp
import itertools
from pyeda.inter import BinaryDecisionDiagram

# keep your earlier helpers: _normalize_matrices, _parse_bdd_assignment, _is_transition_enabled_for_marking, etc.

def deadlock_reachable_marking(
    pn,
    reach_bdd: BinaryDecisionDiagram,
    max_enumerate: int = 2000,
) -> Optional[List[int]]:
    """
    Optimized hybrid: enumerate reachable markings from BDD (expand don't-cares),
    precompute enabled[t][k], then solve small ILP over y_k selecting exactly one
    reachable marking that disables all transitions.

    If the BDD expands to more than `max_enumerate` markings, we bail out (or
    optionally fall back to a different ILP strategy).
    """
    # quick reject
    try:
        if reach_bdd.is_zero():
            return None
    except Exception:
        # if BDD doesn't provide is_zero, continue and rely on satisfy_all
        pass

    # Normalize matrices to canonical I (P x T), O (P x T), M0 (P,)
    I_mat, O_mat, M0 = _normalize_matrices(pn)  # I_mat: P x T
    P = I_mat.shape[0]
    T = I_mat.shape[1]

    # But many BDD-based codes use transitions x places orientation (T x P)
    # We'll use I_t (T x P) and O_t (T x P) for enabled checks like friend's code
    I_t = I_mat.T.copy()  # shape (T, P)
    O_t = O_mat.T.copy()  # shape (T, P)

    place_ids = pn.place_ids  # names in BDD are expected to match these (strings)

    # ---------------------------------------------------------
    # 1) Enumerate reachable markings from BDD (expand don't-cares)
    # ---------------------------------------------------------
    support_vars = list(reach_bdd.support) if hasattr(reach_bdd, "support") else []
    name_to_var = {v.name: v for v in support_vars}
    support_names = set(name_to_var.keys())

    # places missing from the BDD support (don't-care places)
    missing_place_ids = [pid for pid in place_ids if pid not in support_names]

    reachable_markings_set = set()
    reachable_markings: List[List[int]] = []

    # satisfy_all might be generator of dicts mapping var->0/1
    try:
        sat_iter = reach_bdd.satisfy_all()
    except Exception:
        # If BDD doesn't implement satisfy_all, give up
        return None

    for assignment in sat_iter:
        # assignment: {var: 0/1}
        base_assign = {getattr(v, "name", str(v)): int(bool(val)) for v, val in assignment.items()}

        if not missing_place_ids:
            full_assign = base_assign
            m_vec = [full_assign.get(pid, 0) for pid in place_ids]
            t = tuple(m_vec)
            if t not in reachable_markings_set:
                reachable_markings_set.add(t)
                reachable_markings.append(m_vec)
        else:
            # expand don't-care places
            for bits in itertools.product([0, 1], repeat=len(missing_place_ids)):
                extra = dict(zip(missing_place_ids, bits))
                full_assign = dict(base_assign)
                full_assign.update(extra)
                m_vec = [full_assign.get(pid, 0) for pid in place_ids]
                t = tuple(m_vec)
                if t not in reachable_markings_set:
                    reachable_markings_set.add(t)
                    reachable_markings.append(m_vec)

        # safety cutoff while enumerating
        if len(reachable_markings) > max_enumerate:
            # too many markings to enumerate; fallback or bail out
            # For now, bail out (you can instead call the earlier ILP-on-x fallback).
            return None

    K = len(reachable_markings)
    if K == 0:
        return None

    reachable_arr = np.array(reachable_markings, dtype=int)  # K x P

    # ---------------------------------------------------------
    # 2) Precompute enabled[t][k]
    # ---------------------------------------------------------
    n_trans = T
    n_places = P
    enabled = np.zeros((n_trans, K), dtype=int)

    # For each marking k and transition t:
    #  - enough tokens: reachable_arr[k] >= I_t[t]
    #  - next marking M' = M - I_t[t] + O_t[t] is in reachable set
    for t in range(n_trans):
        need = I_t[t]  # (P,)
        ok1 = (reachable_arr >= need).all(axis=1)  # shape (K,)

        M_next = reachable_arr - need + O_t[t]  # (K,P)
        ok2 = np.array([tuple(M_next[k].tolist()) in reachable_markings_set for k in range(K)])

        enabled[t] = (ok1 & ok2).astype(int)

    # If there exists a marking k such that for every t enabled[t,k] == 0, we already
    # have a dead reachable marking — ILP will find it quickly, but we can quick-check:
    for k in range(K):
        if enabled[:, k].sum() == 0:
            return [int(x) for x in reachable_markings[k]]

    # ---------------------------------------------------------
    # 3) ILP: pick exactly one reachable marking y_k (binary) such that
    #    for all t: sum_k enabled[t,k] * y_k == 0
    # ---------------------------------------------------------
    prob = pulp.LpProblem("Deadlock_On_Reachable", pulp.LpMinimize)
    y = [pulp.LpVariable(f"y_{k}", lowBound=0, upBound=1, cat="Binary") for k in range(K)]

    # select exactly one marking
    prob += pulp.lpSum(y[k] for k in range(K)) == 1

    # prevent any transition from being enabled at the chosen marking
    for t in range(n_trans):
        # sum(enabled[t][k] * y_k) == 0
        if enabled[t].sum() > 0:
            prob += pulp.lpSum(enabled[t][k] * y[k] for k in range(K)) == 0
        # if enabled[t].sum() == 0 then constraint is redundant (already covered above by quick-check)

    # objective: dummy (minimize number of picked markings — always 1). CBC needs objective.
    prob += 0

    # Solve with small time limit and silent mode
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=10)
    status = prob.solve(solver)

    if pulp.LpStatus[status] not in ("Optimal", "Feasible"):
        return None

    # extract chosen marking
    chosen_k = None
    for k in range(K):
        val = y[k].value()
        if val is not None and val > 0.5:
            chosen_k = k
            break

    if chosen_k is None:
        return None

    # final sanity: return marking as list[int]
    return [int(x) for x in reachable_markings[chosen_k]]
