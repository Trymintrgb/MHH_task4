from pyeda.inter import * 
from graphviz import Source

reachable = {
     (1, 1, 0),
     (1, 0, 0),
     (0, 1, 0),
     (0, 0, 1),
}

n_places = len(next(iter(reachable)))
vars_bdd = bddvars('p', n_places) 
print(vars_bdd)

bdd = None
for marking in reachable:
     term_bdd = vars_bdd[0] if marking[0] else ~vars_bdd[0]
     for i in range(1, n_places):
          term_bdd &= vars_bdd[i] if marking[i] else ~vars_bdd[i]

     if bdd is None:
          bdd = term_bdd
     else:
          bdd |= term_bdd 
     print("Satisfying points:", list(bdd.satisfy_all()))

print("Satisfying count:", bdd.satisfy_count())
print("Minimized =", espresso_exprs(bdd2expr(bdd)))
Source(bdd.to_dot()).render("bdd", format="png", cleanup=True)