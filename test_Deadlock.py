import numpy as np
from src.Deadlock import deadlock_reachable_marking
from src.PetriNet import PetriNet
from pyeda.inter import *

def test_001():
    P = ["p1", "p2", "p3"]
    T = ["t1", "t2", "t3"]
    I = np.array([[1,0,0],
                  [0,1,0],
                  [0,0,1]])
    O = np.array([[0,1,0],
                  [0,0,1],
                  [1,0,0]])
    M0 = np.array([1,0,0])
    pn = PetriNet(P, T, P, T, I, O, M0)

    p1, p2, p3 = exprvar('p1'), exprvar('p2'), exprvar('p3')
    expected_expr = Or(And(~p1, ~p2, p3),
                    And(~p1, p2, ~p3),
                    And(p1, ~p2, ~p3))
    
    bdd = expr2bdd(expected_expr)

    marking = deadlock_reachable_marking(pn, bdd)

    assert marking == None

def test_002():
    P = ["p1", "p2", "p3"]
    T = ["t1", "t2", "t3"]
    I = np.array([[1,0,0],
                  [0,1,0],
                  [0,1,0]])
    O = np.array([[0,1,0],
                  [0,0,1],
                  [1,0,0]])
    M0 = np.array([1,0,0])
    pn = PetriNet(P, T, P, T, I, O, M0)

    p1, p2, p3 = exprvar('p1'), exprvar('p2'), exprvar('p3')
    expected_expr = Or(And(~p1, ~p2, p3),
                    And(~p1, p2, ~p3),
                    And(p1, ~p2, ~p3))
    
    bdd = expr2bdd(expected_expr)

    marking = deadlock_reachable_marking(pn, bdd)

    assert marking == [0,0,1]

def test_003():
    P = ["p1", "p2", "p3"]
    T = ["t1", "t2", "t3"]
    I = np.array([[1,0,0],
                  [0,1,0],
                  [0,0,1]])
    O = np.array([[0,1,0],
                  [0,0,1],
                  [1,0,0]])
    M0 = np.array([1,1,1])
    pn = PetriNet(P, T, P, T, I, O, M0)

    p1, p2, p3 = exprvar('p1'), exprvar('p2'), exprvar('p3')
    expected_expr = Or(And(p1, p2, p3))
    
    bdd = expr2bdd(expected_expr)

    marking = deadlock_reachable_marking(pn, bdd)

    assert marking == [1, 1, 1]


def test_004():
    P = ["p1", "p2", "p3"]
    T = ["t1", "t2", "t3"]
    I = np.array([[1,0,0],
                  [0,1,0],
                  [0,0,1]])
    O = np.array([[0,1,0],
                  [0,0,1],
                  [1,0,0]])
    M0 = np.array([1,0,1])
    pn = PetriNet(P, T, P, T, I, O, M0)

    p1, p2, p3 = exprvar('p1'), exprvar('p2'), exprvar('p3')
    expected_expr = Or(And(~p1, p2, p3),
                    And(p1, p2, ~p3),
                    And(p1, ~p2, p3))
    
    bdd = expr2bdd(expected_expr)

    marking = deadlock_reachable_marking(pn, bdd)

    assert marking == None


def test_005():
    P = ["p1", "p2", "p3"]
    T = ["t1", "t2", "t3"]
    I = np.array([[1,0,0],
                  [0,1,0],
                  [0,0,1]])
    O = np.array([[0,1,0],
                  [0,0,1],
                  [1,0,0]])
    M0 = np.array([1,0,1])
    pn = PetriNet(P, T, P, T, I, O, M0)

    p1, p2, p3 = exprvar('p1'), exprvar('p2'), exprvar('p3')
    expected_expr = Or(And(~p1, p2, p3),
                    And(p1, p2, ~p3),
                    And(p1, ~p2, p3))
    
    bdd = expr2bdd(expected_expr)

    marking = deadlock_reachable_marking(pn, bdd)

    assert marking == None

def test_006():
    P = ["p1", "p2"]
    T = ["t1"]
    I = np.array([[1, 0]])
    O = np.array([[1, 1]])
    M0 = np.array([1,0])
    pn = PetriNet(P, T, P, T, I, O, M0)

    p1, p2 = exprvar('p1'), exprvar('p2')
    expected_expr = p1
    
    bdd = expr2bdd(expected_expr)

    marking = deadlock_reachable_marking(pn, bdd)

    assert marking == [1, 1]


