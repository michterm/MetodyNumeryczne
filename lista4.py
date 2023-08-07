#!/usr/bin/env python

import math
import numpy as np
import sympy
import matplotlib.pyplot as plt
import queue
from scipy.interpolate import CubicSpline

#zad1

A = np.matrix([[1,1,2,1], [0,1,4,3], [4,6,8,6], [5,5,-5,5]])
print(A)
print(np.linalg.det(A))

B = np.matrix([[2,1,1,2],[1,2,1,2],[2,1,2,1],[2,2,2,2]])
print(B)
print(np.linalg.det(B))

#zad2

def funkcja():
    m = None
    R = None
    X =None
    for _ in range(100):
        M = np.random.randint(10, size=(3, 3))
        b = np.random.randint(10, size=(3, 1))
        if abs(np.linalg.det(M)) > 1:
            x = np.linalg.solve(M, b)
            r = M@x - b
            if R is None or np.linalg.norm(r) > np.linalg.norm(R):
                R = r
                m = M
                X = x       
    return M,x,r,b
print(funkcja())
