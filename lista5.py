#!/usr/bin/env python
import math
import numpy as np
import sympy
import matplotlib.pyplot as plt
import queue
from scipy.interpolate import CubicSpline
import random
from scipy.linalg import solve

#zad1

A = np.array([[5, 1, 1],
            [1, 2, 1],
            [1, 1, 3]])
B = np.array([[7], [4], [5]])
print(np.linalg.solve(A, B))
def zmiana_kolumn(A, B, column):
    A_c = A.copy()
    for i in range(len(A_c)):
        for j in range(len(A_c)):
            if j == column:
                A_c[i][j] = B[i]
    return A_c

def cramer(A, B):
    A_c = A.copy()
    W = np.linalg.det(A)
    x_c = np.zeros(len(A))
    if abs(W) < 1: #żeby uniknąć błędów numerycnzych
        print('Układ jest sprzeczny lub ma nieskończenie wiele rozwiązań !')
        return None
    else:    
        for i in range(len(A)):
            A_c = zmiana_kolumn(A_c, B, i)
            Ws = np.linalg.det(A)
            x_c[i] = Ws/W
        return x_c
print("Rozwiązanie metodą Cramera:", cramer(A, B))

#zad2

M = np.array([[5, 1, 1],
            [1, 2, 1],
            [1, 1, 3]],'float64')
b = np.array([[7], [4], [5]],'float64')

def odwrotna(D):
    for i in range(len(D)):
        D[i, i] = 1/D[i, i]
    return D

def Jacobi(M, b, potęga=1):
    D = np.diagflat(np.diag(M))
    R = M - D
    D_2 = odwrotna(D)
    x = np.array([[0] for i in range(len(M))],'float64')
    if 0 in [M[i,i] for i in range(len(M))]:
        return "Jest 0 na diagonali" 
    if abs(np.linalg.det(M)) < 1/1000: #żeby uniknąć błędów numerycznych
        print('Układ jest sprzeczny lub ma nieskończenie wiele rozwiązań !')
        return None
    for i in range(0, len(M)**potęga):
        x_j = D_2@(b - R@x)
    return x_j
print("Rozwiązanie dla metody Jacobiego:\n", Jacobi(M, b))

#zad3

x_1 = np.linalg.solve(A, B)
print("Rozwiazanie metodą wbudowaną cramer:\n", x_1)

x_2 = np.linalg.solve(M, b)
print(print("Rozwiazanie metodą wbudowaną jacobi:\n", x_2))

#dla cramera 
x_c = cramer(A, B)
b_n = np.linalg.norm(B)

r_1 = A@x_c - B
r_n  = np.linalg.norm(r_1)

r = r_n/b_n
print("Błąd dla cramera:", r)

#dla jacobiego
x_j1 = Jacobi(M, b, potęga=1)
b_m = np.linalg.norm(b)

r_2 = M@x_j1 - b
r_m  = np.linalg.norm(r_2)

rn1 = r_m/b_m
print("Błąd dla jacobiego dla n=1:", rn1)

x_j2 = Jacobi(M, b, potęga=2)
r_3 = M@x_j2 - b
r_h = np.linalg.norm(r_3)

rn2 = r_h/b_m
print("Błąd dla jacobiego dla n=2:", rn2)

x_j3 = Jacobi(M, b, potęga=3)
r_4 = M@x_j3 - b
r_hp = np.linalg.norm(r_4)

rn3 = r_hp/b_m
print("Błąd dla jacobiego dla n=3:", rn3)
