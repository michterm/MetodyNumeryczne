#!/usr/bin/env python
import math
import numpy as np
import sympy
import matplotlib.pyplot as plt
import queue
from scipy.interpolate import CubicSpline
from scipy.integrate import simps
from scipy.linalg import solve
import random
from lista5 import Jacobi


#zad1

def u(x, t):
    if x<0 or x>1 and t<0:
        print("Error: x or t out of range")
    if x == 0 or x == 1:
        return 0
    S = np.sin(np.pi*x)
    return S

def delta_u(x, t, h): # h = 1/M
    if x<0 or x>1 and t<0:
        print("Error: x or t out of range")
    if x == 0 or x == 1:
        return 0
    F = u(x + h, t) - 2*u(x, t) + u(x - h, t)/h**2
    return F

def operator_L(u, x, t, h, k, sigma):
    if x<0 or x>1 and t<0:
        print("Error: x or t out of range")
    if x == 0 or x == 1:
        return 0
    G = (u(x, t +k) - u(x, t))/k - (sigma*delta_u(x, t + k, h) + (1 - sigma)*delta_u(x, t, h))
    return G

def f(x, t):
    return 0

def x0(x):
    return 0

def x1(x):
    return 0

def t0(x):
    return math.sin(math.pi*x)

def Funkcja(f, u_t0, u_x0, u_xl, T, l, M, N, sigma):
    h =l/N
    k = T/M
    v = np.zeros((M+1, N+1))
    for j in range(0, N+1):
        v[0, j] = u_t0(j*h)
    for i in range(0, M+1):
        v[i, 0] = u_x0(i*k)
        v[i, N] = u_xl(i*k)
    alpha = k/(h**2)
    if sigma < 1 - (h**2)/(2*k):
        print("Brak stabilności metody")
    A = np.matrix(np.zeros((N-1, N-1)))
    for i in range(0, N-1):
        A[i, i] = 1 + 2*sigma*alpha
        if i > 0:
            A[i, i-1] = -sigma*alpha
        if i < N-2:
            A[i, i+1] = -sigma*alpha
    for i in range(1, M+1):
        F = np.zeros(N-1)
        for j in range(0, N-1):
            F[j] = k*f((j+1)*h, (i+0.5)*k) + v[i-1, j]*alpha*(1-sigma) + v[i-1, j+1]*(1- 2*alpha*(1-sigma)) + v[i-1, j+2]*alpha*(1-sigma)
            if j == 0:
                F[j] += sigma*alpha*v[i, 0]
            if j == N-2:
                F[j] += sigma*alpha*v[i, N]
        v[i, 1:N] = np.linalg.solve(A, F)
        v[i, 1:N] = Jacobi(A, F, potęga=1)
    return v

s1 = Funkcja(f, t0, x0, x1, 1, 1, 210, 10, 0)
s2 = Funkcja(f, t0, x0, x1, 1, 1, 210, 20, 0.5)
s3 = Funkcja(f, t0, x0, x1, 1, 1, 10, 20, 1)

M = 210
h = 1/M


x = np.linspace(0, 1, 21)
xi = np.linspace(0, 1, 11)
for i in range(len(s1)):
    plt.plot(xi, s1[i])
plt.title(f'Wykres dla $\\sigma = 0$')    
#plt.legend()
plt.grid()
plt.show()

for i in range(len(s2)):
    plt.plot(x, s2[i])
plt.title(f'Wykres dla $\\sigma = 0.5$')
#plt.legend()
plt.grid()
plt.show()

for i in range(len(s3)):
    plt.plot(x, s3[i])
plt.title(f'Wykres dla $\\sigma = 1$')
#plt.legend()
plt.grid()
plt.show()

def stabilnosc(k, sigma):
    for _ in range(len(s1)):
        if sigma < 1 - (h**2)/(2*k):
            print("Brak stabilności metody")
    for _ in range(len(s1)):
        if sigma < 1 - (h**2)/(2*k):
            print("Brak stabilności metody")
    for _ in range(len(s3)):
        if sigma < 1 - (h**2)/(2*k):
            print("Brak stabilności metody")
            

    

