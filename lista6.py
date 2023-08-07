#!/usr/bin/env python
import math
import numpy as np
import sympy
import matplotlib.pyplot as plt
import queue
from scipy.interpolate import CubicSpline
from scipy.integrate import simps
import random

#zad1
 
def f(x):
    return 1
def f2(x):
    return x
def f22(x):
    return x+1
def f3(x):
    return x**2 
def f4(x):
    return x**3

def metoda_trapezów(f, a, b, n):
    h = (b - a)/n
    x = np.linspace(a, b, n+1)
    y = [f(i) for i in x]
    S = h*(sum(y) - y[0]/2 - y[-1]/2) 
    return S

print("Metoda trapezów dla f:", metoda_trapezów(f, 0, 1, 10))
print("Metoda trapezów dla f2:", metoda_trapezów(f2, 0, 1, 10))
print("Metoda trapezów dla f22:", metoda_trapezów(f22, 0, 1, 10))
print("Metoda trapezów dla f3:", metoda_trapezów(f3, 0, 1, 10))
print("Metoda trapezów dla f4:", metoda_trapezów(f4, 0, 1, 10))
# Rząd metory = 2.

#zad2

def f5(x):
    return x**4
def f6(x):
    return x**5

def metoda_paraboli(f, a, b, n):
    h = (b - a)/n
    x = np.linspace(a, b, n+1)
    y = [f(a+k*h) for k in range(len(x))]
    S = h/3*(y[0] + y[n] + sum([4*y[2*i-1] + 2*y[2*i-2] for i in range(1, n//2 + 1)]))
    return S


print("Metoda paraboli dla f:", metoda_paraboli(f, 0, 1, 10))
print("Metoda paraboli dla f2:", metoda_paraboli(f2, 0, 1, 10))
print("Metoda paraboli dla f22:", metoda_paraboli(f22, 0, 1, 10))
print("Metoda paraboli dla f3:", metoda_paraboli(f3, 0, 1, 10))
print("Metoda paraboli dla f4:", metoda_paraboli(f4, 0, 1, 10))
print("Metoda paraboli dla f5:", metoda_paraboli(f5, 0, 1, 10))
print("Metoda paraboli dla f6:", metoda_paraboli(f6, 0, 1, 10))
# Rząd metory = 4.

#zad3

def f12(x):
    return x**12

def f13(x):
    return x**13

def f14(x):
    return x**14

def Romberg(f, a, b, n):
    T = [[0]*(n+1) for _ in range(n+1)]
    for i in range(0, n+1):
        h = (b - a)/(2**i)
        T[0][i] = h*(sum(f(a + k*h) for k in range(0, 2**i + 1)) - 1/2*(f(a) + f(b)))
    for i in range(0, n+1):
        for m in range(1, n+1-i):
            T[1][i] = T[m-1][i+1] + (T[m-1][i+1] - T[m-1][i])/(2**(2*m) - 1)
        for i in range(0, n+1):
            for m in range(1, n+1-i):
                T[m][i] = T[m-1][i+1] + (T[m-1][i+1] - T[m-1][i])/(2**(2*m) - 1)
    return T[n][0]

print("Metoda Romberga dla f:", Romberg(f, 0, 1, 2))
print("Metoda Romberga dla f2:", Romberg(f2, 0, 1, 2))
print("Metoda Romberga dla f22:", Romberg(f22, 0, 1, 2))
print("Metoda Romberga dla f3:", Romberg(f3, 0, 1, 2))
print("Metoda Romberga dla f4:", Romberg(f4, 0, 1, 2))
print("Metoda Romberga dla f5:", Romberg(f5, 0, 1, 2))
print("Metoda Romberga dla f6:", Romberg(f6, 0, 1, 2))
print("Metoda Romberga dla f12:", Romberg(f12, 0, 1, 2) - 1/12)
print("Metoda Romberga dla f13:", Romberg(f13, 0, 1, 2) - 1/13)
print("Metoda Romberga dla f14:", Romberg(f14, 0, 1, 2) - 1/14)
# Rząd metodu = 12.

#zad4

for k in range(0, 21):
    R = Romberg(lambda x: x**k, 0, 1, 10)
    print(R)

def fs(x):
    return math.sin(x)

print("Wynik dla cosx", Romberg(fs, 0, math.pi, 10))

def fc(x):
    return math.cos(x)
print("Wynik dla sinx:", Romberg(fc, 0, math.pi/2, 10))

def ft(x):
    return 1/(1+x**2)
print("Wynik dla tanx", Romberg(ft, 0, math.pi, 10))

#zad5

def ff(x):
    return np.sin(x)

def Romberg2(f, a, b, n=20, epsilon=1e-6):
    T = [[0]*(n+1) for _ in range(n+1)]
    for i in range(0, n+1):
        h = (b - a)/(2**i)
        T[0][i] = h*(sum(f(a + k*h) for k in range(0, 2**i + 1)) - 1/2*(f(a) + f(b)))
    for i in range(0, n+1):
        for m in range(1, n+1-i):
            T[1][i] = T[m-1][i+1] + (T[m-1][i+1] - T[m-1][i])/(2**(2*m) - 1)
        for i in range(0, n+1):
            for m in range(1, n+1-i):
                T[m][i] = T[m-1][i+1] + (T[m-1][i+1] - T[m-1][i])/(2**(2*m) - 1)
            if abs(T[i][i] - T[i-1][i-1]) < epsilon:
                return T[n][0]
    raise Exception("Nie zbiega do wymaganej dokładności.")


a = 0
b = np.pi
wynik = 2

# Metoda Romberga
romberg = Romberg2(ff, a, b, n=20, epsilon=1e-6)
print("Przybliżenie:", romberg)
print("Błąd bezwzględny:", abs(wynik - romberg))

# Metoda trapezów 
n = 20
x = np.linspace(a, b, n)
y = ff(x)
trapez = np.trapz(y, x)
print("Przybliżenie:", trapez)
print("Błąd bezwzględny:", abs(wynik - trapez))

# Metoda parabol 
parabol = simps(y, x)
print("Przybliżenie:", parabol)
print("Błąd:", abs(wynik - parabol))
