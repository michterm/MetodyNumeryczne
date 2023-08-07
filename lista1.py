#!/usr/bin/env python

import math
import sympy
import numpy as np

# zad1

def epsilonmaszynowy():
    epsilon = 1
    n = 0
    while 1 + epsilon/2 > 1:
        epsilon = epsilon/2
        n = n + 1
    return epsilon, n

def epsilonmaszynowy_a(a):
    epsilon = 1
    n = 0
    while a + epsilon/2 > a:
        epsilon = epsilon/2
        n = n + 1
    return epsilon,n

print("Epsilon maszynowy wynosi:", epsilonmaszynowy())
print("Epsilon maszynowy dla a=10 wynosi:", epsilonmaszynowy_a(10))
print("Epsilon maszynowy dla a=100 wynosi:", epsilonmaszynowy_a(100))
print("Epsilon maszynowy dla a=1000 wynosi:", epsilonmaszynowy_a(1000))
print("Epsilon maszynowy dla a=10000 wynosi:", epsilonmaszynowy_a(10000))

#zad2

def rekurencja(n):
    x = [1, 1/5] + [None]*(n-1)
    for i in range(1, n):
        x[i+1] = (26*x[i] - 5*x[i-1])/5
    return x

tablica = rekurencja(30) 
print("x_29 =", tablica[29])
print("x_30 =", tablica[30])

def rekurencja_w_tył(n, tablica):
    z = [None]*(n-1) + [tablica[29], tablica[30]]
    for i in range(n-1, 0, -1):
        z[i-1] = (26*z[i] - 5*z[i+1])/5
    return z

tablica2 = rekurencja_w_tył(30, tablica)
print("x_1 =", tablica2[0])
print("x_2 =", tablica2[1])

def rekurencja2(n):
    y = [1, 1/2] + [None]*(n-1)
    for i in range(1, n):
        y[i+1] = (5*y[i] - 2*y[i-1])/2
    return y

tab = rekurencja2(30) 
print("y_29 =", tab[29])
print("y_30 =", tab[30])

def rekurencja_w_tył2(n, tab):
    t = [None]*(n-1) + [tab[29], tab[30]]
    for i in range(n-1, 0, -1):
        t[i-1] = (5*t[i] - 2*t[i+1])/2
    return t

tab2 = rekurencja_w_tył2(30, tab)
print("y_1 =", tab2[0])
print("y_2 =", tab2[1])


#zad3

def f1(x):
    wynik1 = x - math.sqrt(1 + x**2)
    return wynik1
def f2(x):
    wynik2 = -1/(x + math.sqrt(1+x**2))
    return wynik2

xi = [10**x for x in range(4, 11)]

print("Wartości funkcji f1:")
for x in xi:
    print(f1(x))

print("Wartość funkcji f2:")
for x in xi:
    print(f2(x))

def dokładne_wart1(x):
    return x - sympy.sqrt(1 + x**2)

def dokładne_wart2(x):
    return -1/(x + sympy.sqrt(1 + x**2))

w1 = [dokładne_wart1(x).evalf() for x in xi]
print("Dokładne wartości f1(x):", w1)

w2 = [dokładne_wart2(x).evalf() for x in xi]
print("Dokładne wartości f2(x):", w2)

#zad4

def g1(x):
    return (x-1)**4

def g2(x):
    return (x**4 - 4*x**3 + 6*x**2 - 4*x + 1)

x = np.linspace(1 - math.pow(10, -3), 1 + math.pow(10, -3), 100)
print("Wartość funkcji g1", g1(x))
print("Wartość funkcji g2", g2(x))

def g_sympy(x):
    return sympy.Pow(x-1,4)

print("Dokładne wartości:", g_sympy(x))
