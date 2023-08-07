#!/usr/bin/env python

import math
import numpy as np
import sympy
import matplotlib.pyplot as plt
import queue
from scipy.interpolate import CubicSpline

#zad1

def Aproksymacja(n):
    lista = []
    xi = np.linspace(-1, 1, 100)
    yi = [abs(x) for x in xi]
    for i in range(0, n+1):
        p = np.polyfit(xi, yi, i)
        lista.append(p)
    return lista

x = np.linspace(-1, 1, 1000)
y = abs(x)
plt.plot(x, y, label='|x|')
for i in range(0, 21):
    cos1 = Aproksymacja(i)
    ye = np.polyval(cos1[-1], x)
    plt.plot(x, ye, label='stopień ' + str(i))
plt.legend()
plt.grid()
plt.show()

#zad2

def Aproksymacja2(n):
    lista = []
    xi = np.linspace(-1, 1, 20)
    yi = [1/(25*x**2 + 1) for x in xi]
    for i in range(0, n+1):
        p = np.polyfit(xi, yi, i)
        lista.append(p)
    return lista

x = np.linspace(-1, 1, 1000)
y = 1/(25*x**2 + 1)
plt.plot(x, y, label='1/(25*x**2 + 1)')
for i in range(0, 21):
    cos = Aproksymacja2(i)
    y = np.polyval(cos[-1], x)
    plt.plot(x, y, label='stopień ' + str(i))
plt.legend()
plt.grid()
plt.show()

#zad3

def Aproksymacja_sklejana(n):
    xi = np.linspace(-1, 1, n+1)
    yi = [abs(x) for x in xi]
    t = CubicSpline(xi, yi)
    return t

#print(Aproksymacja_sklejana(20))

x = np.linspace(-1, 1, 1000)
y = abs(x)
plt.plot(x, y, label='|x|')
for i in range(1, 21):
    v = Aproksymacja_sklejana(i)
    plt.plot(x, v(x), label='stopień ' + str(i))
plt.legend()
plt.grid()
plt.show()

#zad4

def Aproksymacja_sklejana2(n):
    xi = np.linspace(-1, 1, n+1)
    yi = [1/(25*x**2 + 1) for x in xi]
    t = CubicSpline(xi, yi)
    return t

#print(Aproksymacja_sklejana(20))

x = np.linspace(-1, 1, 1000)
y = 1/(25*x**2 + 1)
plt.plot(x, y, label='1/(25*x**2 + 1)')
for i in range(1, 21):
    v = Aproksymacja_sklejana2(i)
    plt.plot(x, v(x), label='stopień ' + str(i))
plt.legend()
plt.grid()
plt.show()

#zad5

def Aproksymacja5(n):
    y = np.poly1d([1, 1, 2, 3, 5, 10, 40])
    xi = [x for x in range(1, n+1)]
    yi = [y(x) for x in xi] 
    p = np.polyfit(xi, yi, deg=6)
    z = np.poly1d(p)  
    return z

x = np.linspace(1, 8, 1000)
y = np.poly1d([1, 1, 2, 3, 5, 10, 40])
z = Aproksymacja5(7)
plt.plot(x, y(x), c="blue")
plt.plot(x, z(x), c="yellow")
plt.grid()
plt.legend()
plt.show()
print(y)
print(Aproksymacja5(7))
print(Aproksymacja5(8))
