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

# Metoda Eulera

def f(x, y):
    return 1

def f1(x, y):
    return 2*x

def f2(x, y):
    return 3*x**2

def f3(x, y):
    return 4*x**3

def f4(x, y):
    return 5*x**4

def f5(x, y):
    return 6*x**5

# rząd metody 1
def euler(f, a, b, n, y0):
    h = (b-a)/(n-1)
    x = np.linspace(a,b,n)
    y = np.zeros(len(x))
    y[0] = y0
    for i in range(0, n-1):
        y[i+1] = y[i] + h*f(x[i], y[i])
    return y
#print(euler(f, 0, 1, 10, 0.1))

x = np.linspace(0, 1, 10)
plt.plot(x, euler(f, 0, 1, 10, 0), "bo--" ,color='blue')
plt.plot(x, euler(f1, 0, 1, 10, 0), "bo--" ,color='blue')
plt.plot(x, euler(f2, 0, 1, 10, 0), "bo--" ,color='blue')
for k in range(1, 4):
    z = x**k
    plt.plot(x, z, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Metoda Eulera')
plt.grid()
plt.show()

# Metora Heuna

# rząd metody 2
def heun(f, a, b, n, y0):
    h = (b-a)/(n-1)
    x = np.linspace(a,b,n)
    y = np.zeros(len(x))
    y[0] = y0
    for i in range(0, n-1):
        y[i+1] = y[i] + h*(f(x[i], y[i]+0.5*h*f(x[i], y[i])))
    return y    
#print(heun(f, 0, 1, 10, 0.1))

x = np.linspace(0, 1, 10)
plt.plot(x, heun(f, 0, 1, 10, 0), "bo--" ,color='blue')
plt.plot(x, heun(f1, 0, 1, 10, 0), "bo--" ,color='blue')
plt.plot(x, heun(f2, 0, 1, 10, 0), "bo--" ,color='blue')
plt.plot(x, heun(f3, 0, 1, 10, 0), "bo--" ,color='blue')
for k in range(1, 5):
    z = x**k
    plt.plot(x, z, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Metoda Heuna')
plt.grid()
plt.show()

# Metoda Rungego-Kutty

#rząd metody 4
def runge_kutta(f, a, b, n, y0):
    h = (b-a)/(n-1)
    x = np.linspace(a,b,n)
    y = np.zeros(len(x))
    y[0] = y0
    for i in range(0, n-1):
        k1 = f(x[i], y[i])
        k2 = f(x[i]+0.5*h, y[i]+0.5*h*k1)
        k3 = f(x[i]+0.5*h, y[i]+0.5*h*k2)
        k4 = f(x[i]+h, y[i]+h*k3)
        y[i+1] = y[i] + h*(k1+2*k2+2*k3+k4)/6
    return y
#print(runge_kutta(f, 0, 1, 10, 0.1))

x = np.linspace(0, 1, 10)
plt.plot(x, runge_kutta(f, 0, 1, 10, 0), "bo--" ,color='blue')
plt.plot(x, runge_kutta(f1, 0, 1, 10, 0), "bo--" ,color='blue')
plt.plot(x, runge_kutta(f2, 0, 1, 10, 0), "bo--" ,color='blue')
plt.plot(x, runge_kutta(f3, 0, 1, 10, 0), "bo--" ,color='blue')
plt.plot(x, runge_kutta(f4, 0, 1, 10, 0), "bo--" ,color='blue')
plt.plot(x, runge_kutta(f5, 0, 1, 10, 0), "bo--" ,color='blue')
for k in range(1, 7):
    z = x**k
    plt.plot(x, z, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Metoda Rungego-Kutty')
plt.grid()
plt.show()

#zad2,3

#Ekstrapolcja Richardsona

def REX(y1, y2, p):
    '''
    y1 - wynik z kroku h
    y2 - wynik z kroku h/2
    p - rzad metody
    '''
    yR = np.zeros(len(y1))
    E = []
    for i in range(0,len(y1)):
        yR[i] = ((2**p)*y2[2*i] - y1[i])/(2**p-1)

        error = abs(y2[2*i] - y1[i])/((2**p)-1)
        E.append(error)
    print(E)
    return yR

x = np.linspace(0, 1, 11)
y2 = REX(euler(f1, 0, 1, 11, 0), euler(f1, 0, 1, 21, 0), 1)
plt.plot(x, x**2, color='red')
plt.plot(x, euler(f1, 0, 1, 11, 0), "bo--" ,color='blue')
plt.plot(x, y2, 'bo--', color='green')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ekstrapolacja Richardsona dla metody Euleara')
plt.grid()
plt.show()

y3 = REX(heun(f2, 0, 1, 11, 0), heun(f2, 0, 1, 21, 0), 2)
plt.plot(x, x**3, color='red')
plt.plot(x, heun(f2, 0, 1, 11, 0), "bo--" ,color='blue')
plt.plot(x, y3, 'bo--', color='green')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ekstrapolacja Richardsona dla metody Heuna')
plt.grid()
plt.show()

y4 = REX(runge_kutta(f4, 0, 1, 11, 0), runge_kutta(f4, 0, 1, 21, 0), 4)
plt.plot(x, x**5, color='red')
plt.plot(x, runge_kutta(f4, 0, 1, 11, 0), "bo--" ,color='blue')
plt.plot(x, y4, 'bo--', color='green')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ekstrapolacja Richardsona dla metody Rungego-Kutty')
plt.grid()
plt.show()


#zad4

def euler2(f, a, b, n, y0):
    x = np.linspace(a, b, n)
    x2 = np.linspace(a, b, 2*n - 1)
    y = [0 for i in range(len(x))]
    y2 = np.zeros(len(2*x-1))
    h = (b-a)/(n-1)
    h2 = h/2
    y[0] = y0
    y2[0] = y0
    yR = 0
    T = [yR]
    for i in range(0, len(x) - 1):
        #liczymy z krokiem h
        y1 = yR + h*f(x[i], yR)

        #liczymy z krokiem h/2
        yt = yR + h2*f(x2[2*i],yR)
        y2 = yt + h2*f(x2[2*i + 1],yt)

        yR = 2*y2 - y1
        T.append(yR)
    return T

x = np.linspace(0, 1, 11)
plt.plot(x, euler2(f1, 0, 1, 11, 0), "bo--" ,color='blue')
plt.plot(x, y2, 'bo--', color='green')
plt.plot(x, x**2, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Porównanie Ekstrapolacji czynnej z bierną dla metody Eulera')
plt.grid()
plt.show()
