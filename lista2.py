#!/usr/bin/env python

import math
import numpy as np
import sympy
import matplotlib.pyplot as plt
import queue

#zad1
#a)
def funkcja():
    lista = [1/n**2 for n in range(1, 10**6 + 1)]
    return sum(lista)
print("Suma w a)", funkcja())

#b)
def funkcja2():
    suma = 0
    for i in range(1, 10**6 + 1):
        suma = suma + 1/i**2
    return suma
print("Suma w b)", funkcja2())

#c)
def funkcja3():
    suma = 0
    for i in range(10**6, 0, -1):
        suma = suma + 1/i**2
    return suma
print("Suma w c)", funkcja3())

#d)
def funkcja4():
    q = queue.PriorityQueue()
    for n in range(1, 10**6 + 1):
        q.put(1/n**2)
        
    while q.qsize()>1:
        min_lista1 = q.get()
        min_lista2 = q.get()
        sum = min_lista1 + min_lista2
        q.put(sum)
    
    return q.get()
print("Suma w d)", funkcja4())

#Wyniki:
#Dokładna wartość:
#             1.644934066848226436472415...
#   Suma w a) 1.64493306684877
#   Suma w b) 1.64493306684877
#   Suma w c) 1.6449330668487263
#   Suma w d) 1.6449330668487265


#zad2

#a)
def costam():
    ciąg= [1/(2**n) for n in range(0, 10**4 + 1)]
    return sum(ciąg)
print("Suma w a)", costam())

#b)
def costam2():
    n = 0
    for i in range(0, 10**4 + 1):
        n = n + 1/(2**i) 
    return n
print("Suma w b)", costam2())

#c)
def costam3():
    n = 1
    for i in range(10**4, 0, -1):
        n = n + 1/(2**i) 
    return n
print("Suma w c)", costam3())

#d)
def costam4():
    q = queue.PriorityQueue()
    for n in range(0, 10**4 + 1):
        q.put(1/(2**n))
        
    while q.qsize()>1:
        min_lista1 = q.get()
        min_lista2 = q.get()
        sum = min_lista1 + min_lista2
        q.put(sum)
    
    return q.get()
print("Suma w d)", costam4())
#Wyniki:
#Dokładnawartość:
#             2.0            
#   Suma w a) 2.0
#   Suma w b) 2.0
#   Suma w c) 1.9999999999999998
#   Suma w d) 2.0

#zad3

def funkcja3(n, X): 
    x = np.linspace(-1, 1, n+1) 
    y = abs(x)
    l = 0
    for j in range(0, n+1):
        m = np.prod(X-x[:j])*np.prod(X-x[j+1:])/np.prod(x[j]-x[:j])/np.prod(x[j]-x[j+1:])*y[j]
        l = l + m
    return l

x = np.linspace(-1, 1, 100)
plt.plot(x, abs(x))
for n in range(0, 21):
    ys = []
    for xs in x:
        ys.append(funkcja3(n, xs))
    plt.plot(x, ys , label='n={}'.format(n))
    plt.grid()
    plt.legend()
plt.show()

#zad4

def funkcja3(n, X): 
    x = np.linspace(-1, 1, n+1) 
    y = 1/(25*x**2 + 1)
    l = 0
    for j in range(0, n+1):
        m = np.prod(X-x[:j])*np.prod(X-x[j+1:])/np.prod(x[j]-x[:j])/np.prod(x[j]-x[j+1:])*y[j]
        l = l + m
    return l

x = np.linspace(-1, 1, 100)
plt.plot(x, 1/(25*x**2 + 1), c="magenta")
for n in range(0, 21):
    ys = []
    for xs in x:
        ys.append(funkcja3(n, xs))
    plt.plot(x, ys , label='n={}'.format(n))
    plt.grid()
    plt.legend()
plt.show()
