# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 20:05:42 2018

@author: Dalton Felipe
"""
import numpy as np

def lorenz_RK4(x0, y0, z0, r, s, b, n, h):
    """Metodo de Runge-Kutta 4a ordem."""
    # Inicializacao dos vetores
    q = np.linspace(0,n*h,n)
    x = np.empty(n)
    y = np.empty(n)
    z = np.empty(n)
    # set das condicoes iniciais
    x[0] = x0
    y[0] = y0
    z[0] = z0
    
    # laco para iteracao do Metodo 
    for i in range(n - 1):
        # calculo das inclinacoes kn
        k1x = h * (s * y[i] - s * x[i])
        k1y = h * (r * x[i] - y[i] - x[i] * z[i])
        k1z = h * (x[i] * y[i] - b * z[i])
    
        k2x = h * (s * (y[i] + (1 / 2.0) * k1y) - s * (x[i] + (1 / 2.0) * k1x))
        k2y = h * (r * (x[i] + (1 / 2.0) * k1x) - (y[i] + (1 / 2.0) * k1y) - (x[i] + (1 / 2.0) * k1x) * (z[i] + (1 / 2.0) * k1z))
        k2z = h * ((x[i] + (1 / 2.0) * k1x) * (y[i] + (1 / 2.0) * k1y) - b * (z[i] + (1 / 2.0) * k1z))
    
        k3x = h * (s * (y[i] + (1 / 2.0) * k2y) - s * (x[i] + (1 / 2.0) * k2x))
        k3y = h * (r * (x[i] + (1 / 2.0) * k2x) - (y[i] + (1 / 2.0) * k2y) - (x[i] + (1 / 2.0) * k2x) * (z[i] + (1 / 2.0) * k2z))
        k3z = h * ((x[i] + (1 / 2.0) * k2x) * (y[i] + (1 / 2.0) * k2y) - b * (z[i] + (1 / 2.0) * k2z))
    
        k4x = h * (s * (y[i] + k3y) - s * (x[i] + k3x))
        k4y = h * (r * (x[i] + k3x) - (y[i] + k3y) - (x[i] + k3x) * (z[i] + k3z))
        k4z = h * ((x[i] + k3x) * (y[i] + k3y) - b * (z[i] + k3z))
        
        x[i+1] = (x[i] + (1 / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x))
        y[i+1] = (y[i] + (1 / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y))
        z[i+1] = (z[i] + (1 / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z))
    
    return [x, y, z, q]

import matplotlib.pyplot as plt

x0, y0, z0 = -1.5, 1.5, 25

s, r, b = 10, 28, 8/3
n, h = 10000, 0.01

x, y, z = lorenz_RK4(x0, y0, z0, r, s, b, n, h)

plt.plot(x, y)
plt.legend()
