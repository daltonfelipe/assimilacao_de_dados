#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 13:43:24 2018

@author: dalton
"""
import numpy as np
from runge_kutta4 import lorenz_RK4
import matplotlib.pyplot as plt

# numero de iterações
n_its = int(input("Quantas iterações? "))

#=====================================================
#### Set de confugurações iniciais
#=====================================================
# parametros para inicialização da solução real
s, r, b = 10, 28, 8/3       # rho, sigma, beta
fmax, h = 1000, 0.01        # num max deiteracoes e tempo de passo runge-kutta

#=====================================================
####  Solução Real
#=====================================================
x0, y0, z0 = -5.4458, -5.4841, 22.5606          # condicoes iniciais

x, y, z, t = lorenz_RK4(x0, y0, z0, r, s, b, fmax, h) # metodo de runge-kuta 4a ordem

#=====================================================
####  Solução Background
#=====================================================
xb0, yb0, zb0 = -5.9, -5.0, 24.0         # condicoes iniciais solução
xb, yb, zb, tb = lorenz_RK4(xb0, yb0, zb0, r, s, b, fmax, h) # metodo de runge-kuta 4a ordem

# ====================================================
#### Observações
# ====================================================

ob_f = int(input("Frequência das observações: ")) # frequência das observações
tmax = 600 + ob_f # tempo de assimilação

xob = np.zeros((tmax,1))
yob = np.zeros((tmax,1))
zob = np.zeros((tmax,1))

nobs = int(tmax/ob_f)    # numero de observações 
vec = np.arange(1, tmax-nobs, ob_f-1)

sc_x_noise = np.random.randn(int(nobs),1)
sc_y_noise = np.random.randn(int(nobs),1)
sc_z_noise = np.random.randn(int(nobs),1)

sd = float(input("Variancia do erro de observação: "))  # variância do erro de observação 
var = np.sqrt(sd)

print("\nFazendo os calculos ...\n")

# matriz de covariancia dos erros
Rx = var*sc_x_noise
Ry = var*sc_y_noise
Rz = var*sc_z_noise

j = 0
for i in vec - 1:
    i = int(i)
    xob[i] = x[i] + Rx[j]
    yob[i] = y[i] + Ry[j]
    zob[i] = z[i] + Rz[j]
    j += 1

v = 0.01*vec

# ====================================================
#### Correções sucessivas
# ====================================================

x_ob = np.zeros((3, fmax))
x_sc = np.zeros((3, fmax))
Wx = np.zeros((3,3)) # matriz peso

Wx = 0.5*np.eye(3)
xfc, yfc, zfc = np.zeros(fmax), np.zeros(fmax), np.zeros(fmax)

# condicao inicial de previsao
xfc[0], yfc[0], zfc[0] = xb[0], yb[0], zb[0]

x_ob[0, :len(xob)] = xob.T
x_ob[1, :len(yob)] = yob.T
x_ob[2, :len(zob)] = zob.T

x_sc[:,0] = [xb[0], yb[0], zb[0]]

for i in range(fmax - 1):
    
    if i >= tmax:
        x_ob[0, i+1] = 0
        x_ob[1, i+1] = 0
        x_ob[2, i+1] = 0
    
    xfc[i] = x_sc[0, i]
    yfc[i] = x_sc[1, i]
    zfc[i] = x_sc[2, i]
    
    k1xfc = h * (s * y[i] - s * xfc[i])
    k1yfc = h * (r * xfc[i] - yfc[i] - xfc[i] * z[i])
    k1zfc = h * (xfc[i] * yfc[i] - b * zfc[i])
    
    k2xfc = h * (s * (yfc[i] + (1 / 2.0) * k1yfc) - s * (xfc[i] + (1 / 2.0) * k1xfc))
    k2yfc = h * (r * (xfc[i] + (1 / 2.0) * k1xfc) - (yfc[i] + (1 / 2.0) * k1yfc) - (xfc[i] + (1 / 2.0) * k1xfc) * (zfc[i] + (1 / 2.0) * k1zfc))
    k2zfc = h * ((xfc[i] + (1 / 2.0) * k1xfc) * (yfc[i] + (1 / 2.0) * k1yfc) - b * (zfc[i] + (1 / 2.0) * k1zfc))
    
    k3xfc = h * (s * (yfc[i] + (1 / 2.0) * k2yfc) - s * (xfc[i] + (1 / 2.0) * k2xfc))
    k3yfc = h * (r * (xfc[i] + (1 / 2.0) * k2xfc) - (yfc[i] + (1 / 2.0) * k2yfc) - (xfc[i] + (1 / 2.0) * k2xfc) * (zfc[i] + (1 / 2.0) * k2zfc))
    k3zfc = h * ((xfc[i] + (1 / 2.0) * k2xfc) * (yfc[i] + (1 / 2.0) * k2yfc) - b * (zfc[i] + (1 / 2.0) * k2zfc))
    
    k4xfc = h * (s * (yfc[i] + k3yfc) - s * (xfc[i] + k3xfc))
    k4yfc = h * (r * (xfc[i] + k3xfc) - (yfc[i] + k3yfc) - (xfc[i] + k3xfc) * (zfc[i] + k3zfc))
    k4zfc = h * ((xfc[i] + k3xfc) * (yfc[i] + k3yfc) - b * (zfc[i] + k3zfc))
    
    xfc[i+1] = (xfc[i] + (1 / 6.0) * (k1xfc + 2.0 * k2xfc + 2.0 * k3xfc + k4xfc))
    yfc[i+1] = (yfc[i] + (1 / 6.0) * (k1yfc + 2.0 * k2yfc + 2.0 * k3yfc + k4yfc))
    zfc[i+1] = (zfc[i] + (1 / 6.0) * (k1zfc + 2.0 * k2zfc + 2.0 * k3zfc + k4zfc))

    # metodo correcoes sucessivas
    if np.all(x_ob[:,i+1] != 0):
        x_sc[:,i+1] = [xfc[i+1], yfc[i+1], zfc[i+1]]
        for j in range(n_its):
            x_sc[:,i+1] = x_sc[:,i+1] + Wx.dot(x_ob[:,i+1] - x_sc[:,i+1])  
    else:
        x_sc[:,i+1] = [xfc[i+1], yfc[i+1], zfc[i+1]]

# calculo de erros
x_err = np.abs(x.T - x_sc[0,:])
y_err = np.abs(y.T - x_sc[1,:])
z_err = np.abs(z.T - x_sc[2,:])

print("Concluido!")
print("""
    Método Correções sucessivas
    ===========================
    Valores utilizados:
+=========================================+
+    Número de iterações: \t{}
+    Freq. das observações: \t{}
+    Var. do erro de obs.: \t{}
+=========================================+
""".format(n_its, ob_f, sd))
print("""
    Resultados:
+=========================================+
    x0 = {} \ty0 = {} \tz0 = {}
    Média do erro de x = \t{}
    Média do erro de y = \t{}
    Média do erro de z = \t{}
""".format(
        x_sc[0, 0],
        x_sc[1, 0],
        x_sc[2, 0], 
        np.mean(x_err),
        np.mean(y_err),
        np.mean(z_err)))

#=====================================================
# Plot dos resultados
#=====================================================
# solucao de x
plt.figure("Fig1: Correções Sucessivas (Análise)")
plt.subplot(2, 1, 1)
plt.title("Soluções para x e z")
# solução real
plt.plot(t, x, '-.' ,label="Solução Real (x)", linewidth=1)
# solução background
plt.plot(t, xb, 'k--', label="Solução Background (x)", linewidth=1)
# observações
plt.plot(v, xob[0:tmax-nobs:ob_f-1], '.r', label="Observações")
# análise
plt.plot(t, x_sc[0,:], 'm-', label="Análise", linewidth=1)

plt.grid(linestyle="-.")
plt.legend(loc="upper left") #bbox_to_anchor=(0.8025, 1.75), shadow=True, ncol=2
plt.ylabel("Solução para x")

# solucao de z
plt.subplot(2, 1, 2)
# solução real
plt.plot(tb, z, '-.', label="Solução Real (z)", linewidth=1) # solução real
# solução background
plt.plot(tb, zb, "k--",label="Solução Background (z)", linewidth=1) # solução background
# observações
plt.plot(v, zob[0:tmax-nobs:ob_f-1], '.r', label="Observações")
# análise
plt.plot(t, x_sc[2,:], 'm-', label="Análise", linewidth=1)

plt.grid(linestyle="-.")
plt.legend(loc="lower left") # bbox_to_anchor=(0.8025, -0.4), shadow=True, ncol=2
plt.ylabel("Solução para z")
plt.xlabel("Tempo")

# ERROS
plt.figure("Fig2: Correções sucessivas (Erros)")

# erro de x
plt.subplot(2, 1, 1)
plt.title("Erros das soluções (Real - Análise) de x e z")
plt.plot(t, x_err, 'k-', label="Erro de x")
plt.ylabel("Erro de x")
plt.grid(linestyle="-.")
plt.legend(loc="best")

# erro de z
plt.subplot(2, 1, 2)
plt.plot(t, z_err, 'k-', label="Erro de z")
plt.ylabel("Erro de z")
plt.xlabel("Tempo")
plt.legend(loc="best")
plt.grid(linestyle="-.")
plt.show()