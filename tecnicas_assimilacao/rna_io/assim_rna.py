#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 13:43:24 2018
@author: dalton
"""
import pandas as pd
import numpy as np
from runge_kutta4 import lorenz_RK4
import matplotlib.pyplot as plt
import json
from mlp import NN_MLP

nn = NN_MLP()

with open("confs.json","r") as file:
    confs = json.load(file)

with open("rna_confs.json","r") as file:
    rna_confs = json.load(file)

# carrega os pesos treinados
pesos = pd.read_csv("data/x_pesos.csv")

pesos0 = np.column_stack((pesos['pesos0a'], pesos['pesos0b']))
pesos1 = np.array(pesos['pesos1'])

# num max de iteracoes e tempo de passo runge-kutta
fmax = confs["model_params"]["fmax"]
h = confs["model_params"]["h"]

#=====================================================
#  Solução Real
#=====================================================

# pega os valores utilizados no io salvos em arquivo
x0 = confs["model_params"]["initial_cond"]['x0']
y0 = confs["model_params"]["initial_cond"]['y0']
z0 = confs["model_params"]["initial_cond"]['z0']

s = confs["model_params"]["sigma"]
b = confs["model_params"]["beta"]
r = confs["model_params"]["rho"]

x, y, z, t = lorenz_RK4(x0, y0, z0, r, s, b, fmax, h) # metodo de runge-kuta 4a ordem

#=====================================================
#  Solução Background
#=====================================================
xb0, yb0, zb0 = -5.9, -5.0, 24.0         # condicoes iniciais solução
xb, yb, zb, tb = lorenz_RK4(xb0, yb0, zb0, r, s, b, fmax, h) # metodo de runge-kuta 4a ordem

#====================================================
#  Observações
#====================================================

ob_f = confs["io_params"]["ob_f"]
tmax = confs["io_params"]["tmax"]

xob = np.zeros((tmax,1))
yob = np.zeros((tmax,1))
zob = np.zeros((tmax,1))

nobs = int(tmax/ob_f)   # numero de observações 
vec = np.arange(1, tmax-nobs, ob_f-1)

sc_x_noise = np.random.randn(int(nobs),1)
sc_y_noise = np.random.randn(int(nobs),1)
sc_z_noise = np.random.randn(int(nobs),1)

sd = confs["io_params"].get("sd")
var = confs["io_params"].get("var")

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

x_ob = np.zeros((3,fmax))
x_oi = np.copy(x_ob)

xfc, yfc, zfc = np.zeros(fmax), np.zeros(fmax), np.zeros(fmax)
xfc[0], yfc[0], zfc[0] = xb[0], yb[0], zb[0]


x_ob[0, :len(xob)] = xob.T
x_ob[1, :len(yob)] = yob.T
x_ob[2, :len(zob)] = zob.T

x_oi[:,0] = [xb[0], yb[0], zb[0]]

c = 0
xob = x_ob[0]
# variavel de entrada para previsao da rede neural
features = np.array([x, xob]).T

# reduz a dimensao dos valores com o modulo do vetor saida
scaler = rna_confs["scaler"]
features /= scaler

for i in range(fmax - 1):
    c += 1
    if i >= tmax:
        x_ob[0, i+1] = 0
        x_ob[1, i+1] = 0
        x_ob[2, i+1] = 0
    
    xfc[i] = x_oi[0, i]
    yfc[i] = x_oi[1, i]
    zfc[i] = x_oi[2, i]
    
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
    
    if i%25 == 0:
        xfc[i-1] = nn.ext_previsao(features[0], pesos0, pesos1)
        
    
    xfc[i+1] = (xfc[i] + (1 / 6.0) * (k1xfc + 2.0 * k2xfc + 2.0 * k3xfc + k4xfc))
    yfc[i+1] = (yfc[i] + (1 / 6.0) * (k1yfc + 2.0 * k2yfc + 2.0 * k3yfc + k4yfc))
    zfc[i+1] = (zfc[i] + (1 / 6.0) * (k1zfc + 2.0 * k2zfc + 2.0 * k3zfc + k4zfc))

    if np.all(x_ob[:,i+1] != 0):
        x_oi[:,i+1] = [xfc[i+1], yfc[i+1], zfc[i+1]]
    else:
        x_oi[:,i+1] = [xfc[i+1], yfc[i+1], zfc[i+1]]

# calculo de erros
x_err = np.abs(x.T - x_oi[0,:])
y_err = np.abs(y.T - x_oi[1,:])
z_err = np.abs(z.T - x_oi[2,:])


print("Concluido!")
print("""
    Método Interpolação Ótima
    ===========================
    Valores utilizados:
+=========================================+
+    Freq. das observações: \t{}
+    Var. do erro de obs.: \t{}
+=========================================+
""".format(ob_f, sd))
print("""
    Resultados:
+=========================================+
    x0 = {} \ty0 = {} \tz0 = {}
    Média do erro de x = \t{}
    Média do erro de y = \t{}
    Média do erro de z = \t{}
""".format(
        x_oi[0, 0],
        x_oi[1, 0],
        x_oi[2, 0], 
        np.mean(x_err),
        np.mean(y_err),
        np.mean(z_err)))


#=====================================================
# Plot dos resultados
#=====================================================
# solucao de x
plt.figure("Fig1: Interpolação Ótima (Análise)")
plt.subplot(2, 1, 1)
plt.title("Soluções para x e z")
# solução real
plt.plot(t, x, '-.' ,label="Solução Real (x)", linewidth=1)
# solução background
plt.plot(t, xb, 'k--', label="Solução Background (x)", linewidth=1)
# observações
plt.plot(v, xob[0:tmax-nobs:ob_f-1], '.r', label="Observações")
# análise
plt.plot(t, x_oi[0,:], 'm-', label="Análise", linewidth=1)

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
plt.plot(t, x_oi[2,:], 'm-', label="Análise", linewidth=1)

plt.grid(linestyle="-.")
plt.legend(loc="lower left") # bbox_to_anchor=(0.8025, -0.4), shadow=True, ncol=2
plt.ylabel("Solução para z")
plt.xlabel("Tempo")

# ERROS
plt.figure("Fig2: Interpolação Ótima (Erros)")

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

# salva os dados para treino da rede
x.tofile("data/x_model1.dat","\n")
xob.tofile("data/x_ob1.dat","\n")
x_oi[0].tofile("data/x_oi1.dat","\n")
#Rx.tofile("data/rx_ob.dat","\n")



