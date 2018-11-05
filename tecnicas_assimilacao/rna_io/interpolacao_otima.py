#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 13:43:24 2018

@author: dalton
"""
import numpy as np
from runge_kutta4 import lorenz_RK4
import matplotlib.pyplot as plt
import json
#=====================================================
#### Set de confugurações iniciais
#=====================================================
# parametros para inicialização da solução real

confs = {}

s, r, b = 10, 28, 8/3       # rho, sigma, beta
fmax, h = 1000, 0.01        # num max de iteracoes e tempo de passo runge-kutta

confs['model_params'] = {}
confs["model_params"]["beta"] = b
confs["model_params"]["rho"] = r
confs["model_params"]["sigma"] = s
confs["model_params"]["fmax"] = fmax
confs["model_params"]["h"] = h 

#=====================================================
#  Solução Real
#=====================================================
x0, y0, z0 = -5.4458, -5.4841, 22.5606          # condicoes iniciais

confs["model_params"]["initial_cond"] = {}
confs["model_params"]["initial_cond"]['x0'] = x0
confs["model_params"]["initial_cond"]['y0'] = y0
confs["model_params"]["initial_cond"]['z0'] = z0

x, y, z, t = lorenz_RK4(x0, y0, z0, r, s, b, fmax, h) # metodo de runge-kuta 4a ordem

#=====================================================
#  Solução Background
#=====================================================
xb0, yb0, zb0 = -5.9, -5.0, 24.0         # condicoes iniciais solução
xb, yb, zb, tb = lorenz_RK4(xb0, yb0, zb0, r, s, b, fmax, h) # metodo de runge-kuta 4a ordem

#====================================================
#  Observações
#====================================================

ob_f = int(input("Frequência das observações: ")) # frequência das observações
tmax = 600 + ob_f # tempo de assimilação

confs["io_params"] = {}
confs["io_params"]["ob_f"] = ob_f
confs["io_params"]["tmax"] = tmax

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

confs["io_params"]["var"] = sd 

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

#====================================================
#  Matriz erro covariancia
#====================================================

Bx = np.zeros((3, 3))
Bxi = np.zeros((3, 3, tmax))
R = np.zeros((3, 3))
Ri = np.zeros((3, 3, tmax))

# matriz erro de covariancia das observacoes R

for i in vec - 1:
    Ri[0, 0, i] = ((x[i] - xob[i]).T).dot(x[i] - xob[i])
    Ri[0, 1, i] = ((x[i] - xob[i]).T).dot(y[i] - yob[i])
    Ri[0, 2, i] = ((x[i] - xob[i]).T).dot(z[i] - zob[i])
    Ri[1, 0, i] = ((y[i] - yob[i]).T).dot(x[i] - xob[i])
    Ri[1, 1, i] = ((y[i] - yob[i]).T).dot(y[i] - yob[i])
    Ri[1, 2, i] = ((y[i] - yob[i]).T).dot(z[i] - zob[i])
    Ri[2, 0, i] = ((z[i] - zob[i]).T).dot(x[i] - xob[i])
    Ri[2, 1, i] = ((z[i] - zob[i]).T).dot(y[i] - yob[i])
    Ri[2, 2, i] = ((z[i] - zob[i]).T).dot(z[i] - zob[i])

Rj = np.zeros((3, 3, nobs))

for j in range(3):
    for k in range(3):
        Rj[j, k, :] = Ri[j, k, vec-1]
        R[j, k] = (1/nobs)*R[j, k].sum()

# matriz erro de covariancia da solucao background
B = np.zeros((3,3))

for i in range(tmax):
    Bxi[0, 0, i] = ((x[i] - xb[i]).T)*(x[i] - xb[i])
    Bxi[0, 1, i] = ((x[i] - xb[i]).T)*(y[i] - yb[i])
    Bxi[0, 2, i] = ((x[i] - xb[i]).T)*(z[i] - zb[i])
    Bxi[1, 0, i] = ((y[i] - yb[i]).T)*(x[i] - xb[i])
    Bxi[1, 1, i] = ((y[i] - yb[i]).T)*(y[i] - yb[i])
    Bxi[1, 2, i] = ((y[i] - yb[i]).T)*(z[i] - zb[i])
    Bxi[2, 0, i] = ((z[i] - zb[i]).T)*(x[i] - xb[i])
    Bxi[2, 1, i] = ((z[i] - zb[i]).T)*(y[i] - yb[i])
    Bxi[2, 2, i] = ((z[i] - zb[i]).T)*(z[i] - zb[i])

for j in range(3):
    for k in range(3):
        Bx[j, k] = (1/tmax)*Bxi[j, k, :].sum()

x_ob = np.zeros((3,fmax))
x_oi = np.copy(x_ob)
Wx = np.zeros((3,3))

xfc, yfc, zfc = np.zeros(fmax), np.zeros(fmax), np.zeros(fmax)
xfc[0], yfc[0], zfc[0] = xb[0], yb[0], zb[0]

Wx = Bx.dot(np.linalg.pinv(Bx+R))

x_ob[0, :len(xob)] = xob.T
x_ob[1, :len(yob)] = yob.T
x_ob[2, :len(zob)] = zob.T

x_oi[:,0] = [xb[0], yb[0], zb[0]]

for i in range(fmax - 1):
    
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
    
    xfc[i+1] = (xfc[i] + (1 / 6.0) * (k1xfc + 2.0 * k2xfc + 2.0 * k3xfc + k4xfc))
    yfc[i+1] = (yfc[i] + (1 / 6.0) * (k1yfc + 2.0 * k2yfc + 2.0 * k3yfc + k4yfc))
    zfc[i+1] = (zfc[i] + (1 / 6.0) * (k1zfc + 2.0 * k2zfc + 2.0 * k3zfc + k4zfc))

    if np.all(x_ob[:,i+1] != 0):
        x_oi[:,i+1] = [xfc[i+1], yfc[i+1], zfc[i+1]]
        x_oi[:,i+1] = x_oi[:,i+1] + Wx.dot(x_ob[:,i+1] - x_oi[:,i+1])  
    else:
        x_oi[:,i+1] = [xfc[i+1], yfc[i+1], zfc[i+1]]

# calculo de erros
x_err = np.abs(x.T - x_oi[0,:])
y_err = np.abs(y.T - x_oi[1,:])
z_err = np.abs(z.T - x_oi[2,:])


with open("confs.json","w") as conf_file:
    conf_file.write(json.dumps(confs, indent=2))

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



