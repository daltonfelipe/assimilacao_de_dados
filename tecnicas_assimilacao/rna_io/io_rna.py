#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 19:47:48 2018
@author: dalton
"""
import numpy as np
import pandas as pd
from mlp import NN_MLP 
import matplotlib.pyplot as plt

# carregamento dos dados
# entradas
x_model = np.loadtxt('data/x_model1.dat')
x_ob = np.loadtxt('data/x_ob1.dat')
# saidas
x_io = np.loadtxt('data/x_oi1.dat')
#
## matriz de entrada
features = np.array([x_model[0:625:25-1][:25], x_ob[::25-1][:25]]).T

x_ob_test = np.zeros((len(x_model)))

for i in range(x_ob.size):
    x_ob_test[i] = x_ob[i]

features_test = np.array([x_model, x_ob_test]).T

## matriz de saida
target = np.zeros((25, 1))
#
for i in np.arange(len(target)):
    target[i][0] = x_io[i*24]
#
## inicializacao da classe
nn = NN_MLP(epocas=20000,
            erro=1e-10,
            taxa_aprendizagem=0.05,
            tamanho_camada_oculta=10, 
            momento=1)
#
## treino da rede
## reduz a dimensao dos valores com o modulo do vetor saida
scaler = np.linalg.norm(target)
target /= scaler
features /= scaler
features_test /= scaler
#
nn.treinar(features, target)

pesos = np.column_stack([nn.pesos0.T,nn.pesos1])

df = pd.DataFrame(data=pesos, columns="pesos(entrada1),pesos(entrada2),pesos(saida)".split(","))

df.to_csv("data/x_pesos.dat", index=False)

prev = nn.previsao(features)

plt.plot(target*scaler, label="Saida desejada (I.O)")
#plt.plot(x_model, label="Integracao do Modelo (RK4)")
plt.plot(prev*scaler,'--' ,label="Previsao da Rede")
plt.legend(loc="best")
plt.legend(loc="best")
plt.grid(linestyle="-.")

plt.show()

