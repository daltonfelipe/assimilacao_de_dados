#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 13:33:16 2018

@author: dalton
"""
import numpy as np


class Perceptron():
    
    """
    Variaveis
    =========
    lr: learn rate (float, valores de 0.0 a 1.0)
        taxa de aprendizado.
    
    itmax: numero de epocas do treinamento
    
    w:  weight (array com o num de cols = num de linhas de x + bias w[0])
        pesos sinapticos.
        
    x: (array) dados de entrada.
    
    y: (array) dados de saida.
    
    y_pred: (raray) predicao para os valores do treinamento.
    
    signal: (funcao) funcao de ativacao retorna 0 se o produto
            entre w e x for <= 0 e 1 caso contrario.
    
    fit: (funcao) funcao de treinamento da rede.
    
    predict: (funcao) funcao para predicao apos o treinamento.
    
    
    """   
    
    def __init__(self, itmax=200, lr=0.01):
        """Função de ativação."""
        self.lr = lr
        self.itmax = itmax
        
    def signal(self, soma):
        
        return 0 if soma <= 0 else 1

    def fit(self, x, y):
        """ Funcao de treino da rede."""
        self.w = np.random.random((len(x[0])+1))
        self.y_pred = np.zeros((len(y)))
        
        for i in range(self.itmax):
            self.erro = np.zeros((len(x)))
            for i in range(len(x)):
                soma = x[i].dot(self.w[1:]) + self.w[0]
                self.y_pred[i] = self.signal(soma)            
                self.erro[i] = y[i] - self.y_pred[i]    
                #print(self.erro)
                self.w[1:] = self.w[1:] + self.lr*self.erro[i]*x[i]
                self.w[0] = self.w[0] + self.lr*self.erro[i]
            print(self.erro.mean())
            
    def predict(self, x):
        pred = np.zeros(len(x))
        for i in range(len(x)):
            pred[i] = self.signal(np.dot(x[i], self.w[1:]) + self.w[0]) 
        return pred 


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
sns.set()


iris = load_iris()

x = iris.data[:100]
y = iris.target[:100]

x_teste, x_treino, y_teste, y_treino = train_test_split(x, y, test_size=0.15)

nn = Perceptron(1000, 0.3)

nn.fit(x_treino, y_treino)

y_pred = nn.predict(x_teste)

accuracy = accuracy_score(y_teste, y_pred)

cm = confusion_matrix(y_teste, y_pred)

import matplotlib.pyplot as plt

plt.scatter(iris.data[:, 2], iris.data[:, 0], alpha=0.7)
plt.scatter(iris.data[:, 3], iris.data[:, 1], alpha=0.7)

plt.grid(linestyle="-")

mt = plt.matshow(cm)
plt.colorbar(mt)























