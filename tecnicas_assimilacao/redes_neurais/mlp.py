#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 17:28:29 2018

@author: dalton
"""

import numpy as np
import time

# Variaveis
# entradas = conjunto de dados de entrada
# 

# rede neural multilayer perceptron (MLP) aplicada ao XOR
class NN_MLP():
    
    def __init__(self,
                 epocas=1000,
                 erro=1e-5,
                 taxa_aprendizagem=0.1,
                 momento=1,
                 tamanho_camada_oculta=100):
        # numero de iteracoes de treinamento 
        self.epocas = epocas # padrao 1000
        # erro para segundo criterio de parada
        self.erro = erro # padrao 1e-5
        # variaveis para gradient descent
        self.taxa_aprendizagem = taxa_aprendizagem # padrao 0.1
        self.momento = momento # padrao 1
        # numero de neuronios na camada oculta
        self.nco = tamanho_camada_oculta # padrao 100
        
    # funcao de ativacao sigmoid
    def sigmoid(self, soma):
        return 1 / (1 + np.exp(-soma))
    
    # usada no calculo do gradient descent
    def sigmoid_derivada(self, sig):
        return sig * (1 - sig)
    
    # funcao de treino e ajuste dos pesos
    def treinar(self,entradas,saidas):
        loop_time = []
        init = time.time()
        # numero de neuronios de entrada
        self.nce = len(entradas[0])
        # numero de neuronios de saida
        self.ncs = len(saidas[0])
        pesos0 = 2*np.random.random((self.nce, self.nco)) - 1 # neuronios de entrada -> camada oculta
        pesos1 = 2*np.random.random((self.nco, 1)) - 1 # camada oculta -> neuronio de saida
        media_absoluta = 1
        i = 0
        while (i <= self.epocas):
            # para se o erro for menor que o determinado 
            if media_absoluta < self.erro:
                break
            # conta o tempo para cada loop
            l_init = time.time()
            # FOWARD
            # set da camada de entrada
            camada_entrada = entradas
            # funcao soma entre entradas e pesos
            soma_sinapse0 = np.dot(camada_entrada, pesos0)
            # ativacao da rede (neu. ent -> cam. oculta) -> [func. sigmoid]
            camada_oculta = self.sigmoid(soma_sinapse0)
            # funcao so,a
            soma_sinapse1 = np.dot(camada_oculta, pesos1)
            # ativacao da rede (neu. oculta -> neu. saida) -> [func. sigmoide]
            camada_saida = self.sigmoid(soma_sinapse1)
            # calculo do erro entre saida verdadeira e treinada
            erro_camada_saida = saidas - camada_saida
            # calculo da media abslouta dos erros
            media_absoluta = abs(erro_camada_saida.mean())
            # BACKWARD (Backpropagation - ajuste dos pesos)
            # gradient descent (descida do gradiente - busca do melhor minimo local)
            derivada_saida = self.sigmoid_derivada(camada_saida)
            delta_saida = erro_camada_saida*derivada_saida
            pesos1_transposta = pesos1.T
            delta_saida_peso = delta_saida.dot(pesos1_transposta)
            delta_camada_oculta = delta_saida_peso*self.sigmoid_derivada(camada_oculta)
            camada_oculta_transposta = camada_oculta.T
            pesos_novos1 = camada_oculta_transposta.dot(delta_saida)
            # ajuste dos pesos que ligam camada oculta e neuronios de saida 
            pesos1 = (pesos1*self.momento) + (pesos_novos1*self.taxa_aprendizagem)
            camada_entrada_transposta = camada_entrada.T
            pesos_novos0 = camada_entrada_transposta.dot(delta_camada_oculta)
            # ajuste dos pesos que ligam neuronios de entrada e camada oculta 
            pesos0 = (pesos0*self.momento) + (pesos_novos0*self.taxa_aprendizagem)
            l_end = time.time()
            print('Erro: '+str(media_absoluta))
            loop_time.append(l_end - l_init)
            i = i + 1

        end = time.time()
        self.pesos0 = pesos0
        self.pesos1 = pesos1
        self.time = end - init
        self.camada_oculta = camada_oculta
        self.camada_entrada = camada_entrada
        self.camada_saida = camada_saida
        self.erro_media_absoluta = media_absoluta
        print("\n======== Finalizado ========")
        print('\nEpocas: {}'.format(self.epocas))
        print('Tempo de execucao: ',end-init)
        print('Tempo Medio por epoca: ',np.mean(loop_time))
        print("Erro: {}".format(self.erro_media_absoluta))
        
    # faz a previsao com os pesos ajustados
    def previsao(self, entrada):
        camada_entrada = np.array(entrada)
        # funcao soma entre entradas e pesos
        soma_sinapse0 = np.dot(camada_entrada, self.pesos0)
        # ativacao da rede (neu. ent -> cam. oculta) -> [func. sigmoid]
        camada_oculta = self.sigmoid(soma_sinapse0)
        # funcao so,a
        soma_sinapse1 = np.dot(camada_oculta, self.pesos1)
        # ativacao da rede (neu. oculta -> neu. saida) -> [func. sigmoide]
        camada_saida = self.sigmoid(soma_sinapse1)
        # calculo do erro entre saida verdadeira e treinada
        erro_camada_saida = saidas - camada_saida
        # calculo da media abslouta dos erros
        self.erro_media_absoluta_previsao = abs(erro_camada_saida.mean())
        return camada_saida
        

# Teste
# matriz de entrada
entradas = np.array([[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]])

# matriz coluna de saida xor
saidas = np.array([[0],[1],[1],[0]])

# inicializacao da classe
nn = NN_MLP(epocas=15000,
            erro=1e-5,
            taxa_aprendizagem=0.9,
            tamanho_camada_oculta=200)

# treino da rede
nn.treinar(entradas, saidas)

# previsao  
print("\nPrevisao: ")
print(nn.previsao(entradas))
