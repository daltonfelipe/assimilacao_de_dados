import numpy as np
import time

# rede neural multilayer perceptron (MLP) aplicada ao XOR

"""
    XOR (Ou Exclusivo)
    ==================
    0, 0 = 0
    0, 1 = 1
    1, 0 = 1
    1, 1 = 0
"""

# funcao de ativacao sigmoid
def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

def sigmoid_derivada(sig):
    return sig * (1 - sig)

# matriz de entrada xor
entradas = np.array([[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]])

# matriz de saida xor
saidas = np.array([[0],[1],[1],[0]])

# numero de neuronios camada oculta
nco = 3
# numero de neuronios de entrada
nce = len(entradas[0])
# numero de neuronios de saida
ncs = len(saidas[0])

pesos0 = 2*np.random.random((nce, nco)) - 1 # neuronios de entrada -> camada oculta
pesos1 = 2*np.random.random((nco, 1)) - 1 # camada oculta -> neuronio de saida

# numero de iteracoes de treinamento 
epocas = 50000

taxa_aprendizagem = 0.8
momento = 1.0

loop_time = np.empty(epocas)
init = time.time()

for j in range(epocas):
    # conta o tempo para cada loop
    l_init = time.time()
    # FOWARD
    # set da camada de entrada
    camada_entrada = entradas
    # funcao soma entre entradas e pesos
    soma_sinapse0 = np.dot(camada_entrada, pesos0)
    # ativacao da rede (neu. ent -> cam. oculta) -> [func. sigmoid]
    camada_oculta = sigmoid(soma_sinapse0)
    # funcao so,a
    soma_sinapse1 = np.dot(camada_oculta, pesos1)
    # ativacao da rede (neu. oculta -> neu. saida) -> [func. sigmoide]
    camada_saida = sigmoid(soma_sinapse1)
    # calculo do erro entre saida verdadeira e treinada
    erro_camada_saida = saidas - camada_saida
    # calculo da media abslouta dos erros
    media_absoluta = abs(erro_camada_saida.mean())
    print('Erro: '+str(media_absoluta))
    # BACKWARD (Backpropagation - ajuste dos pesos)
    # gradient descent (descida do gradiente - busca do melhor minimo local)
    derivada_saida = sigmoid_derivada(camada_saida)
    delta_saida = erro_camada_saida*derivada_saida
    pesos1_transposta = pesos1.T
    delta_saida_peso = delta_saida.dot(pesos1_transposta)
    delta_camada_oculta = delta_saida_peso*sigmoid_derivada(camada_oculta)
    camada_oculta_transposta = camada_oculta.T
    pesos_novos1 = camada_oculta_transposta.dot(delta_saida)
    # ajuste dos pesos que ligam camada oculta e neuronios de saida 
    pesos1 = (pesos1*momento) + (pesos_novos1*taxa_aprendizagem)
    camada_entrada_transposta = camada_entrada.T
    pesos_novos0 = camada_entrada_transposta.dot(delta_camada_oculta)
    # ajuste dos pesos que ligam neuronios de entrada e camada oculta 
    pesos0 = (pesos0*momento) + (pesos_novos0*taxa_aprendizagem)
    l_end = time.time()
    loop_time[j] = l_end - l_init
    
end = time.time()

print("\n======== Finalizado ========")
print('\nEpocas: {}'.format(epocas))
print('Tempo de execucao: ',end-init)
print('Tempo Medio por epoca: ',loop_time.mean())
