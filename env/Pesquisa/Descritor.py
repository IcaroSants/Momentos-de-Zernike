from Extrator import extrator
import numpy as np
import os

class Tratamento(object):
    def __init__(self,classes,path):
        self.classe=np.array(classes)
        self.path=path

    def matriz(self,classe,path):
        arquivos=os.listdir(path)
        matriz=[]
        qtElementos=len(arquivos)
        for arq in arquivos:

            im=extrator(os.path.join(path,arq))
            dados=im.momentsZernique()

            dados=np.hstack((dados,classe))
            matriz.append(dados)

        matriz=np.array(matriz)
        qtColunas=25+ self.classe.shape[1]
        matriz=matriz.reshape((qtElementos,qtColunas))

        return matriz

    def constroir_matriz(self):
        matriz=[]
        for pasta,rotulo in zip(self.path,self.classe) :
            matriz.append(self.matriz(rotulo,pasta))

        dados=matriz[0]

        for elemento in range(1,len(matriz)):
            dados=np.vstack((dados,matriz[(elemento)]))


        #np.random.seed(42)
        #np.random.shuffle(dados)

        return dados

    def matriz_Zernique(self,porcentagem_treino):
        matriz=self.constroir_matriz()
        qtElementos=matriz.shape[0]

        tamanhoTreino=int(qtElementos*porcentagem_treino)

        treino_dados=matriz[:tamanhoTreino,:25]
        treino_rot=matriz[:tamanhoTreino,25:]

        validacao_dados=matriz[tamanhoTreino:,:25]
        validacao_rot=matriz[tamanhoTreino:,25:]

        return treino_dados,treino_rot,validacao_dados,validacao_rot

    def matriz_tratada(self,classe,path):
        arquivos=os.listdir(path)
        matriz=[]
        qtElementos=len(arquivos)
        for arq in arquivos:

            im=extrator(os.path.join(path,arq))
            dados=im.momentsZernique_tratado()

            dados=np.hstack((dados,classe))
            matriz.append(dados)

        matriz=np.array(matriz)
        qt_Colunas=25+self.classe.shape[1]
        matriz=matriz.reshape((qtElementos,qt_Colunas))

        return matriz

    def constroir_matriz_tratada(self):
        matriz=[]
        for pasta,rotulo in zip(self.path,self.classe) :
            matriz.append(self.matriz_tratada(rotulo,pasta))

        dados=matriz[0]

        for elemento in range(1,len(matriz)):
            dados=np.vstack((dados,matriz[(elemento)]))


        np.random.seed(42)
        np.random.shuffle(dados)

        return dados

    def matriz_Zernique_tratado(self,porcentagem_treino):
        matriz=self.constroir_matriz_tratada()
        qtElementos=matriz.shape[0]

        tamanhoTreino=int(qtElementos*porcentagem_treino)

        treino_dados=matriz[:tamanhoTreino,:25]
        treino_rot=matriz[:tamanhoTreino,25:]

        validacao_dados=matriz[tamanhoTreino:,:25]
        validacao_rot=matriz[tamanhoTreino:,25:]

        return treino_dados,treino_rot,validacao_dados,validacao_rot

    def matriz_hu(self,classe,path):
        arquivos=os.listdir(path)
        matriz=[]
        qtElementos=len(arquivos)
        for arq in arquivos:

            im=extrator(os.path.join(path,arq))
            dados=im.momentsHu_tratada()

            dados=np.hstack((dados.ravel(),classe))
            matriz.append(dados)

        matriz=np.array(matriz)
        qt_Colunas=7+self.classe.shape[1]
        matriz=matriz.reshape((qtElementos,qt_Colunas))

        return matriz

    def constroir_matriz_hu(self):
        matriz=[]
        for pasta,rotulo in zip(self.path,self.classe) :
            matriz.append(self.matriz_hu(rotulo,pasta))

        dados=matriz[0]

        for elemento in range(1,len(matriz)):
            dados=np.vstack((dados,matriz[(elemento)]))

        np.random.seed(42)
        np.random.shuffle(dados)

        return dados

    def organiza_matriz_hu(self,porcentagem_treino):
        matriz=self.constroir_matriz_hu()
        qtElementos=matriz.shape[0]

        tamanhoTreino=int(qtElementos*porcentagem_treino)

        treino_dados=matriz[:tamanhoTreino,:7]
        treino_rot=matriz[:tamanhoTreino,7:]

        validacao_dados=matriz[tamanhoTreino:,:7]
        validacao_rot=matriz[tamanhoTreino:,7:]

        return treino_dados,treino_rot,validacao_dados,validacao_rot

    def matriz_momentosEstatisticos(self,classe,path):
        arq=os.listdir(path)
        matriz=[]
        qt_samples=len(arq)
        for imagem in arq:
            valores=[]
            arq_im=os.path.join(path,imagem)
            dado=extrator(arq_im)
            for valor in dado.momEstatisticos_imagemTratado():
                valores.append(valor)

            amostra=np.hstack((valores,classe))
            matriz.append(amostra)

        matriz=np.array(matriz)
        qt_Colunas = 24 + self.classe.shape[1]
        matriz=np.reshape(matriz,(qt_samples,qt_Colunas))

        return matriz

    def constroi_matriz_momentosEstatisticos(self):
        matriz=[]

        for pasta,rotulos in zip(self.path,self.classe):
            matriz.append(self.matriz_momentosEstatisticos(rotulos,pasta))

        dados=matriz[0]

        for pos in range(1,len(matriz)):
            dados=np.vstack((dados,matriz[pos]))

        np.random.seed(42)
        np.random.shuffle(dados)

        return dados

    def organiza_matriz_momentosEstatisticos(self,porcentagem_treino):
        matriz=self.constroi_matriz_momentosEstatisticos()
        qtElementos=matriz.shape[0]

        tamanhoTreino=int(qtElementos*porcentagem_treino)

        treino_dados=matriz[:tamanhoTreino,:24]
        treino_rot=matriz[:tamanhoTreino,24:]

        validacao_dados=matriz[tamanhoTreino:,:24]
        validacao_rot=matriz[tamanhoTreino:,24:]

        return treino_dados,treino_rot,validacao_dados,validacao_rot
