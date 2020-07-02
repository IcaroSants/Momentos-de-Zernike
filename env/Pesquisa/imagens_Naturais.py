from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold,cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from Descritor import Tratamento
import numpy as np
import os
from scipy.stats import zscore

#caminho para as imagens
diretorio=os.getcwd()
path1=os.path.join(diretorio,'airplane')
path2=os.path.join(diretorio,'car')
path3=os.path.join(diretorio,'cat')
path4=os.path.join(diretorio,'dog')
path5=os.path.join(diretorio,'flower')
path6=os.path.join(diretorio,'fruit')
path7=os.path.join(diretorio,'motorbike')
path8=os.path.join(diretorio,'person')

#representaçao das classes
aviao=np.array([1])
carro=np.array([2])
gato=np.array([3])
cao=np.array([4])
flor=np.array([5])
fruta=np.array([6])
moto=np.array([7])
pessoa=np.array([8])


cam=[path1,path2,path3,path4,path5,path6,path7,path8]
cla=[aviao,carro,gato,cao,flor,fruta,moto,pessoa]
col=Tratamento(cla,cam)

#construçao das matrizes de caracteristicas( mom_zernique,mom_hu e mom_me)
conjunto=col.constroir_matriz_tratada()
treino_dados_zer=conjunto[:,:25]
treino_rot_zer=conjunto[:,-1]

treino_dados_zer=zscore(treino_dados_zer)

#treinamento
print("momentos zernique")

'''mlp_zer = MLPClassifier(hidden_layer_sizes=(13,13), solver='adam', max_iter=700, activation='tanh',
                            learning_rate_init=0.003)
knn_zer = KNeighborsClassifier(7,metric="euclidean")'''
svm_zer = SVC(kernel='poly', degree=15, gamma='scale',C=10)



kfold=KFold(8,shuffle=True)

'''res_mlp=cross_validate(mlp_zer,treino_dados_zer,treino_rot_zer,
                       cv=kfold.split(treino_dados_zer,treino_rot_zer),
                        return_train_score=True)

res_knn=cross_validate(knn_zer,treino_dados_zer,treino_rot_zer,
                       cv=kfold.split(treino_dados_zer,treino_rot_zer),
                        return_train_score=True)'''

res_svm=cross_validate(svm_zer,treino_dados_zer,treino_rot_zer,
                       cv=kfold.split(treino_dados_zer,treino_rot_zer),
                        return_train_score=True)



'''print("Resultados K-nn:")
for knn_info,knn_obt in res_knn.items():
    print("   {}:{:.2f}".format(knn_info,np.mean(knn_obt)))

print()
print("Resultados mlp:")
for mlp_info,mlp_obt, in res_mlp.items():
    print("   {}:{:.2f}".format(mlp_info, np.mean(mlp_obt)))'''

print()
print("Resultados SVM:")
for  svm_info, svm_obt in  res_svm.items():
    print("   {}:{:.2f}".format(svm_info, np.mean(svm_obt)))

    







