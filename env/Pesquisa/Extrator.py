import cv2 as cv
import numpy as np
from mahotas import features as fea
from  mahotas import surf
from statistics import mean
from math import acos

class extrator(object):
    def __init__(self,imagem):
        self.imagem=cv.imread(imagem)
        self.imagemCinza=cv.imread(imagem,0)


    def nivel_Azul_Medio(self):

        azul,verde,vermelho=cv.split(self.imagem)
        media_azul=np.mean(azul)
        return media_azul

    def nivel_Vermelho_Medio(self):

        azul, verde, vermelho = cv.split(self.imagem)
        media_vermelho = np.mean(vermelho)
        return media_vermelho

    def nivel_Verde_Medio(self):

        azul, verde, vermelho = cv.split(self.imagem)
        media_verde = np.mean(verde)
        return media_verde

    def momentosHU(self):

        mom=cv.moments(self.imagemCinza)
        momHu=cv.HuMoments(mom)

        return momHu

    def Contorno(self):

        limite,imBin=cv.threshold(self.imagemCinza,
                                  cv.ADAPTIVE_THRESH_GAUSSIAN_C,255,cv.THRESH_BINARY +cv.THRESH_OTSU)

        contornos,hierarquia=cv.findContours(imBin,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)


        return contornos[(len(contornos)-1)]


    def Canny(self):
        return cv.Canny(self.imagem,100,200)


    def Distancia_Centroide(self):
        contorno=self.Contorno()

        #calculo do centroide
        mom=cv.moments(self.imagemCinza)

        cx = int(mom['m10'] / mom['m00'])
        cy = int(mom['m01'] / mom['m00'])

        #claculo da distacia de um ponto ao centroide da figura
        distancia=[]
        for ponto in contorno:
            valor=((ponto[0][0]-cx)**2 + (ponto[0][1]-cy)**2)**0.5
            distancia.append(valor)

        return distancia

    def Surf(self):
       su=surf.surf(descriptor_only=True)
       return su

    def momentsZernique(self):
        dis=self.Distancia_Centroide()
        raio=mean(dis)

        return fea.zernike_moments(self.imagemCinza,raio)

    def assinatura(self):
        distancia=self.Distancia_Centroide()

        indice=distancia.index(max(distancia))

        pontos=self.Contorno()

        #realocar os pontos em funcao da maior distancia ao centroide

        pontos1=pontos[indice:]
        aux=pontos[:indice]

        pontos1=np.vstack((pontos1,aux))

        # calculo do centroide
        mom = cv.moments(self.imagemCinza)

        cx = int(mom['m10'] / mom['m00'])
        cy = int(mom['m01'] / mom['m00'])

        # calculo do angulo usando produto vetorial
        vetor_Ref=[(pontos1[0][0][0]-cx),(pontos1[0][0][1]-cy)]
        mod_vetor_Ref=((vetor_Ref[0]**2)+(vetor_Ref[1]**2))**0.5

        pontos1=np.delete(pontos1,pontos1[0],0)
        distancia=[]
        angulos=[]
        for p in pontos1:

            vetor_atual = [(p[0][0] - cx), (p[0][1] - cy)]
            mod_vetor_atual = ((vetor_atual[0] ** 2) + (vetor_atual[1] ** 2)) ** 0.5
            x=((vetor_Ref[0]*vetor_atual[0]) + (vetor_Ref[1]*vetor_atual[1]))/(mod_vetor_atual*mod_vetor_Ref)
            if x > 1:
                x=1

            ang=acos(x)
            angulos.append(ang)
            distancia.append(mod_vetor_atual)

        return angulos,distancia

    def tratamento_imagem(self):
        imTratada=cv.adaptiveThreshold(self.imagemCinza,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,5)

        elementoEstruturante=cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
        imTratada=cv.erode(imTratada,elementoEstruturante,iterations=1)

        return imTratada

    def momentsZernique_tratado(self):
        im=self.tratamento_imagem()
        raio=max(im.shape)

        return fea.zernike_moments(im,raio)

    def momentsHu_tratada(self):
        mom=cv.moments(self.tratamento_imagem())
        hu=cv.HuMoments(mom)
        return hu

    def momEstatisticos_imagemTratado(self):
        mom=cv.moments(self.tratamento_imagem())
        return mom.values()