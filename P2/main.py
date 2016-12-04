#! /usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################################################################
# Práctica 2 - Visión por computador - Universidad de Granada
# Curso 2016/2017
# Alumna: Marta Gómez Macías
# Profesor: Nicolás Pérez de la Blanca Capilla
########################################################################################################################

import cv2
from funciones import *

if __name__ == '__main__':
    # Ejercicio 1
    img = cv2.imread('datos-T2/Tablero1.jpg', cv2.IMREAD_GRAYSCALE)
    escalas = piramide_gaussiana(img=img, scale=3, sigma=1, return_canvas=False)
    harris = Harris(lista_escalas=escalas, umbral=0.00001)
    harris_ref = refina_Harris(escalas=escalas, esquinas=harris)
    orientacion = find_orientacion(escalas=escalas, esquinas=harris_ref)

    img = cv2.imread('datos-T2/yosemite/Yosemite1.jpg', cv2.IMREAD_GRAYSCALE)
    escalas = piramide_gaussiana(img=img, scale=3, sigma=1, return_canvas=False)
    harris = Harris(lista_escalas=escalas, umbral=0.00001)
    harris_ref = refina_Harris(escalas=escalas, esquinas=harris)
    orientacion = find_orientacion(escalas=escalas, esquinas=harris_ref)

    # Ejercicio 2
    img1 = cv2.imread('datos-T2/yosemite/Yosemite1.jpg', cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread('datos-T2/yosemite/Yosemite2.jpg', cv2.IMREAD_UNCHANGED)
    akaze_match(img1=img1, img2=img2)

    # Ejercicio 3 - dos imágenes
    img1 = cv2.imread('datos-T2/yosemite/Yosemite1.jpg', cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread('datos-T2/yosemite/Yosemite2.jpg', cv2.IMREAD_UNCHANGED)
    mosaico_dos(img1=img1, img2=img2)

    # Ejercicio 3 - n imágenes
    img1 = cv2.imread('datos-T2/mosaico-1/mosaico002.jpg', cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread('datos-T2/mosaico-1/mosaico003.jpg', cv2.IMREAD_UNCHANGED)
    img3 = cv2.imread('datos-T2/mosaico-1/mosaico004.jpg', cv2.IMREAD_UNCHANGED)
    img4 = cv2.imread('datos-T2/mosaico-1/mosaico005.jpg', cv2.IMREAD_UNCHANGED)
    img5 = cv2.imread('datos-T2/mosaico-1/mosaico006.jpg', cv2.IMREAD_UNCHANGED)
    img6 = cv2.imread('datos-T2/mosaico-1/mosaico007.jpg', cv2.IMREAD_UNCHANGED)
    img7 = cv2.imread('datos-T2/mosaico-1/mosaico008.jpg', cv2.IMREAD_UNCHANGED)
    img8 = cv2.imread('datos-T2/mosaico-1/mosaico009.jpg', cv2.IMREAD_UNCHANGED)
    img9 = cv2.imread('datos-T2/mosaico-1/mosaico010.jpg', cv2.IMREAD_UNCHANGED)
    img10 = cv2.imread('datos-T2/mosaico-1/mosaico011.jpg', cv2.IMREAD_UNCHANGED)
    mosaico_n(lista_imagenes=[img1,img2,img3,img4,img5,img6,img7,img8,img9,img10])

    img1 = cv2.imread('datos-T2/yosemite_full/yosemite1.jpg', cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread('datos-T2/yosemite_full/yosemite2.jpg', cv2.IMREAD_UNCHANGED)
    img3 = cv2.imread('datos-T2/yosemite_full/yosemite3.jpg', cv2.IMREAD_UNCHANGED)
    img4 = cv2.imread('datos-T2/yosemite_full/yosemite4.jpg', cv2.IMREAD_UNCHANGED)
    mosaico_n(lista_imagenes=[img1,img2,img3,img4])