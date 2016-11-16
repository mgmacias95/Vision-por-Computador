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
    # # Ejercicio 1
    # img = cv2.imread('datos-T2/Tablero1.jpg', cv2.IMREAD_GRAYSCALE)
    # escalas = piramide_gaussiana(img=img, scale=3, sigma=1, return_canvas=False)
    # harris = Harris(lista_escalas=escalas, umbral=0.00001)
    # harris_ref = refina_Harris(escalas=escalas, esquinas=harris)
    # orientacion = find_orientacion(escalas=escalas, esquinas=harris_ref)
    #
    # img = cv2.imread('datos-T2/yosemite/Yosemite1.jpg', cv2.IMREAD_GRAYSCALE)
    # escalas = piramide_gaussiana(img=img, scale=3, sigma=1, return_canvas=False)
    # harris = Harris(lista_escalas=escalas, umbral=0.00001)
    # harris_ref = refina_Harris(escalas=escalas, esquinas=harris)
    # orientacion = find_orientacion(escalas=escalas, esquinas=harris_ref)

    # # Ejercicio 2
    # img1 = cv2.imread('datos-T2/yosemite/Yosemite1.jpg', cv2.IMREAD_UNCHANGED)
    # img2 = cv2.imread('datos-T2/yosemite/Yosemite2.jpg', cv2.IMREAD_UNCHANGED)
    # akaze_match(img1=img1, img2=img2)

    # Ejercicio 3
    img1 = cv2.imread('datos-T2/yosemite/Yosemite1.jpg', cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread('datos-T2/yosemite/Yosemite2.jpg', cv2.IMREAD_UNCHANGED)
    mosaico_dos(img1=img1, img2=img2)