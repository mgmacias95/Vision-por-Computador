#! /usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################################################################
# Práctica 1 - Visión por computador - Universidad de Granada
# Curso 2016/2017
# Alumna: Marta Gómez Macías
# Profesor: Nicolás Pérez de la Blanca Capilla
########################################################################################################################

import cv2
import numpy as np
from funciones import *

if __name__ == '__main__':
    # APARTADO A
    # en primer lugar abrimos una imagen
    # img = cv2.imread('data/motorcycle.bmp', cv2.IMREAD_UNCHANGED)
    # calculamos la máscara que le vamos a aplicar. Por ejemplo, con tamaño 7 y sigma 3
    # mascara = cv2.getGaussianKernel(ksize=7, sigma=1)
    # my_mascara = my_getGaussianKernel(sigma=0.5)
    # aplicamos el filtro gaussiano a la imagen.
    # final_image = cv2.filter2D(src=img,ddepth=-1,kernel=mascara,borderType=cv2.BORDER_REPLICATE)
    # my_final_image = my_filter2D(src=img, kernel=my_mascara,borderType='replicate')
    # # mostramos el filtro gaussiano en un collage
    # cv2.imshow('image', make_collage([img,final_image,my_final_image], ["Original","OpenCV","Propia"]))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # APARTADO B
    my_mascara_alto = my_getGaussianKernel(sigma=1.25)
    my_mascara_bajo = my_getGaussianKernel(sigma=2)
    img = cv2.imread('data/marilyn.bmp', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('data/einstein.bmp', cv2.IMREAD_GRAYSCALE)
    paso_alto = img - my_filter2D(src=img, kernel = my_mascara_alto, borderType='replicate')
    paso_bajo = my_filter2D(src=img2, kernel=my_mascara_bajo, borderType='replicate')
    cv2.imshow('imagen', make_collage([paso_alto,paso_bajo,paso_alto+paso_bajo],["High", "Low", "Both"],210))
    cv2.waitKey()
    cv2.destroyAllWindows()
    # cv2.imshow('imagen', cv2.Laplacian(src=img,ddepth=-1))
    # cv2.waitKey()
    # cv2.destroyAllWindows()
