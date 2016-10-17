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
    img = cv2.imread('data/motorcycle.bmp', cv2.IMREAD_UNCHANGED)
    # calculamos la máscara que le vamos a aplicar. Por ejemplo, con tamaño 7 y sigma 3
    mascara = cv2.getGaussianKernel(ksize=7, sigma=1)
    my_mascara = my_getGaussianKernel(sigma=1)
    # aplicamos el filtro gaussiano a la imagen.
    final_image = cv2.filter2D(src=img,ddepth=-1,kernel=mascara,borderType=cv2.BORDER_REPLICATE)
    my_final_image = my_filter2D(src=img, kernel=my_mascara,borderType='replicate')
    # mostramos el filtro gaussiano en un collage
    cv2.imwrite('memoria/propio_sigma2.jpg',my_final_image)
    # cv2.imwrite('memoria/opencv_sigma2.jpg', final_image)
    # cv2.imwrite('memoria/opencv_sigma3.jpg', cv2.filter2D(src=img,ddepth=-1,kernel=cv2.getGaussianKernel(ksize=19, \
    #                                                                 sigma=3),borderType=cv2.BORDER_REPLICATE))
    # cv2.imwrite('memoria/propio_sigma3.jpg', my_filter2D(src=img,kernel=my_getGaussianKernel(sigma=3),borderType='replicate'))
    # cv2.imshow('image', make_collage([img,final_image,my_final_image], ["Original","OpenCV","Propia"]))
    cv2.imshow('image', make_collage([img, final_image, my_final_image]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # APARTADO B
    einsrilyn = hybrid('data/einstein.bmp','data/marilyn.bmp')
    cv2.imshow('imagen', einsrilyn)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # cv2.imwrite('memoria/einsrilyn.jpg', einsrilyn)
    birdplane = hybrid('data/plane.bmp','data/bird.bmp',space=400,sigma_alta=2)
    cv2.imshow('imagen', birdplane)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # cv2.imwrite('memoria/birdplane.jpg', birdplane)
    catdog = hybrid('data/cat.bmp','data/dog.bmp',space=350, sigma_alta=2.5, sigma_baja=5, blackwhite=True, collage=False)
    cv2.imshow('imagen',catdog)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # cv2.imwrite('memoria/catdog.jpg',catdog)

    # APARTADO C
    img_c = cv2.imread('data/cat.bmp',cv2.IMREAD_UNCHANGED)
    gatito = piramide_gaussiana(img_c,5)
    cv2.imshow('image',gatito)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # # cv2.imwrite('memoria/piramide_gato.jpg',gatito)
    piramide_catdog = piramide_gaussiana(catdog, 5)
    cv2.imshow('image', piramide_catdog)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # cv2.imwrite('memoria/piramide_catdog.jpg', piramide_catdog)
    piramide_einsrilyn = piramide_gaussiana(hybrid('data/einstein.bmp','data/marilyn.bmp',collage=False),5)
    cv2.imshow('image', piramide_einsrilyn)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # cv2.imwrite('memoria/piramide_einsrilyn.jpg', piramide_einsrilyn)
