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
    img = cv2.imread('datos-T2/Tablero1.jpg', cv2.IMREAD_GRAYSCALE)
    harris = Harris(img=img)
    mostrar(harris)
