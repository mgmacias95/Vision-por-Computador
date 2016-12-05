#! /usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################################################################
# Práctica 3 - Visión por computador - Universidad de Granada
# Curso 2016/2017
# Alumna: Marta Gómez Macías
# Profesor: Nicolás Pérez de la Blanca Capilla
########################################################################################################################

import cv2
from funciones import *

if __name__ == '__main__':
    P = genera_camara_finita()
    c1, c2 = genera_puntos_planos_ortogonales_distintos()
    c = proyecta_puntos_en_plano(camara=P, puntos1=c1, puntos2=c2)
    print(c)
