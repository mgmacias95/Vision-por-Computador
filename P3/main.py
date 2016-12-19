#! /usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################################################################
# Práctica 3 - Visión por computador - Universidad de Granada
# Curso 2016/2017
# Alumna: Marta Gómez Macías
# Profesor: Nicolás Pérez de la Blanca Capilla
########################################################################################################################

import cv2
import sys
sys.path.append("/home/marta/Documentos/Git/Vision-por-Computador/")
from funciones import *

if __name__ == '__main__':
    ####################################################################################################################
    #                                           Ejercicio 1                                                            #
    ####################################################################################################################
    # print("Ejercicio 1")
    # P = genera_camara_finita()
    # p = genera_puntos_planos_ortogonales_distintos()
    # c = proyecta_puntos_en_plano(camara=P, puntos=p)
    # dlt = DLT(X=p, x=c)
    # print("Matriz cámara generada")
    # print(P)
    # print("Estimación de la matriz cámara")
    # print(dlt)
    # print("Error en la estimación de P: ", estima_error(orig=P, estimada=dlt))
    # draw_points(real_points=c, estimated_points=proyecta_puntos_en_plano(camara=dlt,puntos=p))
    ####################################################################################################################
    #                                           Ejercicio 2                                                            #
    ####################################################################################################################
    print("Ejercicio 2")
    imgpoints, objpoints, pic_shape, img_index = find_and_draw_chessboard_corners()
    mtx, dist = calibrate(objpoints=objpoints, imgpoints=imgpoints, pic_shape=pic_shape[::-1])
    print(mtx)
    calibrate_undistort(img_index=img_index, mtx=mtx, dist=dist, pic_shape=pic_shape[::-1])
    

