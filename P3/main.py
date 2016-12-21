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
    # print("Ejercicio 2")
    # valid_imgs = find_valid_imgs()
    # imgpoints, objpoints, pic_shape = find_and_draw_chessboard_corners(valid_images=valid_imgs)
    # mtx, dist = calibrate(objpoints=objpoints, imgpoints=imgpoints, pic_shape=pic_shape[::-1])
    # valid_und_imgs = calibrate_undistort(valid_images=valid_imgs, mtx=mtx, dist=dist, pic_shape=pic_shape[::-1])
    # imgpoints_und, objpoints_und, pic_shape_und = find_and_draw_chessboard_corners(valid_images=valid_und_imgs)
    # mtx_und, dist_und = calibrate(objpoints=objpoints_und, imgpoints=imgpoints_und, pic_shape=pic_shape_und)
    ####################################################################################################################
    #                                           Ejercicio 3                                                            #
    ####################################################################################################################
    print("Ejercicio 3")
    # leemos las imágenes
    vmort1 = cv2.imread("vmort/Vmort1.pgm")
    vmort2 = cv2.imread("vmort/Vmort2.pgm")
    matches_a, kps1_a, kps2_a = get_match(img1=vmort1, img2=vmort2, knn_matching=False, mostrar_img=False)
    matches_b, kps1_b, kps2_b = get_match(img1=vmort1, img2=vmort2, type="BRISK", knn_matching=False,
                                          mostrar_img=False)
    matches_o, kps1_o, kps2_o = get_match(img1=vmort1, img2=vmort2, type="ORB", knn_matching=False,
                                          mostrar_img=False)
    maxmins = compare_descriptors(list_matches = [matches_a, matches_b, matches_o])
    print(maxmins)
    
