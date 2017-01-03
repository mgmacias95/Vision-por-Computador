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
    # print("Ejercicio 3")
    # # leemos las imágenes
    # vmort1 = cv2.imread("vmort/Vmort1.pgm")
    # vmort2 = cv2.imread("vmort/Vmort2.pgm")
    # list_matches = make_descriptors(vmort1=vmort1, vmort2=vmort2)
    # maxmins = compare_descriptors(list_matches = list_matches)
    # print("Mejor descriptor: ",maxmins)
    # matches, kps1, kps2 = get_match(img1=vmort1, img2=vmort2, type=maxmins,knn_matching=False,mostrar_img=False)
    # F, pts1, pts2 = find_fundamental_matrix(matches=matches, kps1=kps1, kps2=kps2)
    # lines1 = find_and_draw_epipolar_lines(img=vmort1, pts=pts1, pts_other=pts2, F=F, index=2)
    # mostrar(vmort1)
    # lines2 = find_and_draw_epipolar_lines(img=vmort2, pts=pts2, pts_other=pts1, F=F, index=1)
    # mostrar(vmort2)
    # error_epipolar = epipolar_symmetric_error(points1=pts1, lines1=lines1, points2=pts2, lines2=lines2)
    # print("Error al estimar F:", error_epipolar)
    ####################################################################################################################
    #                                           Ejercicio 4                                                            #
    ####################################################################################################################
    print("Ejercicio 4")
    # la cámara es igual en todos los ficheros, por lo que sólo basta con leerla una vez
    camera = read_camera()
    print("Matriz cámara:")
    print(camera)
    # leemos las imágenes y los parámetros extrínsecos
    path = "reconstruccion/rdimage."
    photos = [path+"000.ppm",path+"001.ppm",path+"004.ppm"]
    imgs = []
    rotations = []
    translations = []
    for photo in photos:
        img, rot, tra = read_images_and_calibration_parameters(img=photo, calib_file=photo+".camera")
        imgs.append(img)
        rotations.append(rot)
        translations.append(tra)

    # una vez hemos leído los datos, calculamos las correspondencias entre las imágenes y la matriz fundamental
    matches, kps1, kps2 = get_match(img1=imgs[0], img2=imgs[1], type="BRISK", mostrar_img=False, knn_matching=False)
    F, pts1, pts2 = find_fundamental_matrix(matches=matches, kps1=kps1, kps2=kps2)
    # una vez tenemos calculada la matriz fundamental pasamos a estimar E
    E = my_find_essential_matrix(F=F, camera=camera)
    compute_r_and_t(E=E)
