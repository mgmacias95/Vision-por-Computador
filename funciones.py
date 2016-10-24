#! /usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################################################################
# Visión por computador - Universidad de Granada
# Curso 2016/2017
# Alumna: Marta Gómez Macías
# Profesor: Nicolás Perez de la Blanca Capilla
########################################################################################################################

from math import floor, exp, ceil
import numpy as np
import cv2

# función gaussiana para la máscara
kernel = lambda x, sigma: exp(-0.5 * ((x*x)/(sigma*sigma)))

def mostrar(imagen):
    cv2.imshow('image', imagen.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()

# Función para calcular la máscara/kernel
def my_getGaussianKernel(sigma):
    # El tamaño de la máscara depende de sigma. Aplicando la estadística báisca,
    # llegamos a que lo óptimo es tomar 3sigma por cada lado. Por lo que el
    # tamaño de la máscara será 6sigma + 1
    mascara = np.arange(-floor(3*sigma),floor(3*sigma + 1))
    # aplicamos la gaussiana a la máscara
    mascara = np.array([kernel(m, sigma) for m in mascara])
    # y la devolvemos normalizada
    return np.divide(mascara, np.sum(mascara))


# Función para darle a la imagen un borde negro ----  BORDER_CONSTANT: iiiiii | abcdefgh | iiiiiii
def black_border(src, space):
    # creamos una imagen negra
    if len(src.shape) == 3:
        img_borde = np.zeros((src.shape[0]+space*2, src.shape[1]+space*2,3), np.uint8)
    else:
        img_borde = np.zeros((src.shape[0] + space * 2, src.shape[1] + space * 2), np.uint8)
    dims = img_borde.shape
    # copiamos en el centro la imagen original
    img_borde[space:dims[0]-space, space:dims[1]-space] = src#.copy(order='F')
    return img_borde

# Función para darle a la imagen un borde ------- BORDER_REPLICATE: aaaaaa | abcdefgh | hhhhhhh
def replicate_border(src, space):
    # le añadimos un borde negro a la imagen
    img_borde = black_border(src,space)
    # cambiamos ese borde negro por una copia del último píxel. Primero por filas
    dims = src.shape
    for fila in range(dims[0]):
        img_borde[space+fila,0:space] = src[fila, 0]
        img_borde[space+fila,dims[1]+space:dims[1]+2*space] = src[fila,dims[1]-1]
    # después, por columnas
    for columna in range(dims[1]):
        img_borde[0:space,columna+space] = src[0,columna]
        img_borde[dims[0]+space:dims[0]+2*space,space+columna] = src[dims[0]-1,columna]
    return img_borde

# Función para reflejar la imagen ----- BORDER_REFLECT: fedcba | abcdefgh | hgfedcb
def reflect_border(src,space):
    # le añadimos un borde negro a la imagen
    img_borde = black_border(src,space)
    # cambiamos ese borde negro por copias de los space primeros píxeles de la imagen
    dims = src.shape
    for fila in range(dims[0]):
        to_copy_left = np.array(src[fila, 0:space])
        img_borde[space + fila, 0:space] = to_copy_left[::-1]
        to_copy_right = src[fila, dims[1]-space-1:dims[1] - 1]
        img_borde[space + fila, dims[1] + space:dims[1] + 2 * space] = to_copy_right[::-1]

    for columna in range(dims[1]):
        to_copy_left = np.array(src[0:space, columna])
        img_borde[0:space, columna + space] = to_copy_left[::-1]
        to_copy_right = np.array(src[dims[0]-space-1:dims[0] - 1, columna])
        img_borde[dims[0] + space:dims[0] + 2 * space, space + columna] = to_copy_right[::-1]
    return img_borde

# Función para reflejar la imagen -----  BORDER_REFLECT_101: gfedcb | abcdefgh| gfedcba
def reflect_101_border(src,space):
    # le añadimos un borde negro a la imagen
    img_borde = black_border(src,space)
    # cambiamos ese borde por los space+1 primeros píxeles de la imagen, sin contar el primero de todos
    dims = src.shape
    for fila in range(dims[0]):
        to_copy_left = np.array(src[fila, 1:space+1])
        img_borde[space + fila, 0:space] = to_copy_left[::-1]
        to_copy_right = src[fila, dims[1]-space-2:dims[1] - 2]
        img_borde[space + fila, dims[1] + space:dims[1] + 2 * space] = to_copy_right[::-1]

    for columna in range(dims[1]):
        to_copy_left = np.array(src[1:space+1, columna])
        img_borde[0:space, columna + space] = to_copy_left[::-1]
        to_copy_right = np.array(src[dims[0]-space-2:dims[0] - 2, columna])
        img_borde[dims[0] + space:dims[0] + 2 * space, space + columna] = to_copy_right[::-1]
    return img_borde

# Función que continua el borde que la imagen deja por el lado contrario ---  BORDER_WRAP: cdefgh | abcdefgh | abcdefg
def wrap_border(src,space):
    # le añadimos un borde negro a la imagen
    img_borde = black_border(src, space)
    # cambiamos ese borde negro por copias de los space primeros píxeles de la imagen
    dims = src.shape
    for fila in range(dims[0]):
        to_copy_left = np.array(src[fila, 0:space])
        to_copy_right = src[fila, dims[1] - space - 1:dims[1] - 1]
        img_borde[space + fila, 0:space] = to_copy_right[::-1]
        img_borde[space + fila, dims[1] + space:dims[1] + 2 * space] = to_copy_left[::-1]

    for columna in range(dims[1]):
        to_copy_left = np.array(src[0:space, columna])
        to_copy_right = np.array(src[dims[0] - space - 1:dims[0] - 1, columna])
        img_borde[0:space, columna + space] = to_copy_right[::-1]
        img_borde[dims[0] + space:dims[0] + 2 * space, space + columna] = to_copy_left[::-1]
    return img_borde

# Función para añadir borde a una imagen
def my_copyMakeBorder(src, space, borderType):
    return {
        'black':black_border(src, space),
        'replicate':replicate_border(src,space),
        'reflect':reflect_border(src,space),
        'reflect_101':reflect_101_border(src,space),
        'wrap':wrap_border(src,space)
    }.get(borderType)

# Función para aplicar el kernel a un trocito de imagen
apply_kernel = lambda original, kernel: np.sum(original * kernel)

# Función para aplicar la máscara 1D a una imagen con más de un canal
def my_filter2D(src, kernel, borderType):
    # en primer lugar comprobamos si la imagen es en color o en blanco y negro
    if len(src.shape) == 3:
        # si es en color, debemos separar sus canales
        canales = cv2.split(src)
        # y aplicar sobre cada uno de ellos el filtro
        for i in range(len(canales)):
            canales[i] = my_filter2D_onechannel(src=canales[i], kernel=kernel, borderType=borderType)
        # una vez hecho esto, los volvemos a juntar con merge
        img = cv2.merge(canales)
    else:
        # si solo tiene un canal, aplicamos directamente el filtro
        img = my_filter2D_onechannel(src=src, kernel=kernel, borderType=borderType)
    return img

# Función para aplicar la máscara 1D a un canal de la imagen
def my_filter2D_onechannel(src, kernel, borderType):
    mitad_mascara = floor(kernel.size/2)
    # En primer lugar, añadimos bordes a la imagen
    img_bordes = my_copyMakeBorder(src=src, space=mitad_mascara, borderType=borderType).astype(np.float64)
    # img_bordes = cv2.copyMakeBorder(src=src, top=mitad_mascara, bottom=mitad_mascara, left=mitad_mascara,
    #                                 right=mitad_mascara, borderType=borderType)
    img_aux = np.ones(img_bordes.shape, np.float64)*255
    # Después, aplicamos el kernel a cada trocito
    for j in range(mitad_mascara, img_bordes.shape[0]-mitad_mascara):
        for i in range(mitad_mascara,img_bordes.shape[1]-mitad_mascara):
            img_aux[j,i] = apply_kernel(img_bordes[j,i-mitad_mascara:i+1+mitad_mascara], kernel)
    img_bordes = img_aux.copy(order='F')
    img_aux = np.ones(img_bordes.shape, np.uint8)*255
    # Después, aplicamos el kernel a cada trocito
    for j in range(mitad_mascara, img_bordes.shape[1]-mitad_mascara):
        for i in range(mitad_mascara,img_bordes.shape[0]-mitad_mascara):
            img_aux[i,j] = apply_kernel(img_bordes[i-mitad_mascara:i+1+mitad_mascara,j], kernel)
    img_bordes = img_aux.copy(order='F')
    # Devolvemos la imagen con el filtro aplicado
    return img_bordes[mitad_mascara:-mitad_mascara, mitad_mascara:-mitad_mascara]

# def make_collage(lista_imagenes, lista_texto, space=450):
def make_collage(lista_imagenes):
    # inicializamos una matriz de 255s con el tamaño deseado
    if len(lista_imagenes[0].shape) == 3:
        # collage = np.ones((lista_imagenes[0].shape[0]+100,lista_imagenes[0].shape[1]*3,3), np.uint8)*255
        collage = np.ones((lista_imagenes[0].shape[0], lista_imagenes[0].shape[1] * 3, 3), np.uint8) * 255
    else:
        # collage = np.ones((lista_imagenes[0].shape[0]+100,lista_imagenes[0].shape[1]*3), np.uint8)*255
        collage = np.ones((lista_imagenes[0].shape[0], lista_imagenes[0].shape[1] * 3), np.uint8) * 255

    dims = lista_imagenes[0].shape
    for i in range(len(lista_imagenes)):
        collage[0:dims[0], i*dims[1]:dims[1]*(1+i)] = lista_imagenes[i]#.copy(order='F')
        # cv2.putText(img=collage, text=lista_texto[i], org=(25+(space*i),dims[0]+70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #        fontScale=3, color=0)

    return collage

# def hybrid(img_alta, img_baja, space=210, sigma_alta=1.5, sigma_baja=4, blackwhite = False, collage = True):
def hybrid(img_alta, img_baja, sigma_alta=1.5, sigma_baja=4, blackwhite=False, collage=True):
    # obtenemos las máscara respectivas para cada imagen
    my_mascara_alto = my_getGaussianKernel(sigma=sigma_alta)
    my_mascara_bajo = my_getGaussianKernel(sigma=sigma_baja)
    # leemos las dos imágenes que vamos a mezclar
    if blackwhite:
        img = cv2.imread(img_alta, cv2.IMREAD_GRAYSCALE).astype(np.float64)
        img2 = cv2.imread(img_baja, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    else:
        img = cv2.imread(img_alta, cv2.IMREAD_UNCHANGED).astype(np.float64)
        img2 = cv2.imread(img_baja, cv2.IMREAD_UNCHANGED).astype(np.float64)
    # para quedarnos con las frecuencias altas de la imagen, restamos las frecuencias bajas que obtenemos con el
    # filtro gaussiano a la imagen original
    paso_alto = img - my_filter2D(src=img, kernel=my_mascara_alto, borderType='replicate')
    paso_bajo = my_filter2D(src=img2, kernel=my_mascara_bajo, borderType='replicate')
    # para obtener la imagen híbrida, sumamos las dos imágenes.
    if collage:
        # return make_collage([paso_bajo,paso_alto,paso_alto+paso_bajo],["Low","High","Both"],space)
        return make_collage([paso_bajo, paso_alto, paso_alto + paso_bajo])
    else:
        return paso_alto+paso_bajo

# Función para redimensionar una imagen 1/scala de su tamaño.
def resize(img, scale):
    # Para hacer un buen redimensionado. Debemos primero aplicar un filtro gaussiano a la imagen y después quedarnos con
    # las filas/columnas %scale. Por ejemplo si scale=2, sólo nos quedaríamos con las pares (una sí, una no, ....).
    img_blur = my_filter2D(src=img, kernel=my_getGaussianKernel(scale/2),borderType='replicate')
    # una vez desenfocada la imagen, creamos una nueva imagen para guardarla y nos quedamos con las filas/columnas %scale
    img_little = img_blur[range(0,img_blur.shape[0],scale)]
    img_little = img_little[:,range(0,img_blur.shape[1],scale)]

    return img_little

# Función para hacer un collage tipo pirámide gaussiana
def piramide_gaussiana(img, scale=5):
    # el tamaño del canvas debe ser ancho_img_original + 0.5*ancho_img_original x altura_img_original
    dims = img.shape
    if len(dims) == 3:
        piramide = np.ones((dims[0],dims[1]+floor(dims[1] * 0.5),3),np.uint8)*255
    else:
        piramide = np.ones((dims[0], dims[1] + floor(dims[1] * 0.5)),np.uint8) * 255

    # colocamos la primera imagen en tamaño original
    piramide[0:dims[0],0:dims[1]] = img
    # calculamos el lugar donde poner la segunda
    start_height = 0
    end_height = ceil(dims[0]/2)
    start_width = dims[1]   # este lugar será igual para todas las imágenes
    start_width -= 1
    little = img
    for i in range(2,scale+1):
        # calculamos la i-esima imagen
        little = resize(img=little, scale=2)
        # guardamos sus medidas
        dims = little.shape
        # la colocamos en el sitio calculado
        piramide[start_height:end_height,start_width:start_width+dims[1]] = little
        # calculamos dónde colocar la siguiente imagen
        start_height = end_height
        end_height = ceil(dims[0]/2) + start_height

    return piramide