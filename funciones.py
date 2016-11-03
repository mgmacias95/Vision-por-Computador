#! /usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################################################################
# Visión por computador - Universidad de Granada
# Curso 2016/2017
# Alumna: Marta Gómez Macías
# Profesor: Nicolás Perez de la Blanca Capilla
########################################################################################################################

from math import floor, exp, ceil
from operator import itemgetter
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
def resize(img, scale, sigma=2):
    # Para hacer un buen redimensionado. Debemos primero aplicar un filtro gaussiano a la imagen y después quedarnos con
    # las filas/columnas %scale. Por ejemplo si scale=2, sólo nos quedaríamos con las pares (una sí, una no, ....).
    img_blur = my_filter2D(src=img, kernel=my_getGaussianKernel(sigma),borderType='replicate')
    # una vez desenfocada la imagen, creamos una nueva imagen para guardarla y nos quedamos con las filas/columnas %scale
    img_little = img_blur[range(0,img_blur.shape[0],scale)]
    img_little = img_little[:,range(0,img_blur.shape[1],scale)]

    return img_little

# Función para hacer un collage tipo pirámide gaussiana
def piramide_gaussiana(img, scale=5, sigma=2, return_canvas=True):
    # el tamaño del canvas debe ser ancho_img_original + 0.5*ancho_img_original x altura_img_original
    little = img
    if return_canvas:
        dims = img.shape
        if len(dims) == 3:
            piramide = np.ones((dims[0],dims[1]+floor(dims[1] * 0.5),3),np.uint8)*255
        else:
            piramide = np.ones((dims[0], dims[1] + floor(dims[1] * 0.5)),np.uint8) * 255

        # colocamos la primera imagen en tamaño original
        piramide[0:dims[0],0:dims[1]] = little
        # calculamos el lugar donde poner la segunda
        start_height = 0
        end_height = ceil(dims[0]/2)
        start_width = dims[1]   # este lugar será igual para todas las imágenes
        start_width -= 1
    else:
        pyramid_list = []
        pyramid_list.append(little)

    for i in range(2,scale+1):
        # calculamos la i-esima imagen
        little = resize(img=little, scale=2, sigma=sigma)
        # guardamos sus medidas
        dims = little.shape
        if return_canvas:
            # la colocamos en el sitio calculado
            piramide[start_height:end_height,start_width:start_width+dims[1]] = little
            # calculamos dónde colocar la siguiente imagen
            start_height = end_height
            end_height = ceil(dims[0]/2) + start_height
        else:
            pyramid_list.append(little)

    if return_canvas:
        return piramide
    else:
        return pyramid_list

# Función que implementa el criterio de Harris: det(M) - k * trace(M)
# este criterio también se puede expresar como: lamba1*lambda2 / (lambda1+lambda2)^2 = det(M)/trace(M)^2
criterio_harris = lambda lambda1, lambda2, k=0.04: lambda1*lambda2 - k*((lambda1+lambda2)**2)

# función que tomando como entrada los valores de un entorno nos indica si el valor del centro es máximo local.
# Estamos presuponiendo una ventana 2D con un número impar de dimensiones (3x3, 5x5, etc)
is_localmax_center = lambda entorno: np.argmax(entorno) == floor((entorno.shape[0]*entorno.shape[1])/2)

# función que pone en negro todos los píxeles de un entorno menos el central
def put_zero_least_center(img, window_size, i, j):
    img[i-window_size:i+window_size+1, j-window_size:j+window_size+1] = 0
    img[i,j] = 255

# función que dada una matriz de harris y un umbral, devuelve una imagen binaria donde los puntos blancos son los que
# superan dicho umbral
binary_harris = lambda matriz, umbral: (matriz >= umbral) * 255

def Harris(img, n_points = 1500, points_to_keep = [0.7, 0.2, 0.1], window_size = 1, umbral=0.00001, scale = 3):
    # hacemos una pirámide gaussiana con escala 3 de la imagen.
    lista_escalas = piramide_gaussiana(img=img, scale=scale, sigma=1, return_canvas=False)
    # y para cada escala, usamos la función de OpenCV "cornerEigenValsAndVecs" para extraer los mapas de
    # auto-valores de la matriz Harris en cada píxel. Debemos tener en cuenta que esta función devuelve
    # 6 matrices (lambda1, lambda2, x1, y1, x2, y2) donde:
    #       * lambda1, lambda2 son los autovalores de M no ordenados
    #       * x1, y1 son los autovectores de lambda1
    #       * x2, y2 son los autovectores de lambda2
    scale_eigenvalues = [] # lista de matrices en el que guardar resultados
    for escala in lista_escalas:
        scale_eigenvalues.append(cv2.cornerEigenValsAndVecs(src=escala, blockSize=3, ksize=3))

    # una vez tenemos los autovalores de cada imagen, creamos una matriz por cada escala con el criterio de harris
    matrices_harris = [] # lista de matrices para guardar las matrices del criterio de harris para cada escala
    for escala in scale_eigenvalues:
        canales = cv2.split(escala) # cornerEigenValsAndVecs devuelve una imagen con seis canales.
        matrices_harris.append(criterio_harris(lambda1 = canales[0], lambda2 = canales[1]))

    # inicializamos una lista de imágenes binarias a 255 si no supera un determinado umbral y a 255 si lo supera. Una por escala.
    binaria = [binary_harris(escala, umbral) for escala in matrices_harris]

    # una vez tenemos nuestra imagen binaria, la recorremos preguntando para cada posición con valor 255 si
    # su correspondiente valor en la matriz de harris es máximo local o no.
    for escala in range(scale):
        # nos quedamos con los índices que superan el umbral
        harris_index = np.where(binaria[escala] == 255) # where devuelve un vector con los índices fila y otro con las columnas
        # una vez tenemos esos indices, comprobamos si el valor de esa posición es máximo local o no
        for i in range(len(harris_index[0])):
            row = harris_index[0][i]
            col = harris_index[1][i]
            # comprobamos si el pixel row,col de la imagen binaria es máximo local
            if row >= window_size and col >= window_size and is_localmax_center(matrices_harris[escala]\
                        [row-window_size:row+window_size+1,col-window_size:col+window_size+1]):
                # si es máximo local, ponemos en negro a todos los píxeles de su entorno
                put_zero_least_center(binaria[escala], window_size, row, col)
            else:
                # si no lo es, ponemos el píxel a 0
                binaria[escala][row,col] = 0

    # una vez tenemos los puntos de Harris eliminando no máximos, los ordenamos por su valor de Harris
    best_harris = []
    for escala in range(scale):
        # nos quedamos con los índices que corresponden con puntos de harris
        harris_index = np.where(binaria[escala] == 255)
        # y también con el valor de harris de esos puntos
        harris_points = matrices_harris[escala][harris_index]
        # obtenemos los índices de los puntos del vector harris_points ordenados
        sorted_indexes = np.argsort(harris_points)
        # juntamos en una matriz con dos columnas las coordenadas x,y de los puntos y nos quedamos con los
        # points_to_keep[escala]*n_points primeros
        best_harris.append(np.vstack(harris_points).T[sorted_indexes[0:points_to_keep[escala]*n_points]])


    return binaria
