#! /usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################################################################
# Visión por computador - Universidad de Granada
# Curso 2016/2017
# Alumna: Marta Gómez Macías
# Profesor: Nicolás Perez de la Blanca Capilla
########################################################################################################################

from math import floor, exp, ceil
from random import sample
import numpy as np
import cv2
from operator import attrgetter

########################################################################################################################
#                                              PRÁCTICA 1                                                              #
########################################################################################################################
# función gaussiana para la máscara
kernel = lambda x, sigma: exp(-0.5 * ((x*x)/(sigma*sigma)))
# index_img_name = 18

def mostrar(imagen):
    # global index_img_name
    # cv2.imwrite(filename="memoria/img"+str(index_img_name)+'.jpg', img=imagen)
    # index_img_name+=1
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
def my_filter2D(src, kernel, borderType, ejex=True, ejey=True):
    # en primer lugar comprobamos si la imagen es en color o en blanco y negro
    if len(src.shape) == 3:
        # si es en color, debemos separar sus canales
        canales = cv2.split(src)
        # y aplicar sobre cada uno de ellos el filtro
        for i in range(len(canales)):
            canales[i] = my_filter2D_onechannel(src=canales[i], kernel=kernel, borderType=borderType, ejex=ejex, ejey=ejey)
        # una vez hecho esto, los volvemos a juntar con merge
        img = cv2.merge(canales)
    else:
        # si solo tiene un canal, aplicamos directamente el filtro
        img = my_filter2D_onechannel(src=src, kernel=kernel, borderType=borderType, ejex=ejex, ejey=ejey)
    return img

# Función para aplicar la máscara 1D a un canal de la imagen
def my_filter2D_onechannel(src, kernel, borderType, ejex, ejey):
    mitad_mascara = floor(kernel.size/2)
    # En primer lugar, añadimos bordes a la imagen
    img_bordes = my_copyMakeBorder(src=src, space=mitad_mascara, borderType=borderType).astype(np.float64)
    # img_bordes = cv2.copyMakeBorder(src=src, top=mitad_mascara, bottom=mitad_mascara, left=mitad_mascara,
    #                                 right=mitad_mascara, borderType=borderType)
    img_aux = np.ones(img_bordes.shape, np.float32)*255
    # Después, aplicamos el kernel a cada trocito
    if ejex:
        for j in range(mitad_mascara, img_bordes.shape[0]-mitad_mascara):
            for i in range(mitad_mascara,img_bordes.shape[1]-mitad_mascara):
                img_aux[j,i] = apply_kernel(img_bordes[j,i-mitad_mascara:i+1+mitad_mascara], kernel)
        img_bordes = img_aux.copy(order='F')
        img_aux = np.ones(img_bordes.shape, np.float32)*255

    if ejey:
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

########################################################################################################################
#                                              PRÁCTICA 2                                                              #
########################################################################################################################
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

# función que dibuja circulos en la imagen original
def draw_circle_on_corners(img, esquinas, scale, orientaciones = None, addOrientation=False):
    img_aux = img.copy()
    # En primer lugar creamos una matriz auxiliar en la que calcular las coordenadas de todos los puntos de todas las escalas.
    best_harris_coords_orig = []  # imagen para guardar las coordenadas de harris en escala original
    best_harris_coords_orig.append(np.array(esquinas[0], dtype=np.int64))
    for escala in range(1, scale):
        # pasamos las coordenadas de la escala escala a las de la imagen original
        best_harris_coords_orig.append(np.array(esquinas[escala] * (2 * escala), dtype=np.int64))

    # dibujamos círculos en la imagen original
    for escala in range(scale):
        for indices in best_harris_coords_orig[escala]:
            cv2.circle(img=img_aux, radius=(escala+1)*scale, center=(indices[1], indices[0]), \
                       color=1, thickness=-1)

    # si el flag de añadir orientacion está activado, pintamos un radio en el punto
    if addOrientation:
        for escala in range(scale):
            radio = (escala+1)*scale
            for i in range(len(orientaciones[escala])):
                punto = best_harris_coords_orig[escala][i]
                angle = (orientaciones[escala][i]*180)/np.pi
                cv2.line(img=img_aux, pt1=(punto[1], punto[0]), pt2 = (punto[1]+floor(np.sin(angle)*radio), \
                                                                   punto[0]+floor(np.cos(angle)*radio)), color=255)

    mostrar(img_aux)

def create_binary_harris(lista_escalas, umbral):
    # y para cada escala, usamos la función de OpenCV "cornerEigenValsAndVecs" para extraer los mapas de
    # auto-valores de la matriz Harris en cada píxel. Debemos tener en cuenta que esta función devuelve
    # 6 matrices (lambda1, lambda2, x1, y1, x2, y2) donde:
    #       * lambda1, lambda2 son los autovalores de M no ordenados
    #       * x1, y1 son los autovectores de lambda1
    #       * x2, y2 son los autovectores de lambda2
    scale_eigenvalues = []  # lista de matrices en el que guardar resultados
    for escala in lista_escalas:
        scale_eigenvalues.append(cv2.cornerEigenValsAndVecs(src=escala, blockSize=3, ksize=3))

    # una vez tenemos los autovalores de cada imagen, creamos una matriz por cada escala con el criterio de harris
    matrices_harris = []  # lista de matrices para guardar las matrices del criterio de harris para cada escala
    for escala in scale_eigenvalues:
        canales = cv2.split(escala)  # cornerEigenValsAndVecs devuelve una imagen con seis canales.
        matrices_harris.append(criterio_harris(lambda1=canales[0], lambda2=canales[1]))

    # inicializamos una lista de imágenes binarias a 255 si no supera un determinado umbral y a 255 si lo supera. Una por escala.
    binaria = [binary_harris(escala, umbral) for escala in matrices_harris]

    return binaria, matrices_harris

def remove_local_maxima(binaria, matrices_harris, window_size, scale):
    # una vez tenemos nuestra imagen binaria, la recorremos preguntando para cada posición con valor 255 si
    # su correspondiente valor en la matriz de harris es máximo local o no.
    for escala in range(scale):
        # nos quedamos con los índices que superan el umbral
        harris_index = np.where(
            binaria[escala] == 255)  # where devuelve un vector con los índices fila y otro con las columnas
        # una vez tenemos esos indices, comprobamos si el valor de esa posición es máximo local o no
        for i in range(len(harris_index[0])):
            row = harris_index[0][i]
            col = harris_index[1][i]
            # comprobamos si el pixel row,col de la imagen binaria es máximo local
            if row >= window_size and col >= window_size and is_localmax_center(matrices_harris[escala] \
                                                            [row - window_size:row + window_size + 1,
                                                            col - window_size:col + window_size + 1]):
                # si es máximo local, ponemos en negro a todos los píxeles de su entorno
                put_zero_least_center(binaria[escala], window_size, row, col)
            else:
                # si no lo es, ponemos el píxel a 0
                binaria[escala][row, col] = 0

def get_best_harris(matrices_harris, binaria, points_to_keep, n_points, scale):
    # una vez tenemos los puntos de Harris eliminando no máximos, los ordenamos por su valor de Harris para quedarnos con
    # los n_points mejores
    best_harris = []
    for escala in range(scale):
        # nos quedamos con los índices que corresponden con puntos de harris
        harris_index = np.where(binaria[escala] == 255)
        # y también con el valor de harris de esos puntos
        harris_points = matrices_harris[escala][harris_index]
        # obtenemos los índices de los puntos del vector harris_points ordenados
        sorted_indexes = np.argsort(harris_points)[::-1]
        # juntamos en una matriz con dos columnas las coordenadas x,y de los puntos y nos quedamos con los
        # points_to_keep[escala]*n_points primeros
        best_harris.append(np.vstack(harris_index).T[sorted_indexes[0:int(points_to_keep[escala] * n_points)]])
        binaria[escala][:] = 0
        binaria[escala][best_harris[escala][:, 0], best_harris[escala][:, 1]] = 255

    return best_harris

# apartado a) Calcular puntos de harris y pintarlos en la imagen original.
def Harris(lista_escalas,  umbral=0.00001, n_points = 1500, points_to_keep = [0.7, 0.2, 0.1], window_size = 1, scale = 3):
    img = lista_escalas[0]

    # Calculamos los puntos Harris que superan un determinado umbral.
    binaria, matrices_harris = create_binary_harris(lista_escalas=lista_escalas, umbral=umbral)

    mostrar(binaria[0])

    # supresión de no máximos
    remove_local_maxima(binaria=binaria, matrices_harris=matrices_harris, window_size=window_size, scale=scale)

    mostrar(binaria[0])

    # nos quedamos con los 1500 mejores puntos
    best_harris = get_best_harris(matrices_harris=matrices_harris, binaria=binaria, points_to_keep=points_to_keep, \
                  n_points=n_points, scale=scale)

    # una vez filtrados los mejores puntos de cada escala, los colocamos en la imagen original, dependiendo de la escala
    # tendrán un radio u otro.
    draw_circle_on_corners(img=img, esquinas=best_harris, scale=scale)

    return best_harris

# Función para refinar las esquinas sacadas en el apartado a con conrnerSubPix
def refina_Harris(escalas, esquinas, scale=3):
    ref_escalas = []
    for i in range(len(escalas)):
        float_esquinas = np.array(esquinas[i], dtype=np.float32)
        cv2.cornerSubPix(image=escalas[i], corners=float_esquinas, winSize=(5,5), zeroZone=(-1,-1), \
                        criteria=(cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_COUNT, 10, 0.01))
        ref_escalas.append(float_esquinas)

    draw_circle_on_corners(img=escalas[0], esquinas=ref_escalas,scale=scale)
    return ref_escalas

# Función para calcular la orientación de cada esquina encontrada.
def find_orientacion(escalas, esquinas, sigma=4.5):
    # calculamos las derivadas en x y en y aplicando un filtro sobel sobre la imagen. Una vez calculadas las derivadas,
    # calcular la arcotangente, que será la orientación del punto.
    orientaciones = []
    for i in range(len(escalas)):
        k = my_getGaussianKernel(sigma=sigma)
        esq_int = esquinas[i].T.astype(int)
        grad_x = my_filter2D(src=escalas[i], kernel=k, borderType='reflect', \
                             ejex=True, ejey=False)[esq_int[0], esq_int[1]]
        grad_y = my_filter2D(src=escalas[i], kernel=k, borderType='reflect', \
                             ejex=False, ejey=True)[esq_int[0], esq_int[1]]
        orientaciones.append(np.arctan2(grad_y, grad_x))

    draw_circle_on_corners(img=escalas[0], esquinas=esquinas, scale=3, orientaciones=orientaciones, addOrientation=True)

    return orientaciones

# Ejercicio 2
def knn_matching(bf, desc1, desc2, kps1, kps2, img1, img2, n, k=1):
    # como crossCheck es True, knnMatch con k = 1 nos devolverá las parejas (i,j) tales que el vecino más cercano de i
    # sea j y viceversa
    # (FUENTE: http://docs.opencv.org/2.4/modules/features2d/doc/common_interfaces_of_descriptor_matchers.html)
    matches = bf.knnMatch(desc1, desc2, k=k)

    # tomamos n aleatorios para dibujarlos
    indices = sample(range(len(matches)), n)

    matches_img = cv2.drawMatchesKnn(img1=img1, keypoints1=kps1, img2=img2, keypoints2=kps2, \
                                     matches1to2=[matches[i] for i in indices], outImg=None)
    mostrar(matches_img)

def normal_matching(bf, desc1, desc2, kps1, kps2, img1, img2, n, mostrar_img):
    matches = np.array(bf.match(desc1, desc2))

    if mostrar_img:
        # tomamos n aleatorios para dibujarlos
        indices = sample(range(len(matches)), n)

        # dibujamos los n primeros
        matches_img = cv2.drawMatches(img1=img1, keypoints1=kps1, img2=img2, keypoints2=kps2, \
                                         matches1to2=[matches[i] for i in indices], outImg=None)
        mostrar(matches_img)

    return matches

def get_match(img1, img2, mask=None, mostrar_img=True, knn_matching=True, n=50, type="AKAZE"):
    # pasamos las fotos a blanco y negro
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # inicializamos el descriptor del tipo que pasamos por parámetro
    if type=="AKAZE":
        detector = cv2.AKAZE_create()
    elif type=="BRISK":
        detector = cv2.BRISK_create()
    else: # type == "ORB"
        detector = cv2.ORB_create()

    # detectamos los keypoint y extraemos los descriptores de ambas fotos.
    kps1, descs1 = detector.detectAndCompute(image=gray1, mask=mask)
    kps2, descs2 = detector.detectAndCompute(image=gray2, mask=mask)

    if mostrar_img:
        # dibujamos los puntos claves en la imagen 1
        keypoints_img1 = cv2.drawKeypoints(image=gray1, keypoints=kps1, outImage=None, color=(0,0,255),
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        mostrar(keypoints_img1)
        # y en la imagen 2
        keypoints_img2 = cv2.drawKeypoints(image=gray2, keypoints=kps2, outImage=None, color=(255, 0, 0),
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        mostrar(keypoints_img2)

    # para hacer el matching, usamos la fuerza bruta y validación cruzada, NORM_L2 es la distancia euclídea.
    # (FUENTE: http://stackoverflow.com/questions/32849465/difference-between-cv2-norm-l2-and-cv2-norm-l1-in-opencv-python#32849908)
    bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)

    # hacemos el matching usando knn
    if mostrar_img and knn_matching:
        knn_matching(bf=bf, desc1=descs1, desc2=descs2, kps1=kps1, kps2=kps2, img1=img1, img2=img2, n=n)

    # hacemos el matching usando el método match
    matches = normal_matching(bf=bf, desc1=descs1, desc2=descs2, kps1=kps1, kps2=kps2, img1=img1, img2=img2,
                              n=n, mostrar_img=mostrar_img)

    return matches, kps1, kps2

# Ejercicio 3

# mosaico con dos imágenes
def mosaico_dos(img1, img2, epsilon=0.5, mostrar_img=True):
    matches, kps1, kps2 = get_match(img1=img1, img2=img2, mask=None, mostrar_img=False)
    puntos_dst = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    puntos_src = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    homografia, mascara = cv2.findHomography(srcPoints=puntos_src, dstPoints=puntos_dst, method=cv2.RANSAC,
                                            ransacReprojThreshold=epsilon)
    result = cv2.warpPerspective(src=img2, M=homografia, dsize=(img1.shape[1] + img2.shape[1], img2.shape[0]))
    result[0:img1.shape[0], 0:img1.shape[1]] = img1
    # eliminamos las partes negras sobrantes
    # result = np.delete(arr=result, obj=np.where(result == 0)[1], axis=1)
    result = np.delete(arr=result, obj=range(np.where(np.sum(a=result[0], axis=1)==0)[0][0]-10, img1.shape[1] + img2.shape[1]), axis=1)
    if mostrar_img:
        mostrar(result)

    return result

# mosaico con más de dos imágenes asumiendo que las pasamos al programa en orden
def mosaico_n(lista_imagenes):
    if len(lista_imagenes) == 3:
        # hacemos un mosaico con las dos primeras imagenes
        mosaico_12 = mosaico_dos(img1 = lista_imagenes[0], img2=lista_imagenes[1], mostrar_img=False)
        mosaico_23 = mosaico_dos(img1 = lista_imagenes[1], img2=lista_imagenes[2], mostrar_img=False)
        # y los unimos
        return mosaico_dos(img1=mosaico_12, img2=mosaico_23, mostrar_img=False)

    elif len(lista_imagenes) == 2:
        return mosaico_dos(img1=lista_imagenes[0], img2=lista_imagenes[1], mostrar_img=False)

    else:
        # nos quedamos con el índice de la imagen central
        central = floor(len(lista_imagenes)/2)
        # llamamos recursivamente a esta función con dos listas
        mosaico_primeramitad = mosaico_n(lista_imagenes[:central])
        mosaico_segundamitad = mosaico_n(lista_imagenes[central:])
        # y hacemos un mosaico con ambas
        mosaico = mosaico_dos(img1=mosaico_primeramitad, img2=mosaico_segundamitad, mostrar_img=False)

        mostrar(mosaico)

        return mosaico

########################################################################################################################
#                                              PRÁCTICA 3                                                              #
########################################################################################################################
# Ejercicio 1
# función para generar una cámara finita. Es decir, generar una matriz 3x4 cuyo determinante de las primeras tres
# columnas con las filas sea distinto de cero.
def genera_camara_finita():
    P = np.random.rand(3,4)
    while np.linalg.det(P[:3,:3]) == 0:
        P = np.random.rand(3,4)
    P=P/P[-1,-1]
    return P

# función para generar puntos del mundo 3D con coordenadas {(0, x1, x2) y (x2, x1, 0)}. Es decir, una rejilla de puntos
# en dos planos distintos ortogonales. x1=0.1:0.1:1 y x2=0.1:0.1:1 significa que tenemos que generar valores de x1 y x2
# desde 0.1 a 1 y que aumenten de 0.1 en 0.1.
def genera_puntos_planos_ortogonales_distintos():
    # posibles valores para x1 y x2
    x1 = x2 = np.arange(start=0.1,stop=1,step=0.1,dtype=np.float32)
    # generamos un conjunto de puntos con todas las combinaciones de (x1,x2)
    conjunto = np.concatenate(np.array(np.meshgrid(x1,x2)).T)
    # y le añadimos una columna de ceros al principio
    zeros_vector = np.zeros(conjunto.shape[0])
    conjunto1 = np.hstack((zeros_vector[..., None], conjunto))
    # y otra al final
    conjunto2 = np.hstack((conjunto, zeros_vector[...,None]))

    return np.concatenate((conjunto1, conjunto2))

# función que dado un punto del mundo calcula sus coordenadas de proyección de la cámara.
# Debemos añadirle al punto x un elemento 1 para poder multiplicarlo por la matriz cámara.
camera_projection = lambda x, P: P.dot(np.hstack((x,[1])))

# Proyectar el conjunto de puntos del mundo con la cámara simulada y obtener las coordenadas píxel de su proyección
def proyecta_puntos_en_plano(camara, puntos):
    # definimos el array en el que guardaremos las coordenadas píxel de los puntos
    conjunto = np.zeros(puntos.shape)
    # iteramos sobre el array de puntos del mundo para proyectar los puntos
    for i in range(puntos.shape[0]):
        conjunto[i] = camera_projection(x=puntos[i], P=camara)

    # calculamos las coordenadas píxel diviendo la coordenada x e y por la coordenada z
    coords_pixel = np.zeros((puntos.shape[0], 2))
    for i in range(puntos.shape[0]):
        z = conjunto[i,2]
        coords_pixel[i,0] = conjunto[i,0]/z
        coords_pixel[i,1] = conjunto[i,1]/z

    return coords_pixel

# Función para normalizar los puntos.
def norm_points(points):
    media = np.mean(a=points, axis=0)
    desv_std = np.std(a=points)
    dims = points.shape[1] # para saber si estamos en 3d o 2d
    if dims==2:
        Tr = np.array([[desv_std, 0, media[0]],[0, desv_std, media[1]],[0,0,1]])
    else:
        Tr = np.array([[desv_std, 0, 0, media[0]], [0, desv_std, 0, media[1]], [0, 0, desv_std, media[2]], [0,0,0,1]])

    Tr = np.linalg.inv(a=Tr)
    x = np.dot(Tr, np.concatenate((points.T, np.ones((1,points.shape[0])))))
    x = x[0:dims,:].T
    return Tr, x


# Implementación del algoritmo DLT basada en el libro Multiple View Geometry y
# http://www.maths.lth.se/matematiklth/personal/calle/datorseende13/notes/forelas3.pdf
# Entrada del algoritmo: Xi (punto del mundo) y xi (proyección del punto).
# Salida del algoritmo P (matriz 3x4 de la cámara)
def DLT(X, x):
    n = x.shape[0] # numero de puntos
    # M tendrá, para cada punto, 2 filas y 12 columnas. Sólo usamos 2 filas ya que las tres ecuaciones de la matriz M son
    # linealmente dependientes.
    M = np.zeros(shape=(2*n, 12))
    z = np.zeros(shape=(4))
    # normalizamos los puntos
    tr, xn = norm_points(x)
    Tr, Xn = norm_points(X)
    # calculamos la matriz M
    for i in range(0,2*n,2):
        j = int(i/2)
        M[i] = np.concatenate((Xn[j], [1], z, -xn[j,0]*Xn[j], [-xn[j,0]]))
        M[i+1] = np.concatenate((z, Xn[j], [1], -xn[j,1]*Xn[j], [-xn[j,1]]))
    # calculamos sus valores propios
    U,S,V = np.linalg.svd(a=M)
    # La última fila de V contiene el autovector con menor autovalor (S).
    P = (V[-1]/V[-1,-1]).reshape(3,4)
    # deshacemos la normalización
    P = np.dot(np.dot(np.linalg.pinv(tr), P), Tr)
    P = P/P[-1,-1]
    return P

# Estimación del error de la cámara estimada
def estima_error(orig, estimada):
    return np.linalg.norm(x=(orig - estimada), ord=None)

# Función para pintar los puntos proyectados por la cámara real y la estimada
def draw_points(real_points, estimated_points):
    # creamos una imagen vacía
    img = np.ones(shape=(200,100,3), dtype=np.uint8)
    rp = np.array(100*real_points, dtype=np.int32)
    ep = np.array(100*estimated_points, dtype=np.int32)
    # pintamos los distintos puntos
    for i in range(real_points.shape[0]):
        cv2.circle(img=img, radius=1, center=(rp[i,0], rp[i,1]), \
                   color=(255,0,0), thickness=-1)
        cv2.circle(img=img, radius=1, center=(ep[i, 0], ep[i, 1]), \
                   color=(0, 0, 255), thickness=-1)
    mostrar(img)

# Ejercicio 2
# Función que lee las imágenes chessboard de la carpeta path y calcula las esquinas
def find_valid_imgs(path="chessboard/Image", n_imgs=25, format=".tif", pat_size=(13,12)):
    valid_imgs = []
    for i in range(n_imgs):
        imgpath = path+str(i+1)+format
        img = cv2.imread(imgpath)
        gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        # encontrar los chess board corner
        retval, corners = cv2.findChessboardCorners(image=gray, patternSize=pat_size,
                                                    flags=(cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                           cv2.CALIB_CB_ADAPTIVE_THRESH
                                                           + cv2.CALIB_CB_FAST_CHECK))
        if retval:
            valid_imgs.append(img)

    return valid_imgs

def find_and_draw_chessboard_corners(valid_images, pat_size=(13,12)):
    imgpoints = [] # puntos 2D de la imagen
    objpoints = [] # puntos 3D del mundo real. Tomando como centro del mundo el tablero.
    objp = np.zeros((pat_size[0]*pat_size[1],3),np.float32)
    objp[:,:2] = np.mgrid[0:pat_size[0], 0:pat_size[1]].T.reshape(-1,2)
    objp = objp.reshape(-1,1,3)
    gray_shape = 0

    for img in valid_images:
        gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape

        # encontrar los chess board corner
        retval, corners = cv2.findChessboardCorners(image=gray, patternSize=pat_size,
                                                    flags=(cv2.CALIB_CB_NORMALIZE_IMAGE+
                                                           cv2.CALIB_CB_ADAPTIVE_THRESH
                                                           +cv2.CALIB_CB_FAST_CHECK))
        # si hemos encontrado, pasamos a refinarlos
        if retval:
            # cada llamada a esta función da un número de puntos entre 0 y patsize[1]*patsize[0]. Por tanto, nos
            # tendremos que quedar con los corners2.shape[0] primeros puntos del mundo objp
            corners2 = cv2.cornerSubPix(image=gray, corners=corners, winSize=(11,11), zeroZone=(-1,-1),
                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
            objpoints.append(objp)
            # mostramos los corner encontrados
            imgcorners = img.copy()
            imgdraw = cv2.drawChessboardCorners(image=imgcorners, patternSize=pat_size, corners=corners2,
                                            patternWasFound=retval)
            mostrar(imgdraw)

    return imgpoints, objpoints, gray_shape

# Función que calibra la cámara usando las esquinas encontradas
def calibrate(objpoints, imgpoints, pic_shape):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints=objpoints, imagePoints=imgpoints,
                                                imageSize=pic_shape, cameraMatrix=None, distCoeffs=None)
    print("Reprojection error: ",ret)
    print("Matriz de la cámara")
    print(mtx)
    print("Parámetros intrínsecos")
    print(dist)
    print("Parámetros extrínsecos")
    print("Rotación")
    print(rvecs)
    print("Traslación")
    print(tvecs)
    return mtx, dist

# Función que calibra la cámara eliminando la distorsión de la imagen original
def calibrate_undistort(valid_images, mtx, dist, pic_shape):
    valid_und_img = []
    # calculamos la camera matrix óptima
    newmtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix=mtx, distCoeffs=dist, imageSize=pic_shape,
                                                alpha=1)
    # leemos todas las imágenes de la lista img_index, calculamos la imagen sin distorsión
    for img in valid_images:
        dst = cv2.undistort(src=img, cameraMatrix=mtx, distCoeffs=dist, newCameraMatrix=newmtx)
        # recortamos la imagen
        # x,y,w,h = roi
        # dst = dst[y:y+h, x:x+w]
        valid_und_img.append(dst)
        mostrar(dst)
    
    return valid_und_img

# Ejercicio 3
# función que realiza los descriptores ORB, BRISK y AKAZE
def make_descriptors(vmort1, vmort2):
    matches_a, kps1_a, kps2_a = get_match(img1=vmort1, img2=vmort2, knn_matching=False, mostrar_img=False)
    matches_b, kps1_b, kps2_b = get_match(img1=vmort1, img2=vmort2, type="BRISK", knn_matching=False,
                                          mostrar_img=False)
    matches_o, kps1_o, kps2_o = get_match(img1=vmort1, img2=vmort2, type="ORB", knn_matching=False,
                                          mostrar_img=False)

    return [matches_a, matches_b, matches_o]


# función que compara los descriptores y nos dice el mejor
def compare_descriptors(list_matches):
    # para cada elemento de la lista de matches sacamos la mínima y la máxima distancia
    getdist = attrgetter('distance')
    maxmins = np.zeros((len(list_matches), 3))
    i = 0
    for matches in list_matches:
        l = list(map(getdist, matches))
        maxmins[i] = (max(l), min(l), len(l))
        i+=1

    print(maxmins)
    # el mejor será el que encuentre un mayor número de correspondencias con la distancia más pequeña
    best = np.where((maxmins[:,1] == min(maxmins[:,1])) | (maxmins[:,2] == max(maxmins[:,2])))[0][0]
    return ["AKAZE","BRISK","ORB"][best]

# función que implementa el algoritmo de los 8 puntos + RANSAC.
# Basado en http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html
def find_fundamental_matrix(matches, kps1, kps2):
    pts1 = []
    pts2 = []
    for m in matches:
        pts1.append(kps1[m.queryIdx].pt)
        pts2.append(kps2[m.trainIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(points1=pts1, points2=pts2)
    # seleccionamos solo inliers
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    print("Matriz fundamental")
    print(F)

    return F, pts1, pts2

# función para dibujar las líneas epipolares en las imágenes
def find_and_draw_epipolar_lines(img, pts, pts_other, F, index, n=200):
    # en primer lugar calculamos las líneas epipolares
    lines = cv2.computeCorrespondEpilines(points=pts_other.reshape(-1,1,2), whichImage=index, F=F)
    lines.reshape(-1,3)
    r,c = img.shape[:2]
    # tomamos n indices aleatorios para pintar
    indices = sample(range(len(lines)), n)
    for l,pt in zip(lines[indices], pts[indices]):
        l = l[0]
        # generamos un color aleatorio
        color = tuple(np.random.randint(0,255,3).tolist())
        x0, y0 = map(int, [0, -l[2]/l[1]])
        x1, y1 = map(int, [c, -(l[2]+l[0]*c)/l[1]])
        img = cv2.line(img=img, pt1=(x0,y0), pt2=(x1, y1), color=color, thickness=1)
        img = cv2.circle(img=img, center=tuple(pt), radius=5, color=color, thickness=-1)
    return lines

# función que calcula la distancia entre un punto y una línea
# fuente: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
def distance_point_line(point, line):
    x0, y0 = point
    a, b, c = line
    return np.abs(a*x0 + b*y0 + c)/np.sqrt(a*a + b*b)

# función que calcula las distancias al cuadrado entre los puntos y sus líneas epipolares y finalmente devuelve la media
def epipolar_distance_points_lines(points, lines):
    n = points.shape[0]
    distance = np.zeros(n, np.float32)
    for i in range(n):
        d = distance_point_line(point=points[i], line=lines[i][0])
        distance[i] = d
    return np.mean(distance)

# función que calcula el error epipolar simétrico
def epipolar_symmetric_error(points1, lines1, points2, lines2):
    # calculamos la media de las distancias de cada punto a su línea epipolar
    distance1 = epipolar_distance_points_lines(points=points1, lines=lines1)
    distance2 = epipolar_distance_points_lines(points=points2, lines=lines2)
    # aplicamos la fórmula
    return (distance1 + distance2)/2

# Ejercicio 4
def read_camera(file="reconstruccion/rdimage.000.ppm.camera"):
    # abrimos el fichero y leemos las tres primeras líneas
    camera = np.zeros(shape=(3,3), dtype=np.float32)
    with open(file) as f:
        for i in range(3):
            linea = np.array(f.readline().split(sep=" ")[:3], dtype=np.float32)
            camera[i] = linea
    return camera

def read_images_and_calibration_parameters(img, calib_file):
    # en primer lugar leemos la imagen y, después, los parámetros de distorsión radial, rotación y
    # traslación que tiene asociados
    img_m = cv2.imread(filename=img, flags=cv2.IMREAD_GRAYSCALE)
    dist_radial = np.zeros(shape=(3), dtype=np.float32)
    rotacion = np.zeros(shape=(3,3), dtype=np.float32)
    traslacion = np.zeros(shape=(3), dtype=np.float32)
    with open(calib_file) as f:
        # las tres primeras líneas del fichero corresponden a la matriz cámara, por tanto, no nos interesan
        lines = f.readlines()[3:]
    dist_radial = np.array(lines[0].split(sep=" ")[:3], dtype=np.float32)
    traslacion = np.array(lines[4].split(sep=" ")[:3], dtype=np.float32)
    for i in range(1,4):
        rotacion[i-1] = np.array(lines[i].split(sep=" ")[:3], dtype=np.float32)

    return img_m, dist_radial, rotacion, traslacion
