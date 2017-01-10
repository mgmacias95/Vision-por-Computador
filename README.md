# Vision-por-Computador
Prácticas de la asignatura Visión por Computador - Grado en Ingeniería Informática (UGR)

Prácticas hechas con __OpenCV__ y __Python 3__.

## Contenidos

* ___Práctica 1___: desarrollar una función de __filtro gaussiano__ (_Box filter_) desarrollando para ello la convolución de vectores 1D, añadir bordes a la imagen y la convolución 2D tomando como base la convolución 1D. Con el filtro gaussiano desarrollado, hacer __imágenes híbridas__ y __pirámides gaussianas__.

* ___Práctica 2___: desarrollar una función que calcule los  __puntos Harris__ de una imagen a varias junto con su orientación y que además, pinte en la imagen dichos puntos con un tamaño proporcional al de su escala y con su orientación. Después, __calcular correspondencias entre dos imágenes__ usando el detector _AKAZE_ y por último, __hacer mosaicos con n imágenes__, con la condición de que debemos dar las imágenes en orden al método.

* ___Práctica 3___: desarrollar una función que estime la __matriz cámara__ a partir de unos puntos 3D y sus proyecciones en 3D, para ello, generar una matriz cámara y unos puntos 3D aleatorios y sus correspondientes proyecciones 2D, implementar la estimación de la cámara usando el __algoritmo DLT__. Tras esto, desarrollar otra función que estime la matriz cámara a partir de homografías con imágenes _chessboard_. Después, desarrollar una función que estime la __matriz fundamental__ a partir de puntos en correspondencias de dos imágenes mediante el __algoritmo de los 8 puntos + RANSAC__. Por último, calcular el __movimiento de la cámara (R,t)__ asociado a cada pareja de imágenes calibradas, para ello, debemos calcular en primer lugar la __matriz esencial__.
