# -*- coding: cp1252 -*-
#Importar librerias
import cv2, os
import numpy as np
from PIL import Image

# Cargar clasificadores XML
rostro = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
completo = cv2.CascadeClassifier('haarcascade_fullbody.xml')
abajo = cv2.CascadeClassifier('haarcascade_lowerbody.xml')
arriba = cv2.CascadeClassifier('haarcascade_upperbody.xml')
perfil = cv2.CascadeClassifier('haarcascade_profileface.xml')
espalda = cv2.CascadeClassifier('HS.xml')
# Carga de Imágenes
img=cv2.imread('cliente1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#deteccion de rostro
faces = rostro.detectMultiScale(gray,1.2,3)
body = completo.detectMultiScale(gray,2.5,8)
lower = abajo.detectMultiScale(gray, 1.8,5)
upper = arriba.detectMultiScale(gray, 1.2,3)
profile = perfil.detectMultiScale(gray, 1.2,3)
shoulder = espalda.detectMultiScale(gray, 1.2,3)

#construir rectangulo
for (i,(x,y,w,h)) in enumerate(faces):
 cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
 cv2.putText(img, "Persona #{}".format(i + 1), (x, y - 10),
 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1,cv2.LINE_AA) 
for (x,y,w,h) in body:
 cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
for (x,y,w,h) in lower:
 cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
for (i,(x,y,w,h)) in enumerate(upper):
 cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
 cv2.putText(img, "Cuerpo #{}".format(i + 1), (x, y - 10),
 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1,cv2.LINE_AA)
for (i,(x,y,w,h)) in enumerate(profile):
 cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
 cv2.putText(img, "En perfil #{}".format(i + 1), (x, y - 10),
 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1,cv2.LINE_AA)
for (i,(x,y,w,h)) in enumerate(shoulder):
 cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
 cv2.putText(img, "Individuo #{}".format(i + 1), (x, y - 10),
 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1,cv2.LINE_AA)
cv2.imshow('Deteccion Grupal', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
