# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 11:09:07 2022

@author: asmaa essam soliman

"""

import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('C:/Users/asmaa/Desktop/project/haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread('C:/Users/asmaa/Desktop/project/lenna.png')

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the output
cv2.imshow('img', img)
cv2.waitKey()
