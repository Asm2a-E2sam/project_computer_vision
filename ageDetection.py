# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 11:35:34 2022

@author: asmaa essam soliman

"""
# %%

import cv2  


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

def initialize_caffe_models():
    
    ageProto = "C:/Users/asmaa/Desktop/project/age_deploy.prototxt"
    ageModel = "C:/Users/asmaa/Desktop/project/age_net.caffemodel"

    genderProto = "C:/Users/asmaa/Desktop/project/gender_deploy.prototxt"
    genderModel = "C:/Users/asmaa/Desktop/project/gender_net.caffemodel"

    age_net = cv2.dnn.readNetFromCaffe(ageModel, ageProto) 
    gender_net = cv2.dnn.readNetFromCaffe(genderModel, genderProto)
    
    return (age_net , gender_net)
def read_from_image(age_net , gender_net):
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.imread("C:/Users/asmaa/Desktop/project/lenna.png")
    face_cascade = cv2.CascadeClassifier('C:/Users/asmaa/Desktop/project/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if(len(faces)>0):
        print ("found {} faces ".format(str(len(faces))))
        
    for (x, y, w, h) in faces :
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_img = img[y:y+h , h:h+w].copy
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        gender_net.setInput(blob)
        genderPreds = gender_net.forward()
        gender = genderList[genderPreds[0].argmax()]
        print("Gender: " + gender)
       
        age_net.setInput(blob)
        agePreds = age_net.forward()
        age = ageList[agePreds[0].argmax()]
        print("Age Range: "+ age)
                
        overlay_text = "%s %s" (gender , age)
        cv2.putText(img , overlay_text , (x ,y), font, 0.5 ,(100, 100, 255), 2, cv2.LINE_AA )
        cv2.imshow("",img)
        cv2.waitKey(0)
        
        
if __name__ == "__main__":
    age_net, gender_net = initialize_caffe_models()
    read_from_image(age_net, gender_net)
    
    
    