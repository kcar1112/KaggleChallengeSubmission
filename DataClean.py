#Links used directly in the Creation of my work
#https://www.atmosera.com/blog/facial-recognition-with-cnns/
#https://www.tensorflow.org/tutorials/images/classification
#https://colab.research.google.com/drive/1C8tLA5__49o-_3vFk3paxPMfHOgYUctG?usp=sharing
#https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
#https://github.com/opencv/opencv/tree/master/data/haarcascades
#https://docs.python.org/3/library/pickle.html
#https://keras.io/api/




import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.models import Model
from keras.models import load_model
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image
import pathlib
import cv2
import pickle
path = pathlib.Path().absolute()


train = pd.read_csv('train.csv')
facePath = "%s\\haarcascade_frontalface_default.xml" % (path)
profileFacePath = "%s\\haarcascade_profileface.xml" % (path)
facefinder = cv2.CascadeClassifier(facePath)
profileFaceFinder = cv2.CascadeClassifier(profileFacePath)


count, width = train.shape
print(f"True count: {count}")

#assume 25% of test data is bad
good_count = np.floor(count * 0.75)
print(f"Good count: {good_count}")

i = 0
y = train.pop("Category")
x = train.pop("File Name")
x_train = []
x_test = []
y_train = []
y_test = []
sucess = 0
picx = 0
picy = 0
picw = 0 
pich = 0
count = 4977
while i < count:
    if (i % 200 == 0):
        print(f"Image: {i} / {count}")
    try:
        img_path = "%s\\test\\%s.jpg" % (path, i)
        #https://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format
        pil_image = Image.open(img_path)
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = facefinder.detectMultiScale(grayimg, 1.1, 8)
        bestSize = 0
        bestx = 0
        besty = 0
        bestw = 0
        besth = 0
        if (len(faces) == 0):
            faces = facefinder.detectMultiScale(grayimg, scaleFactor=1.1, minNeighbors=7, flags=cv2.CASCADE_SCALE_IMAGE)
        if (len(faces) == 0):
            faces = facefinder.detectMultiScale(grayimg, scaleFactor=1.1, minNeighbors=6, flags=cv2.CASCADE_SCALE_IMAGE)
        # if (len(faces) == 0):
        #     faces = facefinder.detectMultiScale(grayimg, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
        # if (len(faces) == 0):
        #     faces = facefinder.detectMultiScale(grayimg, scaleFactor=1.1, minNeighbors=4, flags=cv2.CASCADE_SCALE_IMAGE)
        if (len(faces) == 0):
            faces = profileFaceFinder.detectMultiScale(grayimg, scaleFactor=1.1, minNeighbors=7, flags=cv2.CASCADE_SCALE_IMAGE)
        if (len(faces) == 0):
            faces = profileFaceFinder.detectMultiScale(grayimg, scaleFactor=1.1, minNeighbors=6, flags=cv2.CASCADE_SCALE_IMAGE)
        for(picx, picy, picw, pich) in faces:
            size = picw * pich
            if(size > bestSize):
                bestx = picx
                besty = picy
                bestw = picw
                besth = pich
                bestSize = size
        cropped = img[besty:besty+besth, bestx:bestx+bestw]
        cropped = cv2.resize(cropped, (128, 128))
        #cv2.imshow("Image", cropped)
        cv2.imwrite("%s\\test_imgs_crop_PIL\\%s.jpg" % (path, i), cropped) 
        newx = np.asarray(cropped)
        x_train.append(newx)
        y_train.append(y[i])
        sucess = sucess + 1
        i = i + 1
    except:
        x_train.append(newx)
        #print("Fail")
        i = i + 1
    

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print(x_train.shape[0], "train samples")

with open('test_ids_strict', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(y_train, f, protocol=-1)

with open('test_imgs_strict', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(x_train, f, protocol=-1)
