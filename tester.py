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
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Rescaling, Resizing, Dropout, LayerNormalization
from keras.applications import ResNet50
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image
import pathlib
import cv2
import pickle



path = pathlib.Path().absolute()
facePath = "%s\\haarcascade_frontalface_default.xml" % (path)
profileFacePath = "%s\\haarcascade_profileface.xml" % (path)
facefinder = cv2.CascadeClassifier(facePath)
profileFaceFinder = cv2.CascadeClassifier(profileFacePath)


path = pathlib.Path().absolute()


with open("y_train", "rb") as output:
    y_train = pickle.load(output)
with open("x_train", "rb") as output:
    x_train = pickle.load(output)



x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

categories = pd.read_csv('category.csv')
name_list = categories.pop("Category")
y_train_digits = []
for name in y_train:
    i = 0
    for check in name_list:
        if (name == check):
            y_train_digits.append(i)
        i = i + 1

y_train = keras.utils.to_categorical(y_train_digits, num_classes=100)



i = 0
trainSize = 50000
x_train_new = []
y_train_new = []
x_test = []
y_test = []
while i < len(y_train):
    if i < trainSize:
        x_train_new.append(x_train[i])
        y_train_new.append(y_train[i])
    else:
        x_test.append(x_train[i])
        y_test.append(y_train[i])
    i = i + 1

x_train = np.asarray(x_train_new)
y_train = np.asarray(y_train_new)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)





print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


shape = (128, 128, 3)
n_class = 100

#https://www.atmosera.com/blog/facial-recognition-with-cnns/
# base_model = VGGFace(model='resnet50', include_top=False)
# base_model.trainable = False
# model = Sequential()
# model.add(Resizing(224, 224))
# model.add(Rescaling(1./255, input_shape=(shape)))
# model.add(base_model)
# model.add(Flatten())
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(n_class, activation='softmax'))
# model.build()
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()
# model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=10, epochs=10)

model = Sequential()
model.add(Rescaling(1./255, input_shape=(shape)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dense(n_class, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=7, batch_size=10)

#Comment out to avoid retraining

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
model.save('FaceRecFaceNet4_1.h5')
print('Model Saved')


# model = load_model('FaceRecSimpleCNN128.h5')



numImages = 4977
i = 0
id = []
Category = []
while i < numImages:
    if (i % 100 == 0):
        print(f"Testing Image: {i}/{numImages}")
    try:
        img_path = "%s\\test_imgs_crop\\%s.jpg" % (path, i)
        img = keras.utils.load_img(img_path, target_size=(128, 128))
        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        guess = np.asarray(img_array)
        guess = model.predict(guess, verbose=0)
        guess = name_list[np.argmax(guess)]
    except:
        guess= ['NONE']
    Category.append(guess)
    id.append(i)
    i = i + 1
Category = pd.DataFrame(Category)
id = pd.DataFrame(id)
#array = pd.concat((id, Category), axis = 1)
df = pd.DataFrame(Category)
df.to_csv('submissionFaceNet.csv')