#!/usr/bin/env python

import os
import tensorflow as tf
from skimage.transform import resize
import cv2
import math
import matplotlib.pyplot as plt
import pandas as pd
import os
from keras.preprocessing import image
import numpy as np
from keras.utils import np_utils
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout

import settings
import function_list as ff
cg = settings.Experiment()
# Allow Dynamic memory allocation.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

main_path = os.path.join(cg.oct_main_dir,'Tom and jerry example')
training_frame_folder = os.path.join(main_path,'frames_training')
ff.make_folder([training_frame_folder])

# step 1: read the video, extract frames from it and save them as images
# videoFile = os.path.join(main_path,'Tom and jerry.mp4')
# cap = cv2.VideoCapture(videoFile)
# frameRate = cap.get(5)
# count = 0; x = 1
# while(cap.isOpened()):
#     frameId = cap.get(1) # current frame number
#     ret, frame = cap.read()
    
#     if (ret != True):
#         break
#     if (frameId % math.floor(frameRate) == 0):
#         filename = "frame%d.jpg" % count
#         count += 1
#         frame_save_path = os.path.join(training_frame_folder,filename)
#         cv2.imwrite(frame_save_path,frame)
# cap.release()
# print('Done extracting image!')

# step 2: label image for training model
data = pd.read_csv(os.path.join(main_path,'Label for each frame.csv'))
X = []
for img_name in data.Image_ID:
    img = plt.imread(os.path.join(training_frame_folder,img_name))
    X.append(img)
X = np.asarray(X)
# reshape image to fit the VGG neural network (224x224x3)
image = []
for i in range(0,X.shape[0]):
    a = resize(X[i],preserve_range = True, output_shape = (224,224)).astype(int)
    image.append(a)
X = np.asarray(image)
# preprocess
X = preprocess_input(X,mode = 'tf')
# prepare training
y = data.Class
one_hot_y = np_utils.to_categorical(y)
X_train,X_valid,y_train,y_valid = train_test_split(X,one_hot_y,test_size = 0.3,random_state = 42)
print(y_train.shape,y_valid.shape)


# step 3: build the model
base_model = VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))
X_train_VGG = base_model.predict(X_train)
X_valid_VGG = base_model.predict(X_valid)
X_train_VGG = X_train_VGG.reshape(X_train.shape[0],7*7*512)
X_valid_VGG = X_valid_VGG.reshape(X_valid.shape[0],7*7*512)
X_train_VGG_centered = X_train_VGG/X_train_VGG.max()
X_valid_VGG_centered = X_valid_VGG/X_train_VGG.max()

# build our own nueral netowrk
# build our own neural network
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout
# 1. build model
model = Sequential()
model.add(InputLayer((7*7*512,)))    # input layer
model.add(Dense(units=1024, activation='sigmoid'))   # hidden layer
model.add(Dropout(0.5))      # adding dropout, so there is sync on training and validation
model.add(Dense(units=512, activation='sigmoid'))    # hidden layer, increase number of layers to increase accuracy on training
model.add(Dropout(0.5))      # adding dropout
model.add(Dense(units=256, activation='sigmoid'))    # hidden layer
model.add(Dropout(0.5))      # adding dropout
model.add(Dense(3, activation='softmax'))            # output layer

# 2. compile the model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# 3. train
# there is inbalance of Tom and Jerry in the video, so we use class_weights to assign high weight to low value count
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
class_weights = compute_class_weight('balanced',np.unique(data.Class), data.Class)  # computing weights of different classes

# add callback
from keras.callbacks import ModelCheckpoint
filepath=os.path.join(main_path,'weights_best_for_Tom and Jerry.hdf5')
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]      # model check pointing based on validation loss

model.fit(X_train_VGG_centered,y_train,epochs = 150,validation_data=(X_valid_VGG_centered,y_valid),class_weight=class_weights,callbacks=callbacks_list)

# step 4: evaluate
test_frame_folder = os.path.join(main_path,'frames_test')
ff.make_folder([test_frame_folder])
videoFile = os.path.join(main_path,'Tom and Jerry 3.mp4')
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5)
count = 0
while(cap.isOpened()):
    frameId = cap.get(1)
    ret,frame = cap.read()
    if (ret!=True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename = 'test%d.jpg' % count 
        file_save_path = os.path.join(test_frame_folder,'test%d.jpg' % count)
        count += 1
        cv2.imwrite(file_save_path,frame)
cap.release()
print('Done extracting test')

test_set = pd.read_csv(os.path.join(main_path,'testing.csv'))
test_image = []
for img_name in test_set.Image_ID:
    img = plt.imread(os.path.join(test_frame_folder,img_name))
    test_image.append(img)
test_img = np.asarray(test_image)
# reshape
test_img_reshape = []
for i in range(0,test_img.shape[0]):
    a = resize(test_img[i],preserve_range=True,output_shape = (224,224)).astype(int)
    test_img_reshape.append(a)
test_img = np.asarray(test_img_reshape)
test_image = preprocess_input(test_img,mode='tf')
test_image = base_model.predict(test_image)
test_image = test_image.reshape(test_image.shape[0],7*7*512)
test_image = test_image / test_image.max()
test_y = np_utils.to_categorical(test_set.Class)

model.load_weights(os.path.join(main_path,'weights_best_for_Tom and Jerry.hdf5'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
scores = model.evaluate(test_image,test_y)
print('acc: %.2f%%' %(scores[1]*100))
