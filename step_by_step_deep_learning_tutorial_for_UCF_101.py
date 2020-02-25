#!/usr/bin/env python
import os
import tensorflow as tf
import cv2     
import math    
import pandas as pd
import numpy as np   
from skimage.transform import resize   
from glob import glob
from tqdm import tqdm
import keras
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split

import settings
import function_list as ff
cg = settings.Experiment()
# Allow Dynamic memory allocation.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

main_path = os.path.join(cg.oct_main_dir,'UCF101')

# step1: exploring the UCF101 dataset
txtpath = os.path.join(main_path,'UCF_TrainTestlist')
videopath = os.path.join(main_path,'UCF_Videos')
splittxt1 = os.path.join(txtpath,"subclass2_trainlist.txt")
splittxt2 = os.path.join(txtpath,"subclass2_testlist.txt")

f = open(os.path.join(splittxt1),'r')
temp = f.read()
videos= temp.split('\n')
train = pd.DataFrame()
train['video_name'] = videos; train = train[:-1]
 
f = open(os.path.join(splittxt2),'r')
temp = f.read()
videos= temp.split('\n')
test= pd.DataFrame()
test['video_name'] = videos; test = test[:-1]

# create tags for videos
train_video_tag = []
for i in range(train.shape[0]):
    train_video_tag.append(train['video_name'][i].split('/')[0])
train['tag'] = train_video_tag

test_video_tag = []
for i in range(test.shape[0]):
    test_video_tag.append(test['video_name'][i].split('/')[0])
test['tag'] = test_video_tag
test.head()

# extract time frames from videos
training_frame_folder = os.path.join(main_path,'training_frame_folder')
test_frame_folder = os.path.join(main_path,'test_frame_folder')
ff.make_folder([training_frame_folder,test_frame_folder])

# extract time frames from training and test videos
# for i in tqdm(range(train.shape[0])):
#     videoFile = train['video_name'][i]
#     cap = cv2.VideoCapture(os.path.join(videopath,videoFile.split(' ')[0].split('/')[0],videoFile.split(' ')[0].split('/')[1]))
#     frameRate = cap.get(5)
#     count = 0
#     while(cap.isOpened()):
#         frameId = cap.get(1)
#         ret,frame = cap.read()
#         if (ret != True):
#             break
#         if (frameId % math.floor(frameRate) == 0):
#             filename = os.path.join(training_frame_folder,videoFile.split('/')[1].split(' ')[0]+"_frame%d.jpg"%count)
#             cv2.imwrite(filename,frame)
#             count += 1
#     cap.release()
# print('done training set')

# for i in tqdm(range(test.shape[0])):
#     videoFile = test['video_name'][i]
#     cap = cv2.VideoCapture(os.path.join(videopath,videoFile.split(' ')[0].split('/')[0],videoFile.split(' ')[0].split('/')[1]))
#     frameRate = cap.get(5)
#     count = 0
#     while(cap.isOpened()):
#         frameId = cap.get(1)
#         ret,frame = cap.read()
#         if (ret != True):
#             break
#         if (frameId % math.floor(frameRate) == 0):
#             filename = os.path.join(test_frame_folder,videoFile.split('/')[1].split(' ')[0]+"_frame%d.jpg"%count)
#             cv2.imwrite(filename,frame)
#             count += 1
#     cap.release()
# print('done test set')

# # save the name of these frames with tag in a .csv file
# images = glob(os.path.join(training_frame_folder,'*.jpg'))
# train_image = []
# train_class = []
# for i in tqdm(range(len(images))):
#     train_image.append(images[i].split('/')[-1])
#     train_class.append(images[i].split('/')[-1].split('_')[1])
# train_data = pd.DataFrame()
# train_data['image'] = train_image
# train_data['class'] = train_class
# train_data.to_csv(os.path.join(main_path,'train_frame_list.csv'),header=True, index=False)

# images = glob(os.path.join(test_frame_folder,'*.jpg'))
# test_image = []
# test_class = []
# for i in tqdm(range(len(images))):
#     test_image.append(images[i].split('/')[-1])
#     test_class.append(images[i].split('/')[-1].split('_')[1])
# test_data = pd.DataFrame()
# test_data['image'] = test_image
# test_data['class'] = test_class
# test_data.to_csv(os.path.join(main_path,'test_frame_list.csv'),header=True, index=False)

# # step 2: training the model
# train = pd.read_csv(os.path.join(main_path,'train_frame_list.csv'))
# train_image = []
# for i in tqdm(range(train.shape[0])):
#     img = image.load_img(os.path.join(training_frame_folder,train['image'][i]),target_size=(224,224,3))
#     img = image.img_to_array(img)
#     img = img/255 # normalize the pixel value
#     train_image.append(img)
# X = np.asarray(train_image)
# X = preprocess_input(X,mode = 'tf')
# y = train['class']
# X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2,stratify = y)
# # here , striatify = y keeps the similar distribution of classess in both the training as well as the validation set.
# print(X_train.shape,X_test.shape)
# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)

# # define base model
base_model = VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))
# X_train = base_model.predict(X_train); X_test = base_model.predict(X_test)
# X_train = X_train.reshape(X_train.shape[0],7*7*512)
# X_test = X_test.reshape(X_test.shape[0],7*7*512)
# max_val = X_train.max()
# X_train = X_train/max_val
# X_test = X_test/max_val

# # define FC model
model = Sequential()
model.add(InputLayer((7*7*512,)))    # input layer
model.add(Dense(units=1024, activation='relu'))   # hidden layer
model.add(Dropout(0.5))      # adding dropout, so there is sync on training and validation
model.add(Dense(units=512, activation='relu'))    # hidden layer, increase number of layers to increase accuracy on training
model.add(Dropout(0.5))      # adding dropout
model.add(Dense(units=256, activation='relu'))    # hidden layer
model.add(Dropout(0.5))      # adding dropout
model.add(Dense(units=128, activation='relu'))    # hidden layer
model.add(Dropout(0.5))      # adding dropout
model.add(Dense(8, activation='softmax'))            # output layer

# from keras.callbacks import ModelCheckpoint
# mcp_save = ModelCheckpoint(os.path.join(main_path,'ucf_best_weight.hdf5'), save_best_only=True, verbose=1, monitor='val_loss', mode='min')
# # compile the model
# model.compile(loss = 'categorical_crossentropy',optimizer = 'Adam',metrics=['accuracy'])
# # training the model
# model.fit(X_train,y_train,epochs=200,validation_data=(X_test,y_test),callbacks=[mcp_save],batch_size=128)

# step 3: evaluate the model
model.load_weights(os.path.join(main_path,'ucf_best_weight.hdf5'))
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

test = pd.read_csv(os.path.join(main_path,'test_frame_list.csv'))
test_image = []
for i in tqdm(range(test.shape[0])):
    img = image.load_img(os.path.join(test_frame_folder,test['image'][i]),target_size=(224,224,3))
    img = image.img_to_array(img)
    img = img/255 # normalize the pixel value
    test_image.append(img)
predict_X = np.asarray(test_image)
predict_X = preprocess_input(predict_X,mode = 'tf')
predict_y = pd.get_dummies(test['class'])
predict_X = base_model.predict(predict_X)
predict_X = predict_X.reshape(predict_X.shape[0],7*7*512)
max_val = predict_X.max()
predict_X = predict_X / max_val

scores = model.evaluate(predict_X,predict_y)
print('acc: %.2f%%' %(scores[1]*100))