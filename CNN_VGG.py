#In [1]
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

#In [2]

import pandas as pd
import os
import cv2
import shutil

directory = 'C:/Users/COMPUTER/Desktop/skin_cancer'
directory_2 = 'C:/Users/COMPUTER/Desktop/skin_cancer_images/images'
files = os.listdir(directory+'/train_encoded')
df = pd.read_csv('C:/Users/COMPUTER/Desktop/skin_cancer/ISIC_2020_Training_GroundTruth.csv')
df = df.drop(0, axis=0)
condition_be = (df.target == 0)
condition_mal = (df.target == 1)
benign = df.loc[:][condition_be]
malignant = df.loc[:][condition_mal]
benign.reset_index(inplace=True)
malignant.reset_index(inplace=True)



for i in range(0,len(benign)):
    shutil.move(directory + '/train_encoded/'+benign['image_name'][i]+'.jpg', directory_2 + '/benign/'+benign['image_name'][i]+'.jpg')
    
for i in range(0,len(malignant)):
    shutil.move(directory + '/train_encoded/'+malignant['image_name'][i]+'.jpg', directory_2 + '/malignant/'+malignant['image_name'][i]+'.jpg')


#In [3]

from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_directory = 'C:/Users/COMPUTER/Desktop/skin_cancer_images/color'

train_datagen= ImageDataGenerator( #여기다가 여러 파라미터를 넣어서 새로운 이미지들을 만들어야한다
    rescale = 1. /255,
    horizontal_flip = True,
    vertical_flip = True,
    shear_range = 0.3,
    rotation_range = 60,
    width_shift_range = 0.3,
    height_shift_range = 0.3,
    fill_mode = 'nearest'
)



valid_test_datagen = ImageDataGenerator(
    rescale = 1. /255,
)

train_generator = train_datagen.flow_from_directory(
    image_directory+'/train',
    target_size = (512,512),
    color_mode = 'rgb',
    class_mode = 'categorical',
    shuffle = True,
    batch_size = 32,
)



valid_generator = valid_test_datagen.flow_from_directory(
    image_directory+'/test',
    target_size = (512,512),
    color_mode = 'rgb',
    class_mode = 'categorical',
    shuffle = True,
    batch_size = 32,
)

#In [4]
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf
from tensorflow.keras.layers import Dropout

conv1 = tf.keras.Sequential()
conv1.add(Conv2D(16,(3,3),activation='relu', padding='same',input_shape=(28,28,1)))
conv1.add(MaxPooling2D(2,2))
conv1.add(Conv2D(16,(3,3),activation='relu', padding='same'))
conv1.add(MaxPooling2D(2,2))
conv1.add(Conv2D(16,(3,3),activation='relu', padding='same'))
conv1.add(MaxPooling2D(2,2))

conv1.add(Flatten())


conv1.add(Dense(128,activation='relu'))
conv1.add(Dropout(0.5))
conv1.add(Dense(32,activation='relu'))
conv1.add(Dropout(0.5))
conv1.add(Dense(1,activation='sigmoid'))

conv1.summary()

#In [5] In [6]과 같이 모델구성 부분 둘중 하나만 쓸것
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3)) #3채널만 됨

flat = GlobalAveragePooling2D()(vgg16.output)

#Add_layer = Dense(256, activation= 'relu')(flat)
Add_layer = Dense(64, activation = 'relu')(flat)
# Add_layer = Dense(32, activation = 'relu')(Add_layer)
Add_layer = Dense(1, activation = 'sigmoid')(Add_layer)
model = Model(inputs=vgg16.input, outputs=Add_layer)

model.summary()

#In [6]
from keras import models
from keras import layers
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

#In [7]

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
tf.debugging.set_log_device_placement(True)

with tf.device("/gpu:0"):
    history = model.fit_generator(train_generator,
#                                  steps_per_epoch=len(train_generator),
                                 epochs=200,
                                 validation_data=valid_generator,
#                                  validation_steps=len(valid_generator),
                                 shuffle=True)

#In [7]
                                
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss','val_loss'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_accuracy','val_accuracy'])
plt.show()