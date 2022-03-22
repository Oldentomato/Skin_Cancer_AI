# In [1]
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

# In [2]
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

# In [3]

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
    class_mode = 'binary',
    shuffle = True,
    batch_size = 8,
)

valid_generator = valid_test_datagen.flow_from_directory(
    image_directory+'/test',
    target_size = (512,512),
    color_mode = 'rgb',
    class_mode = 'binary',
    shuffle = True,
    batch_size = 8,
)

# In [4]

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from keras import models
from keras import layers

# Model Resnet50 불러오기
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(512, 512, 3))  # weights='imagenet'
# resnet.summary()
resnet.trainable = False

# set_trainable = False
# for layer in resnet.layers:
#     if layer.name == 'conv4_block1_1_conv':
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False
#for layer in resnet.layers:
#    layer.trainable = False

flat = GlobalAveragePooling2D()(resnet.output)
Add_layer = Dense(1024, activation='relu')(flat)
Add_layer = Dense(512, activation='relu')(Add_layer)
Add_layer = Dense(1, activation='sigmoid')(Add_layer)


model = Model(inputs=resnet.input, outputs=Add_layer)

model.summary()

# In [4]
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
tf.debugging.set_log_device_placement(True)

with tf.device("/gpu:0"):
    history = model.fit_generator(train_generator,
#                                  steps_per_epoch=len(train_generator),
                                 epochs=100,
                                 validation_data=valid_generator,
#                                  validation_steps=len(valid_generator),
                                 shuffle=True)

# In [5]
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

# In [6]모델 저장
# model.save_weights(directory+'/skincancer_model(res)/epoch_068')
model.save(directory+'/model(res)/skincancer_model.h5')

