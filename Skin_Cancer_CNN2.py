#Contributor Nudding, Oldentomato
#In[1]
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

#In[2]
import pandas as pd
import os
import cv2
import shutil

directory = 'C:/Users/COMPUTER/Desktop/skin_cancer_images'
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
    shutil.move(directory + '/train_encoded/'+benign['image_name'][i]+'.jpg', directory + '/benign/'+benign['image_name'][i]+'.jpg')
    
for i in range(0,len(malignant)):
    shutil.move(directory + '/train_encoded/'+malignant['image_name'][i]+'.jpg', directory + '/malignant/'+malignant['image_name'][i]+'.jpg')

#In[3]
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1. /255
)
valid_test_datagen = ImageDataGenerator(
    rescale = 1. /255
)

train_generator = train_datagen.flow_from_directory(
    directory+'/train',
    target_size = (28,28),
    color_mode = 'grayscale',
    class_mode = 'binary',
    shuffle = True,
    batch_size = 32,
)

valid_generator = valid_test_datagen.flow_from_directory(
    directory+'/test',
    target_size = (28,28),
    color_mode = 'grayscale',
    class_mode = 'binary',
    shuffle = True,
    batch_size = 32,
)


#In[4]
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

#In[5]
conv1.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
tf.debugging.set_log_device_placement(True)

with tf.device("/gpu:0"):
    history = conv1.fit_generator(train_generator,
                                 #steps_per_epoch=20,
                                 epochs=100,
                                 validation_data=valid_generator,
                                 #validation_steps=10,
                                 shuffle=True)

#In[6]
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
