#In[1]
import cv2
import numpy as np
import tensorflow as tf

def contour(img):
    
    mask = np.zeros((img.shape[0], img.shape[1],3),np.uint8)

    #쓰레시홀드로 바이너리 이미지를 만들어 검은 배경에 흰색전경으로 반전
    ret,imgthres = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)

    #contours는 검출된 컨투어 좌표리스트, hierarchy는 해당 컨투어의 계층정보 배열
    contours,hierarchy = cv2.findContours(imgthres,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #매개변수에 -1부분은 계층별 구분인데 -1은 모든 계층을 나타낸다
    cv2.drawContours(mask,contours,-1,(255,255,255),4)

    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    result = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
    result = cv2.bilateralFilter(result, d=-1, sigmaColor=50,sigmaSpace=50)
    return result

# def otsu_filter(img):
#     blur = cv2.GaussianBlur(img,(5,5),0)
#     _, th4 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#     #50,150
#     th5 = cv2.Canny(th4,100,200)


#     return th5

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
import cv2
import os
import tensorflow as tf

progress = 0 #100개 제한용

x_train_all = []
y_train_all = []

directory = 'C:\\Users\\COMPUTER\\Desktop\\skin_cancer'
files = os.listdir(directory+'\\train')


#In[3]
for i in files:
    img = cv2.imread(directory+'\\train\\'+i,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28,28))
    img = contour(img)
    # x_train_all.append(img)
    cv2.imwrite(directory+'\\train_encoded\\'+i,img)
    progress += 1
<<<<<<< HEAD
    print('진행도: '+str(progress)+' 전체갯수: '+str(len(files)))
=======
    print('진행도: '+str(progress)+' 전체갯수: '+str(files))
    if(progress == 100):
        break


files2 = []
files2 = os.listdir(directory+'\\train_encoded')
for j in files2:
    img2 = cv2.imread(directory+'train_encoded'+j)
    x_train_all.append(img2)


# print(x_train_all[0])
>>>>>>> 5a42b343dbb0403e98e76ac621088fc95c383e36

#In[4]
import pandas as pd
import numpy as np

df = pd.read_csv(directory+'\\ISIC_2020_Training_GroundTruth.csv')
target = df['target']
y_train_all = target.drop(0, axis=0) #행은 axis 0 열은 axis 1이다 첫번째 행 제거(얘 혼자 다른 사진임)
y_train_all = target.iloc[1:100]
# print(y_train_all)
print('done')

<<<<<<< HEAD

#In[5]
encode_progress = 0
condition = (df.target == 0)
df_encoded = df.loc[0:20000][condition]
y_train_all = df.drop(df_encoded.index)
y_train_all = y_train_all['target']
print(y_train_all)
df_encoded.reset_index(inplace=True)
print(df_encoded)

# for j in range(0,20000):
#     os.remove(directory+'\\train_encoded\\'+df_encoded['image_name'][j]+'.jpg')
#     encode_progress += 1
#     print('진행상황:'+str(encode_progress))
    

#In[6]
y_train_all = df.drop(df_encoded.index)
y_train_all = y_train_all['target']
print(y_train_all)

#In[7]
files2 = os.listdir(directory+'\\train_encoded')
for i in files2:
    img = cv2.imread(directory+'\\train_encoded\\'+i,cv2.IMREAD_GRAYSCALE)
    x_train_all.append(img)

#In[8]
x_train_all = np.array(x_train_all)
print(x_train_all.shape)
print(y_train_all.shape)

#In[9]
=======
#검사용
print(x_train_all.shape())
print(y_train_all.shape())

>>>>>>> 5a42b343dbb0403e98e76ac621088fc95c383e36
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout

# stratify=y_train_all
x_train,x_val,y_train,y_val = train_test_split(x_train_all,y_train_all, stratify=y_train_all, test_size=0.2)
print(x_train.shape)
x_train = x_train.reshape(-1,28,28,1)
print(x_train.shape)
print(y_train.shape)
x_val = x_val.reshape(-1,28,28,1)

x_train = x_train//255
x_val = x_val//255




x_train = np.array(x_train)
x_train = x_train.reshape(-1,28,28,1)
x_train = x_train/255

x_val = np.array(x_val)
x_val = x_val.reshape(-1,28,28,1)
x_val = x_val/255

conv1 = tf.keras.Sequential()
conv1.add(Conv2D(10,(3,3),activation='relu', padding='same',input_shape=(28,28,1)))
conv1.add(MaxPooling2D(2,2))
conv1.add(Flatten())
conv1.add(Dropout(0.5))

conv1.add(Dense(100,activation='relu'))
conv1.add(Dense(1,activation='sigmoid'))

conv1.summary()


#In[10]
conv1.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
tf.debugging.set_log_device_placement(True)
print(x_train.shape)
print(y_train.shape)

with tf.device("/gpu:0"):
    history = conv1.fit(x_train,y_train,epochs=10, batch_size=128, validation_data=(x_val,y_val))

#In[11]
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