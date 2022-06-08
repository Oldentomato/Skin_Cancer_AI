from cv2 import log
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras.models import Model
from keras import models
from keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
from functools import partial
from pymongo import MongoClient
import mongo_url as uri
import datetime

# from sklearn.model_selection import train_test_split


# In [1]
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

mongo_url = uri.Mongo_URL
client = MongoClient()
db = client['Model_Database']
collection = db['resnet_collection']
Model_Name = 'ResNetModel_1'
log_data = list()
start_lr = 0.0001

dirname = 'C:/Users/COMPUTER/Desktop/skin_cancer_images/melanoma_2' 

paths = []
# data_type = []
labels = []

#os.walk는 하위의 폴더들을 for문으로 탐색할 수 있게 해준다.
#3개의 튜플을 넘겨준다
# root: dir과 files가 있는 path
# dirs: root 아래에 있는 폴더들
# files: root 아래에 있는 파일들
for dirname, _, filenames in os.walk(dirname): #for 문으로 써야하는가?
    for filename in filenames:
        if '.jpg' or '.jpeg' in filename:
            path = dirname + '/' + filename
            paths.append(path)
            
#             if 'train' in path:
#                 data_type.append('train')
#             elif 'valid' in path:
#                 data_type.append('valid')
#             else:
#                 data_type.append('N/A')
                
            if 'Positive' in path:
                labels.append('Positive')
            elif 'Negative' in path:
                labels.append('Negative')
            else:
                labels.append('N/A')
print(len(paths),len(labels))
data_df = pd.DataFrame({'path': paths, 'label':labels})

# train_df = data_df[data_df['data_type'] == 'train']
# valid_df = data_df[data_df['data_type'] == 'valid']
# tr_df, val_df = train_test_split(train_df, stratify=train_df['label'],test_size=0.2, random_state=42)
print('Train:',data_df.shape)

# In [2]



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


valid_datagen = ImageDataGenerator(
    rescale = 1. /255
)

skf = StratifiedKFold(n_splits=3,random_state=42,shuffle=True)
for train_index, valid_index in skf.split(data_df,data_df['label']):
    training_data = data_df.iloc[train_index]
    validation_data = data_df.iloc[valid_index]
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe = training_data,
        x_col = 'path',
        y_col = 'label',
        target_size = (512,512),
        color_mode = 'rgb',
        class_mode = 'binary',
        shuffle = True,
        batch_size = 8,
    )


    valid_generator = valid_datagen.flow_from_dataframe(
        dataframe = validation_data,
        x_col = 'path',
        y_col = 'label',
        target_size = (512,512),
        color_mode = 'rgb',
        class_mode = 'binary',
        shuffle = True,
        batch_size = 8,
    )





# In [4]

# Model Resnet50 불러오기
resnet = ResNet50(weights="imagenet", include_top=False, input_shape=(512, 512, 3))  

# include_top = False 를 해야 convolution layer들만 가져오고 밑에 내가 만든 fully connected layer를 더 쌓을 수 있다


resnet.trainable = False

RegularizedDense = partial(Dense,
                    activation='relu',
                    kernel_initializer="he_normal",
                    kernel_regularizer=keras.regularizers.l2(0.01))
#partial을 쓰지 않으면 밑에서 Dense를 호출할때마다 위의 3가지 인자를 각각 써줘야한다.
#하지만 partial을 쓰면 미리 정의를 해주고 사용할 때 필요한 인자만 바꿔주면 된다.
#partial(사용할 함수, 함수의 인자값,...)

flat = GlobalAveragePooling2D()(resnet.output)
Add_layer = RegularizedDense(256)(flat)
Add_layer = RegularizedDense(512)(Add_layer)
Add_layer = RegularizedDense(1,activation="sigmoid",kernel_initializer="glorot_uniform")(flat)

model = Model(inputs=resnet.input, outputs=Add_layer)

model.summary()


#In [6]


class SendLog_ToMongo(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        collection.insert_one({'model_name' : Model_Name})
        collection.insert_many({
            'epoch': 0,
            'loss': 0,
            'acc': 0,
            'val_loss': 0,
            'val_acc': 0
        })
    def on_epoch_end(self, epoch, logs=None):
        log_data.append({
            'epoch' : epoch,
            'loss' : logs.get('loss'),
            'acc' : logs.get('acc'),
            'val_loss' : logs.get('val_loss'),
            'val_acc' : logs.get('val_acc')
        })
        collection.update_many({'model_name':Model_Name},{'$set':log_data})


save_dir = 'C:/Users/COMPUTER/Desktop/skin_cancer/model(res)/checkpoint2/'
checkpoint = ModelCheckpoint(
    save_dir+'epoch_{epoch:03d}-{val_loss:.2f}-{val_accuracy:.2f}.h5', #모델 저장 경로
    monitor='val_accuracy', #모델을 저장할 때 기준이 되는 값
    verbose = 1, # 1이면 저장되었다고 화면에 뜨고 0이면 안뜸
    save_best_only=True,
    # save_weights_only= True
#     mode = 'auto',
    #val_acc인 경우, 정확도이기 때문에 클수록 좋으므로 max를 쓰고, val_loss일 경우, loss값이기 떄문에 작을수록 좋으므로 min을써야한다
    #auto일 경우 모델이 알아서 min,max를 판단하여 모델을 저장한다
#     save_weights_only=False, #가중치만 저장할것인가 아닌가
#     save_freq = 1 #1번째 에포크마다 가중치를 저장 period를 안쓰고 save_freq룰 씀
)

earlystop = EarlyStopping(
    monitor='val_accuracy',
    min_delta = 0.001,
    patience = 3,
    verbose=1,
    mode='auto'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', #val_loss가 더이상 감소되지 않을 경우 ReduceLROnPlateau을 적용합니다.
    factor= 0.01, #Learning rate를 얼마나 감소시킬지 정하는 인자값입니다.
    #새로운 learning rate는 기존 learning rate * factor입니다. 
    #예를 들어 현재 0.01 이고 factor가 0.8일 때, 콜백함수가 실행된다면 그 다음 lr은 0.008이 됩니다.
    patience = 1 #training이 진행됨에도 더이상 monitor되는 값의 개선이 없을 경우, 최적의 monitor값을 기준으로 몇번의 epoch을 진행하고, 
    #learning rate를 조절할 지의 값입니다.
    #예를들어 patience는 3이고, 30에폭에 정확도가 99%였을 때, 만약 31번째에 정확도 98%, 32번째에 98.5%, 33번째에 98%라면 모델의 개선이 
    #3번 동안 개선이 없었기에, 이 콜백함수를 실행합니다.
)

log_dir = "logs\\fit\\"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq = 1)

callbacks = [checkpoint,reduce_lr,earlystop,SendLog_ToMongo(),tensorboard_callback]

# In [7]

model.compile(optimizer=keras.optimizers.Adam(learning_rate = start_lr),loss='binary_crossentropy',metrics=['accuracy'])
tf.debugging.set_log_device_placement(True)

with tf.device("/gpu:0"):
    history = model.fit(train_generator,
                                 steps_per_epoch=len(train_generator),
                                 epochs=1,
                                 validation_data=valid_generator,
                                 validation_steps=len(valid_generator),
                                 shuffle=True)


#In [5]
resnet.trainable = True


# In [7]
model.compile(optimizer=keras.optimizers.Adam(learning_rate = start_lr),loss='binary_crossentropy',metrics=['accuracy'])
tf.debugging.set_log_device_placement(True)
with tf.device("/gpu:0"):
    history = model.fit(train_generator,
                                 steps_per_epoch=len(train_generator),
                                 epochs=100,
                                 validation_data=valid_generator,
                                 validation_steps=len(valid_generator),
                                 callbacks = callbacks,
                                 shuffle=True)

# In [5]

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

# In [5]
model.load_weights(save_dir+'epoch_{epoch:03d}-{val_loss:.2f}-{val_accuracy:.2f}.chpt')
model_path = 'C:/Users/COMPUTER/Desktop/skin_cancer/model(res)/save_model/'
model.save(model_path+'resnet_model.h5')


