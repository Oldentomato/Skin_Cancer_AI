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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
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


# In [2]

dirname = 'C:/Users/COMPUTER/Desktop/skin_cancer_images/melanoma_2' 

paths = []
# data_type = []
labels = []

for dirname, _, filenames in os.walk(dirname):
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



train_datagen= ImageDataGenerator( #???????????? ?????? ??????????????? ????????? ????????? ??????????????? ??????????????????
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

skf = StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
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

# Model Resnet50 ????????????
resnet = ResNet50(weights="imagenet", include_top=False, input_shape=(512, 512, 3))  

# include_top = False ??? ?????? convolution layer?????? ???????????? ?????? ?????? ?????? fully connected layer??? ??? ?????? ??? ??????


resnet.trainable = False


flat = GlobalAveragePooling2D()(resnet.output)
# Add_layer = Dense(5, activation='relu')(flat)
# Add_layer = Dense(51, activation='relu')(Add_layer)
# Add_layer = Dense(256, activation='relu')(Add_layer)
# Add_layer = Dense(512, activation='relu')(Add_layer)
Add_layer = Dense(1, activation='sigmoid')(flat)

model = Model(inputs=resnet.input, outputs=Add_layer)

model.summary()



#In [6]

save_dir = 'C:/Users/COMPUTER/Desktop/skin_cancer/model(res)/checkpoint2/'
checkpoint = ModelCheckpoint(
    save_dir+'epoch_{epoch:03d}-{val_loss:.2f}-{val_accuracy:.2f}.chpt', #?????? ?????? ??????
    monitor='val_acc', #????????? ????????? ??? ????????? ?????? ???
    verbose = 1, # 1?????? ?????????????????? ????????? ?????? 0?????? ??????
    save_best_only=True,
    save_weights_only= True
#     mode = 'auto',
    #val_acc??? ??????, ??????????????? ????????? ????????? ???????????? max??? ??????, val_loss??? ??????, loss????????? ????????? ???????????? ???????????? min???????????????
    #auto??? ?????? ????????? ????????? min,max??? ???????????? ????????? ????????????
#     save_weights_only=False, #???????????? ?????????????????? ?????????
#     save_freq = 1 #1?????? ??????????????? ???????????? ?????? period??? ????????? save_freq??? ???
)

earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta = 0.001,
    patience = 3,
    verbose=1,
    mode='auto'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', #val_loss??? ????????? ???????????? ?????? ?????? ReduceLROnPlateau??? ???????????????.
    factor= 0.01, #Learning rate??? ????????? ??????????????? ????????? ??????????????????.
    #????????? learning rate??? ?????? learning rate * factor?????????. 
    #?????? ?????? ?????? 0.01 ?????? factor??? 0.8??? ???, ??????????????? ??????????????? ??? ?????? lr??? 0.008??? ?????????.
    patience = 1 #training??? ??????????????? ????????? monitor?????? ?????? ????????? ?????? ??????, ????????? monitor?????? ???????????? ????????? epoch??? ????????????, 
    #learning rate??? ????????? ?????? ????????????.
    #???????????? patience??? 3??????, 30????????? ???????????? 99%?????? ???, ?????? 31????????? ????????? 98%, 32????????? 98.5%, 33????????? 98%?????? ????????? ????????? 
    #3??? ?????? ????????? ????????????, ??? ??????????????? ???????????????.
)

callbacks = [checkpoint,reduce_lr]

# In [7]

model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),loss='binary_crossentropy',metrics=['accuracy'])
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
model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),loss='binary_crossentropy',metrics=['accuracy'])
tf.debugging.set_log_device_placement(True)
with tf.device("/gpu:0"):
    history = model.fit_generator(train_generator,
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


