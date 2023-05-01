#In [1]
#Gpu 설정
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

#In [3]
#Image데이터를 train:valid:test => 7:2:1 비율로 나눠야됨 (class.py)
from keras.preprocessing.image import ImageDataGenerator
image_directory = 'C:\\Users\\ISIA\\Desktop\\traveling\\trainingset'

# 학습에 사용될 이미지 데이터 생성
train_datagen= ImageDataGenerator( #여기다가 여러 파라미터를 넣어서 새로운 이미지들을 만들어야한다
    rescale = 1. /255, # 각픽셀이 255넘지 않게
    horizontal_flip = True, #좌우반전
    vertical_flip = True, #상하반전
    shear_range = 0.3, # 옆으로 0.3만큼 움직임
    rotation_range = 60, #회전 최대 60도
    width_shift_range = 0.3, #좌우 이동
    height_shift_range = 0.3, #상하 이동
    fill_mode = 'nearest'
)

test_datagen = ImageDataGenerator(
    rescale = 1. /255,
)

# 검증에 사용될 이미지 데이터 생성
valid_datagen = ImageDataGenerator(
    rescale = 1. /255,
)

# 학습에 사용될 데이터 생성
train_generator = train_datagen.flow_from_directory( #디렉토리에서 가져온 데이터를 flow시키는 것
    image_directory+'/train',
    target_size = (512,512), # (image_size, image_size)
    color_mode = 'rgb',
    class_mode = 'categorical', #class를 어떻게 읽는지 설정. categorical이라고 명시해주면 위에서 설정한 것처럼 파일 디렉토리로 class가 나눠짐
    shuffle = True, # 섞는다는 뜻. 순서를 무작위로 적용한다.
    batch_size = 8, # 배치 size는 한번에 gpu를 몇 개 보는가. 한번에 8장씩 학습시킨다
)

test_generator = test_datagen.flow_from_directory(
    image_directory+'/test',
    target_size = (512,512),
    color_mode = 'rgb',
    class_mode = 'binary',
    shuffle = False,
    batch_size = 8,
)

# 검증에 사용될 데이터 생성
valid_generator = valid_datagen.flow_from_directory(
    image_directory+'/valid',
    target_size = (512,512),
    color_mode = 'rgb',
    class_mode = 'binary',
    shuffle = False,
    batch_size = 8,
)




#In [4]
import keras
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16
from keras.models import Model

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(512,512,3)) #3채널만 됨
#imagenet에서 이미 학습된 가중치를 가져옴. 모델 커스터마이징 하려면 false로. 
vgg16.trainable = False #파인튜닝

flat = GlobalAveragePooling2D()(vgg16.output)

Add_layer = Dense(128, activation = 'relu')(flat)
Add_layer = Dense(75, activation = 'softmax')(Add_layer)
model = Model(inputs=vgg16.input, outputs=Add_layer)

model.summary() #모델 구성을 보여줌

#In [5]
model.compile(optimizer=keras.optimizers.Adam(lr=0.01) ,loss='categorical_crossentropy',metrics=['accuracy'])
#러닝레이트를 설정안해주면 트레이닝은 잘되지만 valid는 이상하게 된다


with tf.device("/gpu:0"):
    model.fit(train_generator,
                                 steps_per_epoch=len(train_generator),
                                 epochs=1,
                                 validation_data=valid_generator,
                                 validation_steps=len(valid_generator),
                                 shuffle=True)

#In [6]
from keras.callbacks import ModelCheckpoint, EarlyStopping
save_dir = 'C:\\Users\\ISIA\\Desktop\\traveling\\trainingset\\checkpoint\\' #C:/Users/COMPUTER/Desktop/skin_cancer/model(vgg)/checkpoint/
checkpoint = ModelCheckpoint(
    save_dir+'{epoch:02d}-{val_loss:.5f}.h5', #모델 저장 경로
    monitor='val_acc', #모델을 저장할 때 기준이 되는 값
    verbose = 1, # 1이면 저장되었다고 화면에 뜨고 0이면 안뜸
    save_best_only=True,
    mode = 'auto',
    #val_acc인 경우, 정확도이기 때문에 클수록 좋으므로 max를 쓰고, val_loss일 경우, loss값이기 떄문에 작을수록 좋으므로 min을써야한다
    #auto일 경우 모델이 알아서 min,max를 판단하여 모델을 저장한다
    save_weights_only=False, #가중치만 저장할것인가 아닌가
    save_freq = 1 #1번째 에포크마다 가중치를 저장 period를 안쓰고 save_freq룰 씀
)

earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta = 0.001,
    patience = 3,
    verbose=1,
    mode='auto'
)

callbacks = [checkpoint]


#In [7]
vgg16.trainable = True

model.compile(optimizer=keras.optimizers.Adam(lr=0.001) ,loss='categorical_crossentropy',metrics=['accuracy'])
#러닝레이트를 설정안해주면 트레이닝은 잘되지만 valid는 이상하게 된다


with tf.device("/gpu:0"):
    history = model.fit(train_generator,
                                 steps_per_epoch=len(train_generator),
                                 epochs=100,
                                 validation_data=valid_generator,
                                 validation_steps=len(valid_generator),
                                 callbacks = callbacks,
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
