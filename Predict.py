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
from tensorflow.keras.models import load_model
directory = 'C:/Users/COMPUTER/Desktop/skin_cancer'
new_model = load_model(directory+'/model(res)/skincancer_model.h5')
new_model.summary()

# In [3]
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_directory = 'C:/Users/COMPUTER/Desktop/skin_cancer_images'



test_datagen = ImageDataGenerator(
    rescale = 1. /255,
)


test_generator = test_datagen.flow_from_directory(
    image_directory+'/predict_images',
    target_size = (512,512),
    color_mode = 'rgb',
    class_mode = 'binary',
    shuffle = True,
    batch_size = 8,
)

# In [4]
import numpy as np

test_generator.reset()
output = new_model.predict_generator(test_generator)
print(test_generator.class_indices) #데이터 클래스 보기
y_predict = np.argmax(output,axis=1)

# In [5]
eval = new_model.evaluate_generator(test_generator,verbose=1)
result_predict = new_model.predict(test_generator)
print(result_predict)
print('Test Loss: ',eval[0])
print('Test Accuracy: ',eval[1])