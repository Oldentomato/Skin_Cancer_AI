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
res = load_model(directory+'/model(res)/skincancer_model(91).h5')
vgg = load_model(directory+'/model(vgg)/skincancer_model(93).h5')
new_model = vgg
new_model.summary()

# In [3]
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_directory = 'C:/Users/COMPUTER/Desktop/skin_cancer_images/melanoma'



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
eval = new_model.evaluate(test_generator,verbose=1)
print('Test Loss: ',eval[0])
print('Test Accuracy: ',eval[1])

# In [5]
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt

result_predict = new_model.predict(test_generator)
y_predict = np.around(result_predict)
y_predict = np.ravel(y_predict) #2차원에서 1차원으로
cm = confusion_matrix(test_generator.classes, y_predict)
df_cm = pd.DataFrame(cm, list(test_generator.class_indices.keys()), list(test_generator.class_indices.keys()))
fig, ax = plt.subplots(figsize=(10,8))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, annot_kws={"size":16}, cmap=plt.cm.Blues)
plt.title('Confusion Matrix\n')
plt.show()

# In [6]
print('Classification Report\n')
target_names = list(test_generator.class_indices.keys())
print(classification_report(test_generator.classes, y_predict, target_names=target_names))