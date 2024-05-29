#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('python3.exe -m pip install --upgrade pip')


# In[2]:


# display, transform, read, split ...
import numpy as np
import cv2 as cv
import os
import splitfolders
import matplotlib.pyplot as plt
import tensorflow as tf

# tensorflow
import tensorflow.keras as keras
import tensorflow as tf

# image processing
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Add


# model / neural network
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


# In[3]:


# adenosis
size = (224,224)
img_adenosis = image.load_img(r"/Users/connormullins/Downloads/archive/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/adenosis/SOB_B_A-14-22549AB-40-001.png", target_size=size)
img_adenosis


# In[4]:


# fibroadenoma 
size = (224,224)
img_fibroadenoma = image.load_img(r"/Users/connormullins/Downloads/archive/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/fibroadenoma/SOB_B_F-14-9133-40-001.png", target_size=size)
img_fibroadenoma


# In[5]:


# phyllodes tumor
size = (224,224)
img_phyllodes_tumor = image.load_img(r"/Users/connormullins/Downloads/archive/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/phyllodes_tumor/SOB_B_PT-14-21998AB-40-001.png", target_size=size)
img_phyllodes_tumor


# In[6]:


# tubular adenona
size = (224,224)
img_tubular_adenona = image.load_img(r"/Users/connormullins/Downloads/archive/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/tubular_adenoma/SOB_B_TA-14-3411F-40-001.png", target_size=size)
img_tubular_adenona


# In[7]:


# ductal carcinoma
size = (224,224)
img_ductal_carcinoma = image.load_img(r"/Users/connormullins/Downloads/archive/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/ductal_carcinoma/SOB_M_DC-14-2523-40-010.png", target_size=size)
img_ductal_carcinoma


# In[8]:


# lobular carcinoma
size = (224,224)
img_lobular_carcinoma = image.load_img(r"/Users/connormullins/Downloads/archive/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/lobular_carcinoma/SOB_M_LC-14-12204-40-001.png", target_size=size)
img_lobular_carcinoma


# In[9]:


# mucinous carcinoma
size = (224,224)
img_mucinous_carcinoma = image.load_img(r"/Users/connormullins/Downloads/archive/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/mucinous_carcinoma/SOB_M_MC-14-10147-40-001.png", target_size=size)
img_mucinous_carcinoma


# In[10]:


# papillary carcinoma
size = (224,224)
img_papillary_carcinoma = image.load_img(r"/Users/connormullins/Downloads/archive/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/papillary_carcinoma/SOB_M_PC-14-9146-40-001.png", target_size=size)
img_papillary_carcinoma


# In[12]:


#image preprocessing



# In[11]:


datagen = ImageDataGenerator()
class_names = ['adenosis','fibroadenoma','phyllodes_tumor','tubular_adenona','ductal_carcinoma','lobular_carcinoma','mucinous_carcinoma','papillary_carcinoma']
splitfolders.ratio("/Users/connormullins/Downloads/archive/BreaKHis_v1/BreaKHis_v1/histology_slides/breast", output="data-split", seed=1337, ratio=(0.7, 0.2, 0.1), group_prefix=None, move=False)


# In[12]:


# training data
train_generator = datagen.flow_from_directory( 
    directory="data-split/train", 
    classes = class_names,
    target_size=(224, 224),  
    batch_size=32, 
    class_mode="categorical", 
)


# In[13]:


# validation data
valid_generator = datagen.flow_from_directory( 
    directory="data-split/val", 
    classes = class_names,
    target_size=(224, 224), 
    batch_size=32, 
    class_mode="categorical", 
)


# In[14]:


# test data
test_generator = datagen.flow_from_directory( 
    directory="data-split/test", 
    classes = class_names,
    target_size=(224, 224), 
    batch_size=32, 
    class_mode="categorical", 
)


# In[15]:


rescale = Sequential ([
    layers.Rescaling(1./255)]
#layers.RandomFlip("horizontal_and_vertical"),
#layers.RandomRotation(0.2),
)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomContrast(0.3, seed=1337)
    
])


# In[16]:


resnet_50 = ResNet50(include_top=False,input_shape=(224,224,3))


# In[49]:


# try before and after relu
model = Sequential([
data_augmentation,
resnet_50,
layers.Flatten(),
layers.Dense(512, kernel_regularizer=regularizers.L1(1e-6), activity_regularizer=regularizers.L1(1e-6), activation = 'relu'),
layers.BatchNormalization(),
layers.Dropout(0.5),
layers.Dense(8, activation = 'softmax')])
model.build((None,224,224,3))


for layer in model.layers:
    layer.trainable = True
for layer in resnet_50.layers[155:]:
    layer.trainable = True
for layer in resnet_50.layers[:155]:
    layer.trainable = False


# In[44]:


model.summary()


# In[45]:


resnet_50.summary()


# In[46]:


# define training function
def trainModel(model, epochs, optimizer):
    batch_size = 32
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model.fit(train_generator, validation_data=valid_generator, epochs=10, batch_size=batch_size)


# In[50]:


model_history = trainModel(model = model, epochs = 10, optimizer = "Adam")


# In[106]:


loss_train_curve = model_history.history["loss"]
loss_val_curve = model_history.history["val_loss"]
plt.plot(loss_train_curve, label = "Train")
plt.plot(loss_val_curve, label = "Validation")
plt.legend(loc = 'upper right')
plt.title("Loss")
plt.show()


# In[107]:


acc_train_curve = model_history.history["accuracy"]
acc_val_curve = model_history.history["val_accuracy"]
plt.plot(acc_train_curve, label = "Train")
plt.plot(acc_val_curve, label = "Validation")
plt.legend(loc = 'lower right')
plt.title("Accuracy")
plt.show()


# In[108]:


test_loss, test_acc = model.evaluate(test_generator)
print("The test loss is: ", test_loss)
print("The best accuracy is: ", test_acc*100)


# In[109]:


img = tf.keras.preprocessing.image.load_img(r"/Users/connormullins/Downloads/invasive-ductal-carcinoma.jpg", target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.array([img_array]) 
img


# In[110]:


# generate predictions for samples
predictions = model.predict(img_array)
print(predictions)


# In[111]:


# generate argmax for predictions
class_id = np.argmax(predictions, axis = 1)
print(class_id)


# In[112]:


# transform classes number into classes name
class_names[class_id.item()]


# In[ ]:




