from warnings import filterwarnings
filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, Input, MaxPooling2D
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16,  preprocess_input
from tensorflow.keras.applications import VGG19

from sklearn.metrics import classification_report, confusion_matrix

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


BASE_PATH = './Yoga_Dataset/DATASET/'

filenames,labels = [],[]

for dirname in os.listdir(f'{BASE_PATH}TRAIN'):
    for file in os.listdir(f'{BASE_PATH}TRAIN/{dirname}'):
        filenames.append(file)
        labels.append(dirname)

df_train = pd.DataFrame(data={
    'filename': filenames,
    'labels': labels
})


df_train.head()


df_train.labels.value_counts()


rows = 5
cols = 5
fig, axs = plt.subplots(rows, cols, figsize=(16, 16))
for i, row in enumerate(list(df_train.labels.unique())):
    for j, filename in enumerate(df_train[df_train.labels == row]['filename'].tolist()[:cols]):
        img = load_img(os.path.join(BASE_PATH,'TRAIN',row, filename))
        axs[i,j].matshow(img)
        axs[i,j].axis('off')
        axs[i,j].set_title(row.upper(), fontsize=24)
fig.tight_layout()

num_classes = list(df_train.labels.unique())
print(num_classes)
len(num_classes)


# ### Config params

IMG_H = 200
IMG_W = 200
IMG_C = 3

BATCH_SIZE = 32
EPOCHS = 1


# ### Data Augumentation using ImageDataGenerator
train_gen = ImageDataGenerator(rescale=1./255,
                               shear_range=0.2,
                               zoom_range=0.2,
                               width_shift_range=0.12,
                               height_shift_range=0.12,
                               horizontal_flip=True)

test_gen = ImageDataGenerator(rescale=1./255)

train_set = train_gen.flow_from_directory(f'{BASE_PATH}TRAIN',
                                          target_size=(IMG_W,IMG_H),
                                          batch_size=BATCH_SIZE,
                                          class_mode='categorical')

test_set = test_gen.flow_from_directory(f'{BASE_PATH}TEST',
                                          target_size=(IMG_W,IMG_H),
                                          batch_size=BATCH_SIZE,
                                          class_mode='categorical')

def create_model():
    tf.keras.backend.clear_session()
    
    cmodel = VGG19(input_shape = (IMG_W, IMG_H, IMG_C), 
                         weights='imagenet', 
                         include_top=False,)
    
    # there is no need to train existing weights
    for layer in cmodel.layers:
        layer.trainable = False
        
    x = Flatten()(cmodel.output)
    #x = cmodel.output
 
    prediction = Dense(len(num_classes), activation='softmax')(x)

    # create model object
    model = Model(inputs = cmodel.input, outputs = prediction)
    return model

model = create_model()
model.summary()

lr_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
es = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=0, mode='auto')


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
callbacks_list = [es, lr_reduction]

history = model.fit(train_set,
                    validation_data=test_set,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=callbacks_list,
                    shuffle=True
                    )


# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

model.evaluate(train_set), model.evaluate(test_set)

y_val_org = []
for i in range( test_set.__len__() ):
    y_val_org.extend(
        test_set.__getitem__( i )[1] 
    )
y_val_org = np.array(y_val_org)
y_val_org = np.argmax(y_val_org, axis=1)

y_val_org

ypreds = model.predict(test_set)
ypreds = np.argmax(ypreds, axis=1)
ypreds

cf_matrix = confusion_matrix(y_val_org, ypreds)

plt.figure(figsize=(20,8))
ax = sns.heatmap(cf_matrix, annot=True, fmt='g')
plt.show()

print("\n\n")
print(classification_report(y_val_org, ypreds))
