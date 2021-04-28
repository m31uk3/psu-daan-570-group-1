# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os,shutil
import argparse
import numpy as np
from keras import layers
from keras import models
from keras import optimizers
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Working directory = C:\\Users\\gvmds

base_dir = 'C:\\Users\\gvmds\\Desktop\\Yoga_Dataset\\DATASET\\'
#os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'TRAIN')
#os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'VALIDATION')
#os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'TEST')
#os.mkdir(test_dir)

# make test, train and valiadation directories for each animal

train_downdog_dir = os.path.join(train_dir, 'downdog')
train_goddess_dir = os.path.join(train_dir, 'goddess')
train_plank_dir = os.path.join(train_dir, 'plank')
train_tree_dir = os.path.join(train_dir, 'tree')
train_warrior2_dir = os.path.join(train_dir, 'warrior2')


validation_downdog_dir = os.path.join(validation_dir, 'downdog')
validation_goddess_dir = os.path.join(validation_dir, 'goddess')
validation_plank_dir = os.path.join(validation_dir, 'plank')
validation_tree_dir = os.path.join(validation_dir, 'tree')
validation_warrior2_dir = os.path.join(validation_dir, 'warrior2')


test_downdog_dir = os.path.join(test_dir, 'downdog')
test_goddess_dir = os.path.join(test_dir, 'goddess')
test_plank_dir = os.path.join(test_dir, 'plank')
test_tree_dir = os.path.join(test_dir, 'tree')
test_warrior2_dir = os.path.join(test_dir, 'warrior2')

#os.mkdir(train_cats_dir)

# Directory with our training dog pictures
#train_dogs_dir = os.path.join(train_dir, 'dogs')
#os.mkdir(train_dogs_dir)

# Directory with our training panda pictures
#train_pandas_dir = os.path.join(train_dir, 'panda')
#os.mkdir(train_pandas_dir)

# Directory with our validation cat pictures
#validation_cats_dir = os.path.join(validation_dir, 'cats')
#os.mkdir(validation_cats_dir)

# Directory with our validation dog pictures
#validation_dogs_dir = os.path.join(validation_dir, 'dogs')
#os.mkdir(validation_dogs_dir)

# Directory with our validation pandas pictures
#validation_pandas_dir = os.path.join(validation_dir, 'panda')
#os.mkdir(validation_pandas_dir)

# Directory with our validation cat pictures
#test_cats_dir = os.path.join(test_dir, 'cats')
#os.mkdir(test_cats_dir)

# Directory with our validation dog pictures
#test_dogs_dir = os.path.join(test_dir, 'dogs')
#os.mkdir(test_dogs_dir)

# Directory with our validation pandas pictures
#test_pandas_dir = os.path.join(test_dir, 'panda')
#os.mkdir(test_pandas_dir)


# Copy first 500 cat images to train_cats_dir
'''fnames = ['{}.jpg'.format(str(i).zfill(7)) for i in range(0,500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_cat, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 250 cat images to validation_cats_dir


# Copy next 250 cat images to validation_cats_dir
fnames = ['cats_{}.jpg'.format(str(i+1).zfill(5)) for i in range(500, 750)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_cat, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 250 cat images to test_cats_dir
fnames = ['cats_{}.jpg'.format(str(i+1).zfill(5)) for i in range(750, 1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_cat, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy first 500 dog images to train_dogs_dir
fnames = ['dogs_{}.jpg'.format(str(i+1).zfill(5)) for i in range(0,500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_dog, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 250 dog images to validation_dogs_dir
fnames = ['dogs_{}.jpg'.format(str(i+1).zfill(5)) for i in range(500, 750)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_dog, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 250 dog images to test_dogs_dir
fnames = ['dogs_{}.jpg'.format(str(i+1).zfill(5)) for i in range(750, 1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_dog, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)


# Copy first 500 panda images to train_pandas_dir
fnames = ['panda_{}.jpg'.format(str(i+1).zfill(5)) for i in range(0,500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_panda, fname)
    dst = os.path.join(train_pandas_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 250 panda images to validation_pandas_dir
fnames = ['panda_{}.jpg'.format(str(i+1).zfill(5)) for i in range(500, 750)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_panda, fname)
    dst = os.path.join(validation_pandas_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 250 panda images to test_pandas_dir
fnames = ['panda_{}.jpg'.format(str(i+1).zfill(5)) for i in range(750, 1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_panda, fname)
    dst = os.path.join(test_pandas_dir, fname)
    shutil.copyfile(src, dst)'''
    
print('total training downdog pose images:', len(os.listdir(train_downdog_dir)))
print('total training goddess pose images:', len(os.listdir(train_goddess_dir)))
print('total training plank pose images:', len(os.listdir(train_plank_dir)))
print('total training tree pose images:', len(os.listdir(train_tree_dir)))
print('total training warrior2 pose images:', len(os.listdir(train_warrior2_dir)))


print('total validation downdog pose images:', len(os.listdir(validation_downdog_dir)))
print('total validation goddess pose images:', len(os.listdir(validation_goddess_dir)))
print('total validation plank pose images:', len(os.listdir(validation_plank_dir)))
print('total validation tree pose images:', len(os.listdir(validation_tree_dir)))
print('total validation warrior2 pose images:', len(os.listdir(validation_warrior2_dir)))

print('total test downdog pose images:', len(os.listdir(test_downdog_dir)))
print('total test goddess pose images:', len(os.listdir(test_goddess_dir)))
print('total test plank pose images:', len(os.listdir(test_plank_dir)))
print('total test tree pose images:', len(os.listdir(test_tree_dir)))
print('total test warrior2 pose images:', len(os.listdir(test_warrior2_dir)))

   


#Building the CNN

model = models.Sequential()



model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(150,150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

model.summary()



model.compile(loss='categorical_crossentropy',
              optimizer='adam',#optimizers.RMSprop(lr=1e-6),
              metrics=['acc'])


# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150,150),
        batch_size=25,
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='categorical',
        shuffle=True
        )

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150,150),
        batch_size=20,
        class_mode='categorical')



# Fit the model 
history = model.fit(
      train_generator,
      steps_per_epoch=30,
      epochs=10,
      validation_data= validation_generator,
      validation_steps=10)

# Plot training and validation accuracy and loss

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))


plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break
#Save the model
model.save('yoga_pose_classifier.model')

'''
# Data Augmentation to enhance model perfromance

datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# This is module with image preprocessing utilities
from keras.preprocessing import image

fnames = [os.path.join(train_downdog_dir, fname) for fname in os.listdir(train_downdog_dir)]

# We pick one image to "augment"
img_path = fnames[3]

# Read the image and resize it
img = image.load_img(img_path, target_size=(150,150))

# Convert it to a Numpy array with shape (150,150, 3)
x = image.img_to_array(img)

# Reshape it to (1, 150,150, 3)
x = x.reshape((1,) + x.shape)

# The .flow() command below generates batches of randomly transformed images.
# It will loop indefinitely, so we need to `break` the loop at some point!
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break

plt.show()'''

model = models.Sequential()
'''model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150,150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))'''


model.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(150,150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))


#Compile the model wil augmented data
model.compile(loss='categorical_crossentropy',
              optimizer= 'adam',#optimizers.RMSprop(lr=1e-6),
              metrics=['acc'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    #rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150,150),
        batch_size=25,
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')

#Validate with TEST dataset
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150,150),
        batch_size=20,
        class_mode='categorical')

'''history = model.fit_generator(
      train_generator,
      steps_per_epoch=10,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=10)'''




plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Test accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Test loss')
plt.title('Training and Test loss')
plt.legend()

plt.show()

'''

# plots images with labels within jupyter notebook
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

imgs, labels = next(train_generator)
plots(imgs, titles=labels )

w=10
h=10
fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 5
for i in range(1, columns*rows +1):
    img = np.random.randint(10, size=(h,w))
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()'''









