# Import libraries
from datetime import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.optimizers as Opti
import keras.models as Modl
import keras.losses as Loss
import keras.layers as Layr
import keras.preprocessing as PreP
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define runtime variables
# input vars
epochs = 2
batch_size = 32
img_dims = 128
root_path = os.sep + os.path.join('Users', 'lujackso', 'Downloads', 'DAAN-570-302-scratch', 'psu-daan-570-group-1',
                                  'images', 'DATASET') + os.sep  # path is now OS agnostic


# Define custom functions
def show_batch(image_batch, label_batch, class_indices, size=10):
    intLabels = np.argmax(label_batch, axis=1)  # Convert OneHot To Int Values
    textLabels = list(class_indices.keys())  # Convert Class Indices Keys to List

    plt.figure("Collage", tight_layout=True, figsize=(10, 5))
    for n in range(size):
        plt.subplot(2, 5, n + 1)
        plt.imshow(image_batch[n])
        plt.title(textLabels[intLabels[n]])
        # plt.colorbar(im)
        plt.axis('off')
    plt.show()


# Data Structure, Define and Initalize ImageDataGenerators
train_data = PreP.image.ImageDataGenerator(rescale=1. / 255)

test_data = PreP.image.ImageDataGenerator(rescale=1. / 255)

classes = 5
training_set = train_data.flow_from_directory(root_path + 'TRAIN',  # 5 classes
                                              target_size=(img_dims, img_dims),
                                              batch_size=batch_size,
                                              # color_mode='rgba',  # (img_dims, img_dims, 4)
                                              class_mode='categorical')

test_set = test_data.flow_from_directory(root_path + 'TEST',  # 5 classes
                                         target_size=(img_dims, img_dims),
                                         batch_size=batch_size,
                                         # color_mode='rgba',  # (img_dims, img_dims, 4)
                                         class_mode='categorical')


# Explore Numpy Array Attributes
print("Explore Numpy Array Attributes for ImageDataGenerator.next() -> DirectoryIterator object")
image_batch, label_batch = training_set.next()

print("X-ndim: ", image_batch.ndim)
print("X-shape:", image_batch.shape)
print("X-size: ", image_batch.size)

print("Y-ndim: ", label_batch.ndim)
print("Y-shape:", label_batch.shape)
print("Y-size: ", label_batch.size)

# print(training_set.labels[0])
# print(label_batch[0])


# Explore ImageDataGenerator -> DirectoryIterator object
print(type(training_set.class_indices))
print(training_set.class_indices)
print(list(training_set.class_indices.keys()))

# Summarize and Plot ImageDataGenerator Batch Samples
image_batch, label_batch = next(training_set)
intLabels = np.argmax(label_batch, axis=1)  # Convert OneHot To Int Values
textLabels = list(training_set.class_indices.keys())
cntClasses = pd.Series(intLabels).value_counts()

i = 1
print("Summarize Random Sample by Count(#) of each class.")
for c in range(len(textLabels)):
    print(c, ":", textLabels[c], cntClasses[c])

print("\n")
print("Visualize Random Sample from ImageDataGenerator -> DirectoryIterator object.")
for n in range(i):
    # print(n, np.argmax(label_batch[n], axis=0))
    print(n, ":", textLabels[intLabels[n]], "-", intLabels[n], "-", label_batch[n])
    plt.figure(n)
    plt.imshow(image_batch[n])
    plt.title(textLabels[intLabels[n]])
    plt.axis('on')
    plt.show()

# Plot Collage of ImageDataGenerator Batch Samples
# image_batch, label_batch = next(training_set)
show_batch(image_batch, label_batch, training_set.class_indices)