# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 15:54:58 2021

@author: gvmds
"""

from keras.models import load_model
from keras import optimizers
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from keras.preprocessing import image
import argparse

#Construct the argument parser and parse the arguments

'''ap = argparse.ArgumentParser()
ap.add_argument("--d", "--image", required=True,help="path to input dataset of images" , action="store_true")
ap.add_argument("--m", "--model", required=True,help="path to output trained model" , action="store_true")
args = vars(ap.parse_args())

parser = argparse.ArgumentParser()
parser.add_argument('--values', type=int, nargs=3)
args = parser.parse_args()'''




# dimensions of our images
img_width, img_height = 150, 150

# load the model we saved
model_name =  'C:\\Users\\gvmds\\.spyder-py3\\_Data_augmented_yoga_pose_classifier.model' 
#'C:\\Users\\gvmds\\zoo_classifier.model'
model = load_model(model_name)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

test_image = 'C:\\Users\\gvmds\\test.jpg'

# predicting images
img = image.load_img(test_image, target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict_classes(images, batch_size=20)
if classes == 0: prediction = "DOWNDOG"
if classes == 1: prediction = "GODDESS"
if classes == 2: prediction = "PLANK"
if classes == 3: prediction = "TREE"
if classes == 4: prediction = "WARRIOR 2"

print(classes)
print (prediction)

#python predict.py --image 'C:\\Users\\gvmds\\test.jpg' --model 'C:\\Users\\gvmds\\zoo_classifier.model'