# importing libraries
from glob import glob

import matplotlib.pyplot as plt  
import numpy as np

from PIL import Image
import cv2


dataset_path = 'D:/IMAGEE SEG/4j/*.jpg'

set1 = glob(dataset_path)

#reading images of set1
#blue image
blue = Image.open(set1[0])
blue.size

#green image
green = Image.open(set1[1])
green.size

#red image
red= Image.open(set1[2])
red.size

#Pan image
pan = Image.open(set1[3])
pan.size

#code to display images
#blue image
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(blue,cmap='gray')
ax.axis('off')
plt.show()

#green image
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(green,cmap='gray')
ax.axis('off')
plt.show()

#red image
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(red,cmap='gray')
ax.axis('off')
plt.show()

#pan image
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(pan,cmap='gray')
ax.axis('off')
plt.show()



# Merge the RGB channels into a single image
multi = Image.merge('RGB', (red, green, blue))

# plot the merged image
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(multi)
ax.axis('off')
plt.show()

multi_w , multi_h = multi.size


#Resizing Panchromatic Image according to the multispectral
resized_pan = pan.resize((multi_w , multi_h))


#plotting resized panchromatic image
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(resized_pan,cmap='gray')
ax.axis('off')
plt.show()



#exporting the fused Multi-spectral image and resized Panchromatic Image
multi.save('D:/IMAGEE SEG/Processed imgs/4j_Multi.jpg')
resized_pan.save('D:/IMAGEE SEG/Processed imgs/4j_pan.jpg' )






















