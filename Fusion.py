import numpy as np
import cv2
import matplotlib.pyplot as plt 

# Load the multispectral image
multispectral_img = cv2.imread('D:/IMAGEE SEG/Registered imgs/4j_Reg_Multi.jpg')

# Load the panchromatic image
panchromatic_img = cv2.imread('D:/IMAGEE SEG/Registered imgs/4j_Reg_pan.jpg')

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(multispectral_img)
ax.axis('off')
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(panchromatic_img)
ax.axis('off')
plt.show()

# Convert the multispectral image to HSI color space
multispectral_hsi = cv2.cvtColor(multispectral_img, cv2.COLOR_BGR2HSV)

# Convert the panchromatic image to grayscale
panchromatic_gray = cv2.cvtColor(panchromatic_img, cv2.COLOR_BGR2GRAY)

# Upsample the grayscale panchromatic image
upsampled_panchromatic = cv2.resize(panchromatic_gray, (multispectral_img.shape[1], multispectral_img.shape[0]), interpolation=cv2.INTER_CUBIC)

# Replace the intensity component of the multispectral image with the upsampled grayscale panchromatic image
fused_hsi = np.copy(multispectral_hsi)
fused_hsi[:,:,2] = upsampled_panchromatic

fused_hsi[:,:,2] = fused_hsi[:,:,2]*1.2
fused_hsi[:,:,1] = fused_hsi[:,:,1]*1.2
# Convert the fused HSI image back to RGB color space
fused_rgb = cv2.cvtColor(fused_hsi, cv2.COLOR_HSV2BGR)
fused_rgb = cv2.cvtColor(fused_rgb , cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(fused_rgb)
ax.axis('off')
plt.show()



# Save the fused image
plt.imsave('D:/IMAGEE SEG/Registered imgs/4j_fused.jpg',fused_rgb )




