import cv2
import matplotlib.pyplot as plt  
import numpy as np

#Reading The images
multi= cv2.imread('D:/IMAGEE SEG/Processed imgs/4j_Multi.jpg')
multi = cv2.cvtColor(multi , cv2.COLOR_BGR2RGB)
panc = cv2.imread('D:/IMAGEE SEG/Processed imgs/4j_pan.jpg',0)

clahe = cv2.createCLAHE(4)
msr = clahe.apply(multi[:,:,0])
msg = clahe.apply(multi[:,:,1])
msb = clahe.apply(multi[:,:,2])
multi = np.stack([msr,msg,msb],axis=2)
panc = clahe.apply(panc)


#plotting the images
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(multi)
ax.axis('off')
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(panc,cmap='gray')
ax.axis('off')
plt.show()

# Create a SIFT object
sift = cv2.SIFT_create()

# Detect and compute keypoints and descriptors for both images
pan_kp, pan_des = sift.detectAndCompute(panc, None)
ms_kp, ms_des = sift.detectAndCompute(multi, None)

# Match the keypoints between the images
bf = cv2.BFMatcher()
matches = bf.knnMatch(pan_des, ms_des, k=2)

# Apply ratio test to select good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)


# Extract the matched keypoints in both images
pan_pts = np.float32([pan_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
ms_pts = np.float32([ms_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Calculate the transformation matrix using RANSAC
M, mask = cv2.findHomography(ms_pts, pan_pts, cv2.RANSAC, 5.0)

# Warp the multispectral image to align it with the panchromatic image
aligned_ms_image = cv2.warpPerspective(multi, M, (panc.shape[1], panc.shape[0]))

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(aligned_ms_image)
ax.axis('off')
plt.show()

plt.imsave('D:/IMAGEE SEG/Registered imgs/4j_Reg_Multi.jpg',aligned_ms_image)
cv2.imwrite('D:/IMAGEE SEG/Registered imgs/4j_Reg_pan.jpg',panc )
