import cv2
import cv2 as cv
import numpy as np

im1 = cv.imread('imagens/Fig0228(a)(angiography_mask_image).png',cv.IMREAD_GRAYSCALE)
im2 = cv.imread('imagens/Fig0228(b)(angiography_live_ image).png',cv.IMREAD_GRAYSCALE)

def correcao(img):
    maxi = np.amax(img)
    mini = np.amin(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = (255*(img[i][j]-mini))/(maxi-mini)
    return img

# Load the image
img = cv2.imread('D:/downloads/forest.jpg')
# Check the datatype of the image
print(img.dtype)
# Subtract the img from max value(calculated from dtype)
img_neg = 255 - img
# Show the image
cv2.imshow('negative',img_neg)
cv2.waitKey(0)