import cv2
import numpy as np

mask_hori = np.array(
    [[1, 2, 1],
     [0, 0, 0],
     [-1, -2, -1]]
)

mask_vert = np.array(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]]
)

mask_45 = np.array(

    [[0, 1, 2],
     [-1, 0, 1],
     [-2, -1, 0]]
)
mask_135 = np.array(
    [[-2, -1, 0],
     [-1, 0, 1],
     [0, 1, 2]])

mask_30 = np.array(
    [[1, 1, 1, 1, 0],
     [1, 1, 0, 0, 0],
     [0, 0, 0, -1, -1],
     [0, -1, -1, -1, -1]])
masks = [-mask_hori, -mask_vert, -mask_45,-mask_135, -mask_30]

for m in masks:

    img = cv2.imread('imagens/Fig1007(a)(wirebond_mask).png', )
    img_mask =cv2.filter2D(img,-1, kernel = m)
    cv2.imshow(str(m), img_mask)
    cv2.waitKey(0)