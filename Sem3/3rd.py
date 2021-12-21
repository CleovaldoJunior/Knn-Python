import cv2 as cv
import numpy as np

filament = cv.imread('imagens/Fig0229(a)(tungsten_filament_shaded).png',cv.IMREAD_GRAYSCALE)
sensor = cv.imread('imagens/Fig0229(b)(tungsten_sensor_shading).png',cv.IMREAD_GRAYSCALE)

def correcao(img):
    maxi = np.amax(img)
    mini = np.amin(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = (255*(img[i][j]-mini))/(maxi-mini)
    return img

img_top = filament/sensor
corrigida = correcao(img_top)
cv.imshow("img", corrigida)
cv.waitKey(0)

