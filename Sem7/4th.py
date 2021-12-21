import cv2
import cv2 as cv
import numpy as np
from PIL import Image
from itertools import product
import os

def tile(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size
    grid = product(range(0, h - h % d, d), range(0, w - w % d, d))
    for i, j in grid:
        box = (j, i, j + d, i + d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)

def otsu(img, nome):
    histograma, limite = np.histogram(img, bins=256)
    limites = (limite[:-1] + limite[1:]) / 2.
    p1k = np.cumsum(histograma)
    p2k = np.cumsum(histograma[::-1])[::-1]
    m1k = np.cumsum((histograma*limites)) / p1k
    m2k = (np.cumsum((histograma*limites)[::-1]) / p2k[::-1])[::-1]
    delta_classes = p1k[:-1] * p2k[1:] * (m1k[:-1] - m2k[1:]) ** 2
    limiar_retorno = limites[:-1][np.argmax(delta_classes)]
    _, img_limiar = cv2.threshold(img, limiar_retorno, 255, cv2.THRESH_BINARY)
    cv2.imwrite("crops_out/"+nome.replace("crops/",""), img_limiar)

def limiarizar(imagens):
    for imagem in imagens:
        img = cv.imread(imagem, 0)
        cv.waitKey(0)
        otsu(img, imagem)


imagens = ["crops/Fig1039(a)(polymersomes)_0_0.png",
           "crops/Fig1039(a)(polymersomes)_0_216.png",
           "crops/Fig1039(a)(polymersomes)_0_432.png",
           "crops/Fig1039(a)(polymersomes)_216_0.png",
           "crops/Fig1039(a)(polymersomes)_216_216.png",
           "crops/Fig1039(a)(polymersomes)_216_432.png",
           "crops/Fig1039(a)(polymersomes)_432_0.png",
           "crops/Fig1039(a)(polymersomes)_432_216.png",
           "crops/Fig1039(a)(polymersomes)_432_432.png"]

limiarizar(imagens=imagens)