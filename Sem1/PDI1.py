import cv2 as cv

imagem = cv.imread('cinza.png',cv.IMREAD_GRAYSCALE)

vet = []
for i in range(imagem.shape[0]):
    for j in range(imagem.shape[1]):
        if imagem[i][j] not in vet:
            vet.append(imagem[i][j])
print(len(vet))


