import cv2
import numpy as np
from matplotlib import pyplot as plt

def contrast_straching(image, niza_tocki):
    no_rows=image.shape[0]
    no_cols=image.shape[1]
    niza_tocki=sorted(sorted(niza_tocki, key=lambda tocka: tocka[1]), key=lambda tocka: tocka[0])
    #print(niza_tocki)
    nova_slika = np.zeros((no_rows, no_cols), np.uint8)
    if len(niza_tocki)==1:
        # print('Dva intervali')
        prva_tocka=niza_tocki[0]
        for i in range(0, no_rows):
            for j in range(0, no_cols):
                if image[i,j] < prva_tocka[0]:
                    k=(prva_tocka[1])/(prva_tocka[0])
                    nova_slika[i,j]=k*image[i,j]
                else:
                    k=(255-prva_tocka[1])/(255-prva_tocka[0])
                    nova_slika[i,j]=(k*(image[i,j]-prva_tocka[0])+prva_tocka[1])%255
        return image
    else:
        # print('Poveke intervali')
        for i in range(0, no_rows):
            for j in range(0, no_cols):
                prva_tocka=niza_tocki[0]
                a = image[i,j] >= 0
                b = image[i,j] < prva_tocka[0]
                if (a ^ b).all():
                    if prva_tocka[0] != 0:
                        k = prva_tocka[1] / prva_tocka[0]
                        nova_slika[i,j]=k*image[i,j]
                else:
                    for index in range(1, len(niza_tocki)):
                        leva_tocka=niza_tocki[index-1]
                        desna_tocka=niza_tocki[index]
                        if image[i,j]>=leva_tocka[0] and image[i,j]<desna_tocka[0]:
                            k=(desna_tocka[1]-leva_tocka[1])/(desna_tocka[0]-leva_tocka[0])
                            nova_slika[i,j]=k*(image[i,j]-leva_tocka[0])+leva_tocka[1]
                    posledna_tocka=niza_tocki[len(niza_tocki)-1]
                    if image[i,j]>=posledna_tocka[0] and image[i,j]<=255:
                        k = (255 - posledna_tocka[1]) / (255 - posledna_tocka[0])
                        nova_slika[i,j] = k * (image[i, j] - posledna_tocka[0]) + posledna_tocka[1]
        return nova_slika



# image=cv2.imread('Tom_and_Jerry.png', 0)
# plt.figure()
# plt.subplot(121)
# plt.title('Original image')
# plt.imshow(image, cmap='gray')
# plt.subplot(122)
# points=[(0,120), (120, 150), (150, 180), (180, 230), (210, 250)]
# ret_image=contrast_straching(image,  points)
# plt.title('Contrast streched image')
# plt.imshow(ret_image, cmap='gray')
# plt.show()