import cv2
path = "ISIC_0000276.jpg"
melanoma = cv2.imread(path)
import numpy as np
import os
import morphsnakes
from matplotlib import pyplot as ppl
import PIL
import ImageCompare

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def centeredCrop(img):
   width =  np.size(img,1)
   height =  np.size(img,0)
   left = np.ceil((width - width/2)/2)
   left = int(left)
   top = np.ceil((height - height/2)/2)
   top = int(top)
   right = np.floor((width + width/2)/2)
   right = int(right)
   bottom = np.floor((height + height/2)/2)
   bottom = int(bottom)
   cImg = img[top:bottom, left:right]
   return cImg

def circle_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T)**2, 0))
    u = np.float_(phi > 0)
    return u

def RemoveBackground(Image):
    IGray=cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
    ret,BW=cv2.threshold(IGray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #ret1,thresh1 = cv2.threshold(th,127,255,cv2.THRESH_BINARY)
    ret2,thresh2 = cv2.threshold(BW,127,255,cv2.THRESH_BINARY_INV)
    maskedImage=cv2.bitwise_and(Image,Image,mask=thresh2)
    return maskedImage,IGray,thresh2

def change_contrast(img, level):
    slika = PIL.Image.fromarray(img)
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return np.array(slika.point(contrast))



# cnt = 0
# for image in os.listdir('ValidationImages'):
#     a,b = image.split('.')
#     if (b == 'jpg'):
#         cnt += 1
#         img = cv2.imread('ValidationImages/'+image)
#
#         #img = change_contrast(img, 100)
#         img = centeredCrop(img)
#         img = cv2.resize(img, (768,560))
#         cv2.imwrite("ResizedImages/"+image, img)
#         BGImage, A, th = RemoveBackground(img)
#         kernel = np.ones((5, 5), np.uint8)
#         img = cv2.erode(BGImage, kernel=kernel, iterations=1)
#         img = cv2.dilate(img, kernel=kernel, iterations=1)
#         img = cv2.medianBlur(img, ksize=3)
#         cv2.imwrite('SegmentedImagesValid/'+image, img)
#         print(cnt)

# cnt = 0
# for image in os.listdir('ValidationImages'):
#     a,b = image.split('.')
#     if (b == 'jpg'):
#         cnt += 1
#         img = cv2.imread('ValidationImages/'+image)
#         img = centeredCrop(img)
#         img = cv2.resize(img, (768,560))
#         cv2.imwrite("ResizedImages/"+image, img)
#         BGImage, A, th = RemoveBackground(img)
#         kernel = np.ones((5, 5), np.uint8)
#         img = cv2.erode(th, kernel=kernel, iterations=1)
#         img = cv2.dilate(img, kernel=kernel, iterations=1)
#         img = cv2.medianBlur(img, ksize=3)
#         cv2.imwrite('SegmentedImagesThresh/'+a + '_segmentation.' + b, img)
#         print(cnt)
#
# cnt = 0
# for image in os.listdir('SegmentedImagesValid'):
#     a, b = image.split('.')
#     if (b == 'jpg'):
#         cnt += 1
#         imgCol = cv2.imread('SegmentedImagesValid/' + image)
#         img = cv2.cvtColor(imgCol, cv2.COLOR_BGR2GRAY)
#
#         # MorphACWE does not need g(I)
#         # Morphological ACWE. Initialization of the level-set.
        #macwe = morphsnakes.MorphACWE(img, smoothing=3, lambda1=1, lambda2=1)
#         macwe.levelset = circle_levelset(img.shape, (280, 384), 280)
#         # Visual evolution.
#         figure = ppl.figure()
#         morphsnakes.evolve_visual(macwe, num_iters=190, background=imgCol)
#         a, b = image.split('.')
#         filename = a+'.png'
#         figure.savefig('SegmentedImagesValidMorph/' + filename)
#         print(cnt)

imageCount = 0
sum = 0
#print('Otsu cumulative error: %.2f', err)
for image in os.listdir('ImagesGroundTruth'):
    a, b = image.split('.')
    groundTruthImage = cv2.imread('ImagesGroundTruth/' + image)
    groundTruthImage = centeredCrop(groundTruthImage)
    groundTruthImage = cv2.resize(groundTruthImage, (768, 560))
    cv2.imwrite('ResizedImagesGroundTruth/' + image, groundTruthImage)
    segmentedImage = cv2.imread('SegmentedImagesThresh/' + a + '.jpg')
    im1 = PIL.Image.fromarray(groundTruthImage)
    im2 = PIL.Image.fromarray(segmentedImage)
    accuracy = ImageCompare.PixelCompare(im1=im1, im2=im2, mode='pct')
    sum += accuracy[0]
    imageCount += 1
    print(imageCount)
err = sum / imageCount
print('Otsu cumulative error: ', err)
