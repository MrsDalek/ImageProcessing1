import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('kiraz1.jpg')
imge = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

value = input("Saymak istediğiniz meyve rengini giriniz.")
value = value.lower()

if value == 'turuncu':
    lower = np.array([5, 90, 80])
    upper = np.array([30, 255, 255])
elif value == 'kırmızı':
    lower = np.array([160, 100, 139])
    upper = np.array([179, 255, 255])
elif value == 'sarı':
    lower = np.array([0, 101, 139])
    upper = np.array([224, 255, 255])
elif value == 'yeşil':
    lower = np.array([38, 100, 100])
    upper = np.array([75, 255, 255])

mask = cv2.inRange(img_hsv, lower, upper)
res = cv2.bitwise_and(img, img, mask=mask)
res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (11, 11), 0)
img_blur2 = cv2.medianBlur(img_blur, 5)
img_blur3 = cv2.bilateralFilter(img_blur2, 9, 75, 75)
_, th = cv2.threshold(img_blur3, 0, 255, cv2.THRESH_OTSU)
img_canny = cv2.Canny(th, 0, 130, 3)
img_dilated = cv2.dilate(img_canny, (5, 5), 2)
contours, hierarchy = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img_rgb, contours, -1, (0, 255, 0), 2)

print('Görüntüdeki meyve sayısı:', len(contours))

titles = ['image', 'Masked', 'Res', 'Gray', 'blur', 'Threshold', 'canny', 'dilated', 'Contour']
images = [imge, mask, res, img_gray, img_blur3, th, img_canny, img_dilated, img_rgb]

for i in range(9):
    plt.subplot(3, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
"""
titles2 = ['Gray', 'Gaussian', 'Median', 'Bilater']
images2 = [img_gray, img_blur, img_blur2, img_blur3]

for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images2[i], 'gray')
    plt.title(titles2[i])
    plt.xticks([]), plt.yticks([])

plt.show()


b,g,r = cv2.split(img)
b = cv2.equalizeHist(b)
g = cv2.equalizeHist(g)
r = cv2.equalizeHist(r)
img_eqHis = cv2.merge((b, g, r))

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
img_sharp = cv2.filter2D(img_blur, -1, kernel)
#img_contrast = cv2.addWeighted(img, 2.0, np.zeros(img.shape, img.dtype), 0,0)
"""
