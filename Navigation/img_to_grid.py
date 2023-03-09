import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

img = cv.imread("Warehouse-Layout.png")
img = img[72:57*12, 13*12:76*12]
#cv.imshow("originl_img", img)
img_size = img.shape[:2]
resized_img = cv.resize(img, (img_size[1] // 12, img_size[0] // 12), interpolation = cv.INTER_CUBIC)
gray_img = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
#cv.imshow("gray_img", gray_img)

#img2 = cv.Laplacian(gray_img, cv.CV_16SC1)
blured_img = cv.GaussianBlur(gray_img, (5, 5), cv.BORDER_DEFAULT)
#canny_img = cv.Canny(blured_img, 75, 75)
#img2 = cv.Laplacian(blured_img, cv.CV_16SC1)
#cv.imshow("canny_img",canny_img)
# Resize to fill the whole screen
#cv.imshow("laplacian_img", img2)
#cv.imshow("blured_img", blured_img)
#ret3,th3 = cv.threshold(blured_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
ret, thresh = cv.threshold(gray_img, 90, 255, cv.THRESH_BINARY)
#thresh = cv.adaptiveThreshold(gray_img, 150, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 199, 37)

#cv.imshow("threshold", thresh)
plt.imshow(thresh)
plt.show()

def bool_mat(mat):
    new_mat = np.zeros(mat.shape, dtype=bool)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            new_mat[i, j] = not bool(mat[i, j])
    return new_mat

maze  = bool_mat(thresh)

np.savetxt("maze.txt", maze, delimiter=',')

start = (0, 0)
end = (17, 30)

m = np.zeros(maze.shape)
m[start] = 1

r, c = maze.shape

def makeStep(k):
  for i in range(r):
    for j in range(c):
      if m[i, j] == k:
        if i-1 > 0 and m[i-1, j] == 0 and maze[i-1, j] == 0:
          m[i-1, j] = k + 1
        if i+1 < r and m[i+1, j] == 0 and maze[i+1, j] == 0:
          m[i+1][j] = k + 1
        if j-1 > 0 and m[i, j-1] == 0 and maze[i, j-1] == 0:
          m[i, j-1] = k + 1
        if j+1 < c and m[i, j+1] == 0 and maze[i, j+1] == 0:
          m[i, j+1] = k + 1
          
print(maze.shape)



k = 0
while m[end] == 0:
    k += 1
    makeStep(k)

print(m)

i, j = end
k = m[i, j]
path = [(i,j)]
while k > 1:
  if i > 0 and m[i - 1, j] == k-1:
    i, j = i-1, j
    path.append((i, j))
    k-=1
  elif j > 0 and m[i, j - 1] == k-1:
    i, j = i, j-1
    path.append((i, j))
    k-=1
  elif i < r - 1 and m[i + 1, j] == k-1:
    i, j = i+1, j
    path.append((i, j))
    k-=1
  elif j < c - 1 and m[i, j + 1] == k-1:
    i, j = i, j+1
    path.append((i, j))
    k-=1

path = path[::-1]
print(path)

backtorgb = cv.cvtColor(thresh,cv.COLOR_GRAY2RGB)


path_arr = np.zeros([r, c])
for cord in path:
    path_arr[cord] = 1
    backtorgb[cord] = (0, 255, 0)
    

np.savetxt("solved_maze.txt", path_arr, delimiter=',')

#cv.imshow("solved_img", backtorgb)

'''resized_img_2 = cv.resize(backtorgb, (thresh.shape[1] * 25, thresh.shape[0] * 25), interpolation = cv.INTER_CUBIC)

cv.imshow("final_resized", resized_img_2)'''



'''
contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


processed_img = np.zeros(gray_img.shape[:2], dtype = "uint8")

cv.drawContours(image=processed_img, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=2, lineType=cv.LINE_AA)
#cv.fillPoly(gray_img, contours, color=(0, 0, 255))
#cv.drawContours(processed_img, contours, -1, color=(255, 255, 255), thickness=cv.FILLED)
cv.imshow("processed_img", processed_img)

cv.imshow("threshold_img", thresh)

print("Number of contours is %d" % len(contours))
'''

plt.imshow(backtorgb)
plt.show()
cv.waitKey(0)
