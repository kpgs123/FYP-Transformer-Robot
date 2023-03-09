import numpy as np
import math
import cv2 as cv

r, c = 400, 400
size = r, c

start = (0, 0)
end = (25, 40)

openList = []
g_dict = {}
h_dict = {}
f_dict = {}
parent_dict = {}

for i in range(r):
    for j in range(c):
        openList.append((i, j))
        g_dict[(i, j)] = float('inf')
        h_dict[(i, j)] = float('inf')
        f_dict[(i, j)] = float('inf')
closedList = []
g_dict[start] = 0
h_dict[start] = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
f_dict[start] = g_dict[start] + h_dict[start]

while (len(openList)):
    # find the minimum node which has minimum f value
    q = min(f_dict, key=f_dict.get)
    openList.remove(q)
    del f_dict[q]
    i, j = q
    lst = []
    if (i - 1 >= 0) and (j - 1 >= 0):
        parent_dict[(i - 1, j - 1)] = q
        lst.append((i-1, j-1))
    if (i -1 >= 0):
        parent_dict[(i - 1, j)] = q
        lst.append((i-1, j))
    if (i - 1 >= 0) and (j + 1 < c):
        parent_dict[(i - 1, j + 1)] = q
        lst.append((i-1, j+1))
    if (j + 1 < c):
        parent_dict[(i, j + 1)] = q
        lst.append((i, j+1))
    if (i + 1 < r) and (j + 1 < c):
        parent_dict[(i + 1, j + 1)] = q
        lst.append((i+1, j+1))
    if (i + 1 < r):
        parent_dict[(i + 1, j)] = q
        lst.append((i+1, j))
    if (i + 1 < r) and (j - 1 >= 0):
        parent_dict[(i + 1, j - 1)] = q
        lst.append((i+1, j-1))
    if (j - 1 >= 0):
        parent_dict[i, j - 1] = q
        lst.append((i, j-1))

    if end in lst:
        break

    for successor in lst:
        if successor in [(i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)]:
            distance_q_succ = math.sqrt(2)
        else:
            distance_q_succ = 1
        g_dict_temp = g_dict[q] + distance_q_succ
        h_dict_temp = math.sqrt((end[0] - successor[0]) ** 2 + ((end[1] - successor[1]) ** 2))
        f_dict_temp = g_dict_temp + h_dict_temp

        if successor in f_dict.keys():
            if f_dict[successor] > f_dict_temp:
                g_dict[successor] = g_dict_temp
                h_dict[successor] = h_dict_temp
                f_dict[successor] = f_dict_temp
        if successor not in closedList:
            openList.append(successor)
    closedList.append(q)
#print(openList)
#print(closedList)

blank_image = np.zeros((r,c,3), np.uint8)
parent = end
print(parent_dict)
while(parent != start):
    blank_image[parent] = 0,0,255
    #print(parent)
    parent = parent_dict[parent]

cv.imshow("Path", blank_image)
cv.waitKey(0)