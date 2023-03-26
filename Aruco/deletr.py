import cv2
from cv2 import aruco 
import os

loc = r'C:\test'

mark=2
size = 20

output = aruco.Dictionary_get(aruco.DICT_4X4_50)


print(output)