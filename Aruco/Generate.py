### Generate markers
import cv2
from cv2 import aruco
import os

#########install following
#pip install opencv-python==4.6.0.66 
#pip install opencv-contrib-python==4.6.0.66

### --- parameter --- ###

# Save location
dir_mark = r'G:\sem 7\FYP\Git\FYP-Transformer-Robot\Aruco'

# Parameter
num_mark = 15 #Number of markers
size_mark = 1000 #Size of markers

### --- marker images are generated and saved --- ###
# Call marker type
dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)

for count in range(num_mark) :

    id_mark = count
    img_mark = aruco.drawMarker(dict_aruco, id_mark, size_mark)

    if count < 10 :
        img_name_mark = 'mark_id_0' + str(count) + '.jpg'
    else :
        img_name_mark = 'mark_id_' + str(count) + '.jpg'
    path_mark = os.path.join(dir_mark, img_name_mark)

    cv2.imwrite(path_mark, img_mark)