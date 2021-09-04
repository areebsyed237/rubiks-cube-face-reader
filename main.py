import cv2
import numpy as np
from helper import ciede2000, bgr2lab
import copy
import os

color_palette = {
            'red'   : (0, 0, 255),
            'orange': (0, 120, 255),
            'blue'  : (255, 0, 0),
            'green' : (0, 255, 0),
            'white' : (205, 205, 205),
            'yellow': (0, 205, 255)
        }
color_codes = {
            'red'   : 'R',
            'orange': 'O',
            'blue'  : 'B',
            'green' : 'G',
            'white' : 'W',
            'yellow': 'Y'
        }

path1 = "input_images/"
path2 = "working/"
path3 = "output/"
listing = os.listdir(path1)
for file in listing:
    image = cv2.imread(path1 + file)
    img = copy.deepcopy(image)

    print('Image : ',file)
    print('Original Dimensions : ',img.shape)
    while(img.shape[0]<420 or img.shape[0]>540):
        if (img.shape[0]<480):
            scale_percent = 110 # percent of original size
        else:
            scale_percent = 90 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    print('Resized Dimensions : ',img.shape)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("image1", gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    #cv2.imshow("image2", blurred)
    canny = cv2.Canny(blurred, 20, 40)
    #cv2.imshow("image3", canny)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(canny, kernel, iterations=8)
    #cv2.imshow("image4", dilated)
    #cv2.waitKey(0)

    (contours, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    hierarchy = hierarchy[0]

    index = 0
    pre_cX = 0
    pre_cY = 0
    center = []

    print('Facelet Dimensions (Perimeter, Area) :')
    for component in zip(contours, hierarchy):
        contour = component[0]
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.1 * peri, True)
        area = cv2.contourArea(contour)
        corners = len(approx)

        # compute the center of the contour
        M = cv2.moments(contour)

        if M["m00"]:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX = None
            cY = None

        if 2000 < area < 25000 and peri < 600 and cX is not None:
            tmp = {'index': index, 'cx': cX, 'cy': cY, 'contour': contour}
            center.append(tmp)
            index += 1

            print(peri, area)

    center.sort(key=lambda k: (k.get('cy', 0)))
    row1 = center[0:3]
    row1.sort(key=lambda k: (k.get('cx', 0)))
    row2 = center[3:6]
    row2.sort(key=lambda k: (k.get('cx', 0)))
    row3 = center[6:9]
    row3.sort(key=lambda k: (k.get('cx', 0)))

    center.clear()
    center = row1 + row2 + row3

    for component in center:
        candidates.append(component.get('contour'))

    def get_pix(num,a,b):
        return int(img[center[num]['cy']+a][center[num]['cx']+a][b])

    color_list = []
    color_id_list = []

    print("Facelet Colors (BGR) : ")
    for num in range(index):
        if num==9:
            break
        try:
            bgr=(int((get_pix(num,0,0)+get_pix(num,3,0)+get_pix(num,-3,0)+get_pix(num,5,0)+get_pix(num,-5,0))/5.0),
                int((get_pix(num,0,1)+get_pix(num,3,1)+get_pix(num,-3,1)+get_pix(num,5,1)+get_pix(num,-5,1))/5.0),
                int((get_pix(num,0,2)+get_pix(num,3,2)+get_pix(num,-3,2)+get_pix(num,5,2)+get_pix(num,-5,2))/5.0))
        except IndexError:
            pass
        print(bgr)
        
        lab = bgr2lab(bgr)
        distances = []
        for color_name, color_bgr in color_palette.items():
            distances.append({
                'color_name': color_name,
                'color_bgr': color_bgr,
                'distance': ciede2000(lab, bgr2lab(color_bgr))
                })
        closest = min(distances, key=lambda item: item['distance'])
        if closest['color_name'] not in color_list:
            color_list.append(closest['color_name'])
        val = color_list.index(closest['color_name'])+1
        try:
            cv2.putText(img, "{}({})".format(color_codes[closest['color_name']],val), (center[num]['cx']-20,center[num]['cy']), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        except IndexError:
            pass
        color_id_list.append(val)

    print('Face unique color ids : ')
    print(color_id_list, '\n')
    cv2.drawContours(img, candidates, -1, (0, 255, 0), 3)
    #cv2.imshow("image5", img)
    #cv2.waitKey(0)

    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    dilated = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
    col1 = cv2.vconcat([gray,dilated])
    col2 = cv2.vconcat([canny,img])
    collage = cv2.hconcat([col1, col2])

    filename = os.path.splitext(file)[0]
    cv2.imwrite(path2+filename+"_working.jpg", collage)
    with open(path3+"output_"+filename+".txt", 'w') as f:
        for i in range(9):
            f.write('%d' %color_id_list[i])
            if (i+1)%3 == 0:
                f.write('\n')
    