import cv2
import numpy as np
import math
kernel = np.ones((5, 5), np.uint8)

#cv2.imshow(img)
#Reading the image and making the mask
img= cv2.imread('22.png',-1)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([0, 0, 60], dtype = "uint8")
upper = np.array([20 , 150, 254], dtype = "uint8")
skinmask=cv2.inRange(hsv,lower,upper)

skinmask = cv2.erode(skinmask, kernel, iterations=2)
#cv2.imshow('blueMask',blueMask)
skinmask = cv2.morphologyEx(skinmask, cv2.MORPH_OPEN, kernel)
#cv2.imshow('blueMask',blueMask)
skinmask = cv2.dilate(skinmask, kernel, iterations=1)
#Finding the hull and the contours

cnt, hierarchy = cv2.findContours(skinmask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#contour_sizes = [(cv2.contourArea(cnt), cnt) for contour in cnt]
#countour_max = max(contour_sizes, key=lambda x: x[0])[1]
c = max(cnt, key = cv2.contourArea)#contour with max area
topmost = tuple(c[c[:,:,1].argmin()][0])#topmost point of contour

bottommost = tuple(c[c[:,:,1].argmax()][0])#bottommost point of contour
dst= math.sqrt((topmost[0]-bottommost[0])**2 + (topmost[1]-bottommost[1])**2)#Distance between topmost and bottommost points
#print('dst =',dst)
cnt1 = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)#approximating the contour
#img = cv2.drawContours(img, [cnt1], 0, (0,255,0), 3)
#print(cnt1)
hull = cv2.convexHull(c)#finding the hull
area = cv2.contourArea(c)#finding the area of the contour
hull_area=cv2.contourArea(hull)#finding the hull area
ar=((hull_area-area)/area)*100#finding the sacred ratio
#print(area,hull_area,ar)
#M = cv2.moments(c)
#cx = int(M['m10']/M['m00'])
#cy = int(M['m01']/M['m00'])
#print(cx,cy)
#Finding the defects
l=0
x,y,w,h = cv2.boundingRect(cnt1)
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
hull = cv2.convexHull(cnt1,returnPoints=False)
defects = cv2.convexityDefects(cnt1,hull)#defects are found here
for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt1[s][0])
    end = tuple(cnt1[e][0])
    far = tuple(cnt1[f][0])
     # find length of all sides of triangle
    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
    s = (a+b+c)/2
    ar1 = math.sqrt(s*(s-a)*(s-b)*(s-c))

    #distance between point and convex hull
    d=(2*ar1)/a

    # apply cosine rule here
    angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57


    # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
    if angle <= 90 and d>30:
        l += 1
        cv2.circle(img, far, 3, [255,255,0], -1)

    #draw lines around hand
    cv2.line(img,start, end, [0,255,0], 2)
    #cv2.line(img,start,end,[0,255,0],2)
    #cv2.circle(img,far,5,[0,0,255],-1)
l=l+1

#Using the defects to find the appropriate sign translation

font = cv2.FONT_HERSHEY_SIMPLEX
if l==1:#if there are no defects
    if area<2000:#if there was no hand
        cv2.putText(frame,'NO HAND',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    else:
        #if there is no contour there can be 4 translations
        if dst<250:#as there is a comparably small distance for a closed fist
            print('value = 0')
        #The other values are found by testing with other various images
        elif ar<15:
            print('value = 9')
        elif ar<25:
            print('value = 6')

        else:
            print('value = 1')

elif l==2:#if there is one defect
    if ar>39:
        print('value = 7')
    else:
        print('value = 2')

elif l==3:#if there are 2 defects

    if ar<40:
        print('value = 3')
    else:
        print('value = 8')

elif l==4:#if there are 3 defects
        print('value = 4')

elif l==5:#if there are 4 defects
        print('value = 5')

#displaying the results
cv2.imshow('img',img)
cv2.imshow('skinmask',skinmask)
cv2.waitKey(0)
cv2.destroyAllWindows()
