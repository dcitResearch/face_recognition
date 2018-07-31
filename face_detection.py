import numpy as np
import cv2 

faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml') #Cascade used to train model to recognize frontal faces
cap = cv2.VideoCapture(0) #Captruing video from the first camera detected on this device
cap.set(3,640) #Setting the Height
cap.set(4,480) #Setting the Width

while True:
	ret, img =  cap.read()
	img = cv2.flip(img, -1) #Flipping the image vertically
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Grayscale version of the capture
	faces = faceCascade.detectMultiScale(
		gray, #The input grayscale image
		scaleFactor=1.2, #Specifies how much the image size is reduced at each image scale
		minNeighbors=5, #How many neighbors each candidate rectangle should have (higher number gives lower false positives)
		minSize=(20, 20) #Minimum rectangle size to be considered a face
	)
	
	for (x,y,w,h) in faces: #If faces are found we return the positions of the detected faces (x & y left up corner, w & h height and width)
		cv2.rectangle(img,(x,y),(x+w, y+h), (255, 0, 0), 2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
	
	cv2.imshow('video', img)
	k = cv2.waitKey(30) & 0xff
	if k == 27: #Pressing escape to end the video capture
		break
cap.release()
cv2.destroyAllWindows()


