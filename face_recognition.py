import cv2
import numpy as np
import os 

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = 'cascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

#Iniitiate Counter
id = 0

#Names related to ids in the training set
names = ['None', 'Tevin', 'Renelle']

#Initialize and start real-time video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) #Set Video Width 
cam.set(4, 480) # Set Video Height

#Defining minimum window size to be recognized as a face
minWidth = 0.1 * cam.get(3)
minHeight = 0.1 * cam.get(4)

while True:
	ret, img = cam.read()
	img = cv2.flip(img, -1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(
		gray, 
		scaleFactor = 1.2, 
		minNeighbors = 5, 
		minSize = (int(minWidth), int(minHeight))
	)
	
	for (x, y, w, h) in faces:
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
		id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

		if (confidence < 100):
			id = names[id]
			confidence = " {0}%".format(round(100 - confidence))
		else:
			id = "unknown"
			confidence = " {0}%".format(round(100 - confidence))
		
		cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255,255,255), 2)
		cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255,255,0), 1)
	
	cv2.imshow('camera', img)
	k = cv2.waitKey(10) & 0xff #Press ESC to exit
	if k == 27:
		break
	
print('Exiting program...')
cam.release()
cv2.destroyAllWindows()
		
