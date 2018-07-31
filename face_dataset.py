import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) #Setting video width
cam.set(4, 480) #Setting video height

#Cascade
face_detector = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

#Enter one numeric face id for each person
face_id = input('\nEnter user ID and press enter: ')
print('Look at the camera and wait..')

#Initialize individual sampling face count
count = 0
while(True):
	ret, img = cam.read()
	img = cv2.flip(img, -1) #Flipping the image vertically
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Grayscale version of image
	faces = face_detector.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
		count += 1
		
		#Saving the captured image in the dataset folder
		cv2.imwrite('dataset/User.' + str(face_id) + '.' + str(count) + '.jpg', gray[y:y+h, x:x+w])
		cv2.imshow('image', img)
	k = cv2.waitKey(100) & 0xff 
	if k == 27:
		break
	elif count >= 30: #Take a maximum of 30 sample shots
		break

print('Exiting progam...')
cam.release()
cv2.destroyAllWindows()
