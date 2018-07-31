import cv2 
import numpy as np

from PIL import Image 
import os 

#Path for dataset
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

#Function to get the images and label data
def getImagesAndLabels(path):
	imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
	faceSamples = []
	ids = []
	for imagePath in imagePaths:
		PIL_img = Image.open(imagePath).convert('L') #Converting the image to gray
		img_numpy = np.array(PIL_img, 'uint8')
		id = int(os.path.split(imagePath)[-1].split('.')[1]) #Retrieving the ID
		faces = detector.detectMultiScale(img_numpy)
		for (x,y,w,h) in faces:
			faceSamples.append(img_numpy[y:y+h, x:x+w])
			ids.append(id)
	return faceSamples, ids

print('Training faces. Please wait...')
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

## Saving the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml')

##Printing the number of  faces trained
print('\n [INFO] {0} faces trained. Exiting Program...'.format(len(np.unique(ids))))
