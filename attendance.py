import cv2
import face_recognition

import numpy as np
import pyttsx3
import speech_recognition as sr
import os 
from datetime import datetime


engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
#print(voices[0].id)
engine.setProperty('voice', voices[0].id)

def speak(audio):
	engine.say(audio)
	engine.runAndWait()

# path of image folder
path = 'images'
# list of all images array
images = []
# array of persone name
personName = []
# this will list all images in images folder
myList = os.listdir(path)
#print(myList)

# to grab person name from image name by spiliting image name from extension
for cu_img in myList:
	current_img = cv2.imread(f'{path}/{cu_img}')
	images.append(current_img)
	personName.append(os.path.splitext(cu_img)[0])
#print(personName)

# function to encode images
def faceEncodings(images):
	encodeList = []
	for img in images:
		# converted default BGR to RGB
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		encode = face_recognition.face_encodings(img)[0]
		encodeList.append(encode)
	return encodeList

def attendance(name):
	# read xl file
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        # if entry already exist
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')
            speak("Hello")
            speak(name)
        else:
        	welcome = "Welcome Back", name
        	speak(welcome)
            


receivedEncodings = faceEncodings(images)
print('All Encodings Complete!!!')

# capture videos - NOTE - 1 = external cam and 0 = laptop that is internal cam
cap = cv2.VideoCapture(1)

while True:
	ret, frame = cap.read()
	faces = cv2.resize(frame, (0,0) , None, 0.25, 0.25)
	faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

	# this is going to search for faces only and no other object
	currentFrame = face_recognition.face_locations(faces)
	encodeCurrentFrame = face_recognition.face_encodings(faces,currentFrame)

	for encodeFace, faceLocation in zip(encodeCurrentFrame, currentFrame):
		matches = face_recognition.compare_faces(receivedEncodings, encodeFace)
		faceDistance = face_recognition.face_distance(receivedEncodings, encodeFace)
		matchIndex = np.argmin(faceDistance)

		if matches[matchIndex]:
			name = personName[matchIndex].upper()
			#print(name)
			y1,x2,y2,x1 = faceLocation
			y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
			# rectengle around face
			cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0) ,2)
			# rectengle to display name on image
			cv2.rectangle(frame, (x1,y2-35),(x2,y2), (0,255,0) ,cv2.FILLED)
			cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)
			attendance(name) # called attendance function


	cv2.imshow("Camera", frame)

	if cv2.waitKey(10) == 13:
		break
cap.release()
cv2.destroyAllWindows()

