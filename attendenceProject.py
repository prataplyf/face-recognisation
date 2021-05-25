import os
import cv2
import numpy as np
import icecream as ic
import face_recognition
from datetime import datetime

global userName
userName = []

# path of the image dir. of all students/people
path = "ImageAttendance"
images = []
className = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImage = cv2.imread(f'{path}/{cl}')  # current image from ImageAttendance Folder
    images.append(curImage)
    className.append(os.path.splitext(cl)[0])

print(className)


def findEncodings(encodeImage):
    encodeList = []
    for img in encodeImage:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
print("Encoding Complete.")

def markAttendance(name):
    global userName
    with open("Attendance.csv", "r+") as f:
        myDataList = f.readline()
        print("_"*100)
        print(myDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in userName:
            userName.append(name)
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name}, {dtString}')


webCam = cv2.VideoCapture(0)
while True:
    success, img = webCam.read()
    imgScanSmall = cv2.resize(img, (0, 0), None, 0.25,
                              0.25)  # resizing the image size up to 1/4th of the actual(WebCam) image
    imgScanSmall = cv2.cvtColor(imgScanSmall, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgScanSmall)
    encodeCurFrame = face_recognition.face_encodings(imgScanSmall, facesCurFrame)

    #  finding all the matches
    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = className[matchIndex].upper()
            # print("WebCam Person Name: ", name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    # press 'esc' to close program
    if cv2.waitKey(1) == 27:
        break
#release camera
webCam.release()
cv2.destroyAllWindows()
