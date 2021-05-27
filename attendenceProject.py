import os
import cv2
import glob
import numpy as np
import pandas as pd
import icecream as ic
import face_recognition
from datetime import datetime

userNameList = []

# path of the image dir. of all students/people
# path = "ImageAttendance"
path = "allImages"
# path = "croppedImages"
images = []
className = []
myList = os.listdir(path)
# print(myList)
for i in range(0, len(myList)):
    print(i + 1, ' : ', myList[i])
print(len(myList))

for cl in myList:
    curImage = cv2.imread(f'{path}/{cl}')  # current image from ImageAttendance Folder
    images.append(curImage)
    className.append(os.path.splitext(cl)[0])

print(className)
print(len(className))

print("Encoding all Images...")


def findEncodings(encodeImage):
    count = 0
    encodeList = []
    for img in encodeImage:
        count += 1
        print('-' * 10)
        print("count: ", count)
        print(len(img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(len(img))
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
print("Done.")
# exit()

def markAttendance(name):
    global userNameList
    df = pd.read_csv("Attendance.csv")
    nameList = list(df['Name'])
    # print(nameList)
    if name not in nameList and name not in userNameList:
        userNameList.append(name)
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        with open('Attendance.csv', 'a') as fd:
            fd.write(f'\n{name}, {dtString}')


##########################################################################################################
##########################################################################################################
##########################################################################################################
#   ## Image Recognize via WebCam
##########################################################################################################
##########################################################################################################
##########################################################################################################
# webCam = cv2.VideoCapture(1)
# while True:
#     success, img = webCam.read()
#     imgScanSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # resizing the image size up to 1/4th of the actual(WebCam) image
#     imgScanSmall = cv2.cvtColor(imgScanSmall, cv2.COLOR_BGR2RGB)
#     facesCurFrame = face_recognition.face_locations(imgScanSmall)
#     encodeCurFrame = face_recognition.face_encodings(imgScanSmall, facesCurFrame)
#
#     #  finding all the matches
#     for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
#         matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
#         faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
#         matchIndex = np.argmin(faceDis)
#         if matches[matchIndex]:
#             name = className[matchIndex].upper()
#             # print("WebCam Person Name: ", name)
#             y1, x2, y2, x1 = faceLoc
#             y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
#             cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
#             cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
#             markAttendance(name)
#
#     cv2.imshow('Webcam', img)
#     # cv2.waitKey(0)
#     # press 'esc' to close program
#     if cv2.waitKey(1) == 27:
#         break
#
# # release camera
# webCam.release()
# cv2.destroyAllWindows()


##########################################################################################################
##########################################################################################################
##########################################################################################################
#   ## Image Recognize via Image
##########################################################################################################
##########################################################################################################
##########################################################################################################

# img = face_recognition.load_image_file("cskGroupImage.jpg")
def imageTesting(image):
    img = cv2.imread("testImages/" + image)
    imgScanSmall = cv2.resize(img, (0, 0), None, 1,
                              1)  # resizing the image size up to 1/4th of the actual image
    w, h, c = imgScanSmall.shape
    imgScanSmall = cv2.resize(imgScanSmall, (h * 2, w * 2))
    imgScanSmall = cv2.cvtColor(imgScanSmall, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgScanSmall)
    encodeCurFrame = face_recognition.face_encodings(imgScanSmall, facesCurFrame)

    #  finding all the matches
    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        print("match: ", matches, "\nfaceDis: ", faceDis, "\nmatchIndex: ", matchIndex)
        if matches[matchIndex]:
            name = className[matchIndex].upper()
            # print("WebCam Person Name: ", name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1, x2, y2, x1
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name.split('_')[0], (x1 + 2, y2 - 2), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            markAttendance(name)

    cv2.imshow('System Image', img)
    cv2.waitKey(0)


# img = cv2.imread("testImages/cskG1.png")
# w, h, c = img.shape
# imgResize = cv2.resize(img, (h*2, w*2))
imageTesting("cskG2.png")

# t_Image = ['cskG2.png', 'cskG3.png', 'cskG4.jpg', 'cskG5.jpeg', 'cskGroupImage.jpg']
# for i in range(0, len(t_Image)):
#     print(i, ' : Processing')
#     imageTesting(t_Image[i])
#     print('Done.')
