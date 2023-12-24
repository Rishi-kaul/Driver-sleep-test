""" Driver drowiness test 
EAR = sum of vertical distance 
     ----------------------------
     2 * sum of hoetizontal disstance 
"""
import cv2
from imutils import face_utils
import dlib
from scipy.spatial import distance

def EAR(eye): #eye aspect ratio
    A = distance.euclidean(eye[1],eye[5])
    B = distance.euclidean(eye[2],eye[4])
    C= distance.euclidean(eye[0],eye[3])
    ear = (A+B)/(2.0*C)
    return ear
(lstart,lend) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rstart,rend) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
""" why we need ear as it remains constant but as eyes cloes its frame drops  value of a+b decreses """




detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks .dat")

cap = cv2.VideoCapture(0)

while True:
    ret , frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    subjects = detect(gray,0 )
    for  subject in subjects:
        shape = predict(gray,subjects)
        shape = face_utils.shape_to_np(shape)
        Lefteye = shape[lstart:lend]
        Righteye = shape[rstart:rend]
        left_ear = EAR(Lefteye)
        right_ear =EAR(Righteye)
        ear = (left_ear +right_ear)/2.0 


 
    cv2.imshow("Frame",frame)
    cv2.waitKey(0)




    