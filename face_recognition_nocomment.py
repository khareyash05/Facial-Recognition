import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# cascade is a series of filters
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# detect faces and inside these faces only we detect eyes

def detect(gray,frame):
    faces = face_cascade.detectMultiScale(gray,1.3,5) 
    # atleast 5 neighbour zones should be accepted and scaling the image by 1.3
    # faces is tuple of X,Y,W,H(coordinates of upper left and width and height of rectangle)
    for (x,y,w,h) in faces :
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w] # creating a region of interest
        roi_color = frame[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray,1.1,3)
        for (ex,ey,ew,eh) in eyes :
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2)
    return frame

video_capture = cv2.VideoCapture(0)

while True :
    _,frame = video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas = detect(gray,frame)
    cv2.imshow('Video',canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()