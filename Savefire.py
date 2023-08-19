import cv2
from playsound import playsound

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc,30.0,(640,480))
fire_cascade = cv2.CascadeClassifier('fire_detection.xml')  # Machine Learning Based approach
cap = cv2.VideoCapture(0)  # captures video from webcam

while (True):
    ret, frame = cap.read()  # Returns boolean value if frame is read correctly
    out.write(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converts BGR frame to GRAY
    fire = fire_cascade.detectMultiScale(gray, 1.2, 5)  # returns rectangle of size (x,y,w,h)

    for (x, y, w, h) in fire:  # Searching in rectangle if size(x,y,w,h)
        cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 0, 255), 5)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        print("fire is detected")
        playsound('Alarm Sound.mp3')

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()