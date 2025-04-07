import threading
import cv2
from deepface import DeepFace

# Load webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False
lock = threading.Lock()  # Prevent multiple threads from overlapping
reference_img = cv2.imread("reference.jpg")

# Use a faster model like 'Facenet'
MODEL_NAME = "Facenet"  # Options: VGG-Face, Facenet, OpenFace, DeepFace, DeepID, ArcFace, Dlib, SFace

def check_face(frame):
    global face_match
    with lock:
        try:
            result = DeepFace.verify(frame, reference_img.copy(), model_name=MODEL_NAME, enforce_detection=False)
            print(result)  # Print for debugging
            face_match = result['verified']
        except Exception as e:
            print("Error during verification:", e)
            face_match = False

while True:
    ret, frame = cap.read()
    if ret:
        if counter % 60 == 0:  # Match every 2 seconds (reduce CPU load)
            threading.Thread(target=check_face, args=(frame.copy(),)).start()
        
        counter += 1

        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
