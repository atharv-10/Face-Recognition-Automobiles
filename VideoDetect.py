import dlib
import cv2

# Initialize the camera
camera = cv2.VideoCapture(0)

# Initialize the facial landmark detector
detector = dlib.get_frontal_face_detector()

while True:
    # Get the current frame from the cameraB
    ret, frame = camera.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = detector(gray, 1)

    # Draw a rectangle around the detected faces
    for face in faces:
        x1, y1, x2, y2, w, h = (face.left(), face.top(), face.right(), face.bottom(), face.width(), face.height())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Face Detection", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
camera.release()
cv2.destroyAllWindows()
