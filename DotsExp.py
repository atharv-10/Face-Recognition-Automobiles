import dlib
import cv2

# Initialize the camera
camera = cv2.VideoCapture(0)

# Initialize the facial landmark predictor
predictor = dlib.shape_predictor("C:/Nxt softwares/Python 38/shape_predictor_68_face_landmarks.dat")

while True:
    # Get the current frame from the camera
    ret, frame = camera.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = dlib.get_frontal_face_detector()(gray, 1)

    # Draw a rectangle and dots on the detected faces
    for face in faces:
        # Draw a rectangle around the face
        x1, y1, x2, y2, w, h = (face.left(), face.top(), face.right(), face.bottom(), face.width(), face.height())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Get the facial landmarks
        landmarks = predictor(gray, face)

        # Draw dots on the facial landmarks
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    # Show the frame
    cv2.imshow("Face Detection", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
camera.release()
cv2.destroyAllWindows()
