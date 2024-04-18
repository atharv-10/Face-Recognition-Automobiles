import cv2

# Load the video capture
cap = cv2.VideoCapture(0)

# Create the background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Loop over each frame in the video
while True:
    # Read the next frame
    ret, frame = cap.read()

    # Apply the background subtractor to the frame
    fgmask = fgbg.apply(frame)

    # Threshold the foreground mask to only keep the moving pixels
    _, thresh = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over each contour
    for c in contours:
        # If the contour is large enough, consider it a body movement
        if cv2.contourArea(c) > 1000:
            # Draw a bounding box around the contour
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Body Movement Detection", frame)

    # Break the loop if the "q" key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and destroy the windows
cap.release()
cv2.destroyAllWindows()

