import cv2

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam or video feed
cap = cv2.VideoCapture(0)

# Get the dimensions of the camera frame
frame_width = int(cap.get(3))  # Width of the frame
frame_height = int(cap.get(4))  # Height of the frame

# Center coordinates
center_x = frame_width // 2
center_y = frame_height // 2

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Initialize variables for the largest face (most centered face)
    face_closest_to_center = None
    face_center_offset = float('inf')

    # Loop over all detected faces
    for (x, y, w, h) in faces:
        # Calculate the center of the face
        face_center_x = x + w // 2
        face_center_y = y + h // 2

        # Calculate the offset of the face from the center of the frame
        offset_x = abs(center_x - face_center_x)
        offset_y = abs(center_y - face_center_y)
        total_offset = offset_x + offset_y

        # Track the face that is closest to the center of the frame
        if total_offset < face_center_offset:
            face_closest_to_center = (x, y, w, h)
            face_center_offset = total_offset

    # If a face is detected, draw a rectangle around it
    if face_closest_to_center:
        (x, y, w, h) = face_closest_to_center
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Optionally: Draw crosshairs at the center of the frame for alignment
        cv2.line(frame, (center_x, 0), (center_x, frame_height), (0, 255, 0), 1)
        cv2.line(frame, (0, center_y), (frame_width, center_y), (0, 255, 0), 1)

    # Display the resulting frame
    cv2.imshow('Face Look Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
