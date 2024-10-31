import cv2
import mediapipe as mp

# Initialize MediaPipe Objectron and Drawing Utilities
mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

# Initialize webcam
cap = cv2.VideoCapture(0)

# Select the object you want to detect: "Cup", "Chair", "Shoe", "Camera"
with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=3,  # Max number of objects to detect
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5,
                            model_name='Shoe') as objectron:  # Change 'Cup' to 'Chair', 'Shoe', or 'Camera'
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to RGB (since MediaPipe requires RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Objectron to detect objects
        results = objectron.process(rgb_frame)

        # If objects detected
        if results.detected_objects:
            for detected_object in results.detected_objects:
                # Draw the 3D bounding box landmarks on the original frame
                mp_drawing.draw_landmarks(
                    frame, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)

                # Draw the 3D axis for object orientation
                mp_drawing.draw_axis(frame, detected_object.rotation, detected_object.translation)

        # Show the output
        cv2.imshow('Objectron 3D Object Detection', frame)

        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()