import cv2

# Initialize video capture (webcam or robot's camera)
cap = cv2.VideoCapture(0)

# Create tracker object
tracker = cv2.TrackerCSRT_create()

# Read the first frame to select ROI (region of interest)
ret, frame = cap.read()

roi = cv2.selectROI(frame, False)

tracker.init(frame, roi)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Update tracker on the frame
    ret, roi = tracker.update(frame)

    if ret:
        # Draw the tracked object
        (x, y, w, h) = tuple(map(int, roi))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
