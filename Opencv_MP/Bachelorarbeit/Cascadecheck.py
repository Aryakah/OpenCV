import cv2

# Load the Haar cascade
face_cascade = cv2.CascadeClassifier('C:/Users/ichverdienees/Desktop/OpenCV-Dashcam-Car-Detection-master/cascade_dir/cascade.xml')

# Read an image
image = cv2.imread('path_to_your_image.jpg')  # Replace with your image path
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Show the output
cv2.imshow('Image with Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

