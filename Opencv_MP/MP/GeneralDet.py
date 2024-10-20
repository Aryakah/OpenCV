# STEP 1: Import the necessary modules.
import numpy as np
import cv2  # Ensure OpenCV is imported for image processing
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path='Opencv_MP/MP/Model/efficientdet_lite2.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# STEP 3: Load the input image.
image_file_path = 'Model/pet-3157961_640.jpg'  # Ensure the path is correct
image = mp.Image.create_from_file(image_file_path)

# STEP 4: Detect objects in the input image.
detection_result = detector.detect(image)

# STEP 5: Define the visualize function to annotate the image.
def visualize(image, detection_result):
    for detection in detection_result.detections:
        bboxC = detection.bounding_box  # Get the bounding box
        # Convert bounding box coordinates to integers
        x_min = int(bboxC.origin_x)
        y_min = int(bboxC.origin_y)
        x_max = int(bboxC.origin_x + bboxC.width)
        y_max = int(bboxC.origin_y + bboxC.height)
        # Draw the bounding box on the image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    return image

# Call the visualize function after defining it
image_copy = np.copy(image.numpy_view())
annotated_image = visualize(image_copy, detection_result)

# Convert the image to RGB for displaying
rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

# Display the annotated image using OpenCV
cv2.imshow('Annotated Image', rgb_annotated_image)
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close the displayed window
